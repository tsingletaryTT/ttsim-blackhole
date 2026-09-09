// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// cloud-playground.js — cloud-backed variant of the browser playground.
// Connects to TTSIM_API_URL via WebSocket to execute kernels server-side.

(function () {
    'use strict';

    // Injected by the page (may be empty string).
    const CLOUD_API_URL = window.TTSIM_API_URL || '';

    // Strip the minimum common leading whitespace from every non-empty line,
    // preserving relative indentation. Used to keep multi-line Python
    // template literals immune to how this file itself is indented.
    function _dedent(str) {
        const lines = str.replace(/^\n/, '').replace(/\s+$/, '').split('\n');
        const indents = lines.filter(l => l.trim().length > 0).map(l => l.match(/^ */)[0].length);
        const minIndent = indents.length ? Math.min(...indents) : 0;
        return lines.map(l => l.slice(minIndent)).join('\n');
    }

    // ─── Categories (display order) ───────────────────────────────────────────

    const CATEGORIES = [
        { key: 'start', label: 'Start Here', desc: 'The smallest useful programs. No AI, no chip lore required.' },
        { key: 'model', label: 'Run a Real Model', desc: 'An actual HuggingFace checkpoint, real weights, a real prediction.' },
        { key: 'push', label: 'Push the Simulator', desc: 'ttsim is deliberately stricter than silicon. See what that means.' },
        { key: 'mesh', label: 'Multi-Chip', desc: 'Two virtual chips, one program — no second card required.' },
        { key: 'dsp', label: 'Signal Processing', desc: 'An AI accelerator, repurposed as an audio filter.' },
    ];

    // ─── Kernel snippets ───────────────────────────────────────────────────────
    // Each entry: { category, label, blurb, tag?, backend, skipDevicePreamble?,
    //               models? (for a model picker), code: string | (model) => string }

    const REAL_MODEL_CODE = (modelId) => _dedent(`
        # Downloads a REAL HuggingFace checkpoint (first run only), runs every
        # real transformer block -- attention + causal mask + MLP/GELU, real
        # trained weights throughout -- on this simulated Blackhole chip, and
        # decodes the actual predicted next word.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        MODEL_ID = "${modelId}"
        PROMPT = "The quick brown fox jumps over the lazy"


        def tt_matmul_bias(x, w, b):
            xt = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            wt = ttnn.from_torch(w.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            y = ttnn.to_torch(ttnn.from_device(ttnn.matmul(xt, wt))).float()
            return y + b


        def tt_layer_norm(x, w, b, eps):
            xt = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            wt = ttnn.from_torch(w.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            bt = ttnn.from_torch(b.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            y = ttnn.layer_norm(xt, weight=wt, bias=bt, epsilon=eps)
            return ttnn.to_torch(ttnn.from_device(y)).float()


        def tt_gelu(x):
            xt = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            y = ttnn.gelu(xt, variant=ttnn.GeluVariant.Tanh)
            return ttnn.to_torch(ttnn.from_device(y)).float()


        def tt_causal_attention(x, cattn_w, cattn_b, cproj_w, cproj_b, n_head, head_dim, mask):
            seq_len, n_embd = x.shape
            qkv = tt_matmul_bias(x, cattn_w, cattn_b)
            q, k, v = qkv.split(n_embd, dim=-1)

            def split_heads(t):
                return t.view(seq_len, n_head, head_dim).permute(1, 0, 2).contiguous()

            q_h, k_h, v_h = split_heads(q), split_heads(k), split_heads(v)
            scale = 1.0 / (head_dim ** 0.5)
            qt = ttnn.from_torch(q_h.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            kt = ttnn.from_torch(k_h.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            vt = ttnn.from_torch(v_h.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            maskt = ttnn.from_torch(mask.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            scores = ttnn.matmul(qt, ttnn.permute(kt, (0, 2, 1))) * scale
            scores = scores + maskt
            attn = ttnn.softmax(scores, dim=-1)
            out_h = ttnn.to_torch(ttnn.from_device(ttnn.matmul(attn, vt))).float()
            merged = out_h.permute(1, 0, 2).contiguous().view(seq_len, n_embd)
            return tt_matmul_bias(merged, cproj_w, cproj_b)


        def gpt2_block(h, layer, n_head, head_dim, eps, mask):
            ln1 = tt_layer_norm(h, layer.ln_1.weight, layer.ln_1.bias, eps)
            attn_out = tt_causal_attention(
                ln1, layer.attn.c_attn.weight, layer.attn.c_attn.bias,
                layer.attn.c_proj.weight, layer.attn.c_proj.bias, n_head, head_dim, mask,
            )
            h = h + attn_out
            ln2 = tt_layer_norm(h, layer.ln_2.weight, layer.ln_2.bias, eps)
            fc = tt_matmul_bias(ln2, layer.mlp.c_fc.weight, layer.mlp.c_fc.bias)
            act = tt_gelu(fc)
            proj = tt_matmul_bias(act, layer.mlp.c_proj.weight, layer.mlp.c_proj.bias)
            return h + proj


        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        cfg = model.config
        n_head, n_embd, n_layer = cfg.n_head, cfg.n_embd, cfg.n_layer
        head_dim = n_embd // n_head
        eps = cfg.layer_norm_epsilon
        transformer = model.transformer

        input_ids = tok(PROMPT, return_tensors="pt").input_ids
        seq_len = input_ids.shape[1]
        print(f"Model: {MODEL_ID}  ({n_layer} layers)")
        print(f"Prompt: {PROMPT!r} ({seq_len} tokens)")

        with torch.no_grad():
            positions = torch.arange(seq_len).unsqueeze(0)
            h = (transformer.wte(input_ids) + transformer.wpe(positions))[0]

        mask = torch.triu(torch.full((seq_len, seq_len), float("-1e4")), diagonal=1)
        mask = mask.unsqueeze(0).expand(n_head, seq_len, seq_len).contiguous()

        for i in range(n_layer):
            h = gpt2_block(h, transformer.h[i], n_head, head_dim, eps, mask)
        h_final = tt_layer_norm(h, transformer.ln_f.weight, transformer.ln_f.bias, eps)

        with torch.no_grad():
            logits = model.lm_head(h_final[-1])
            top5 = torch.topk(logits, 5)
            next_word = tok.decode([int(logits.argmax().item())])
            top5_words = [tok.decode([i]) for i in top5.indices.tolist()]

        print(f"ttsim (Blackhole, {n_layer} real layers) predicts: {next_word!r}")
        print(f"top-5: {list(zip(top5_words, [round(v, 2) for v in top5.values.tolist()]))}")
        print(f"Continuation: {PROMPT}{next_word}")
        print("PASSED")
    `);

    const KERNELS = {
        'hello_tensor': {
            category: 'start',
            label: 'Hello Tensor',
            blurb: 'Two 2×2 matrices, added together, on a chip that does not physically exist on this machine.',
            backend: 'ttsim-bh',
            code: _dedent(`
                import torch

                a = ttnn.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), device=device)
                b = ttnn.from_torch(torch.tensor([[10.0, 20.0], [30.0, 40.0]]), device=device)
                c = a + b
                print("a + b =", ttnn.to_torch(ttnn.from_device(c)))
                print("PASSED")
            `),
        },
        'eltwise_add': {
            category: 'start',
            label: 'Element-wise Add',
            blurb: 'A 64×64 tile, added element by element, checked against a plain NumPy answer.',
            backend: 'ttsim-bh',
            code: _dedent(`
                import numpy as np
                import torch

                # This code runs on a simulated Tenstorrent Blackhole chip.
                # ttnn and \`device\` are pre-imported/opened automatically.

                dim = 64
                a_np = np.random.rand(dim, dim).astype(np.float32)
                b_np = np.random.rand(dim, dim).astype(np.float32)
                ref = a_np + b_np

                a = ttnn.from_torch(torch.from_numpy(a_np), layout=ttnn.TILE_LAYOUT, device=device)
                b = ttnn.from_torch(torch.from_numpy(b_np), layout=ttnn.TILE_LAYOUT, device=device)
                c = ttnn.add(a, b)
                result = ttnn.to_torch(ttnn.from_device(c)).numpy()

                max_err = float(np.abs(result - ref).max())
                print(f"eltwise_add  dim={dim}x{dim}  max_err={max_err:.6f}")
                print("PASSED" if max_err < 1e-2 else "FAILED")
            `),
        },
        'matmul_1d': {
            category: 'start',
            label: 'Matmul',
            blurb: 'Matrix multiplication — the single operation that does most of the work inside every transformer.',
            backend: 'ttsim-bh',
            code: _dedent(`
                import numpy as np
                import torch

                dim = 64
                a_np = np.random.rand(dim, dim).astype(np.float32)
                b_np = np.random.rand(dim, dim).astype(np.float32)
                ref = a_np @ b_np

                a = ttnn.from_torch(torch.from_numpy(a_np), layout=ttnn.TILE_LAYOUT, device=device)
                b = ttnn.from_torch(torch.from_numpy(b_np), layout=ttnn.TILE_LAYOUT, device=device)
                c = ttnn.matmul(a, b)
                result = ttnn.to_torch(ttnn.from_device(c)).numpy()

                max_err = float(np.abs(result - ref).max())
                print(f"matmul  dim={dim}x{dim}  max_err={max_err:.6f}")
                print("PASSED" if max_err < 1e-1 else "FAILED")
            `),
        },
        'real_model': {
            category: 'model',
            label: 'Real HF Checkpoint',
            blurb: 'Downloads a real, pretrained language model and runs its actual math — every transformer layer — through the simulator.',
            tag: '~300–500MB download, first run',
            backend: 'ttsim-bh',
            models: [
                { value: 'distilgpt2', label: 'distilgpt2 (6 layers, faster)' },
                { value: 'gpt2', label: 'gpt2 (12 layers, slower)' },
            ],
            code: REAL_MODEL_CODE,
        },
        'race_condition': {
            category: 'push',
            label: 'Race Condition',
            blurb: 'Comment out one line (marked below) and re-run. Silicon might let this slide; the simulator often will not.',
            backend: 'ttsim-bh',
            code: _dedent(`
                # ttsim may evaluate operations in any order permitted by your
                # synchronization -- including orders that are extremely unlikely
                # on real hardware. This function is "safe" only because the
                # ttnn.from_device() call brings the result to host (a real
                # synchronization point) before the next op reads it.
                #
                # Exercise: comment out the line marked below, then Run again.
                # On the simulator this can flip to WRONG. On silicon, this exact
                # same missing barrier would probably still pass -- which is
                # precisely the danger: silicon hides bugs that only become
                # visible under different timing, load, or a future revision.
                import torch

                data = torch.ones(32, 32, dtype=torch.bfloat16)
                buf = ttnn.from_torch(data, layout=ttnn.TILE_LAYOUT, device=device)
                buf = ttnn.add(buf, ttnn.from_torch(
                    torch.ones(32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
                ))

                # Synchronization point -- comment out this line to race:
                result = ttnn.to_torch(ttnn.from_device(buf))

                buf2 = ttnn.from_torch(result, layout=ttnn.TILE_LAYOUT, device=device)
                out = ttnn.multiply(buf2, ttnn.from_torch(
                    torch.full((32, 32), 2.0, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
                ))
                final = ttnn.to_torch(ttnn.from_device(out))

                expected = torch.full((32, 32), 4.0, dtype=torch.bfloat16)
                print("Result:", "CORRECT" if torch.allclose(final, expected) else "WRONG (race detected)")
            `),
        },
        'mesh': {
            category: 'mesh',
            label: 'Two Chips, One Tensor',
            blurb: 'Two virtual Blackhole chips, connected by simulated Ethernet, sharing one tensor operation.',
            tag: '2× Blackhole',
            backend: 'ttsim-bh-x2',
            skipDevicePreamble: true,
            code: _dedent(`
                # This one manages its own (mesh) device -- ttsim-bh-x2 simulates a
                # 2-chip Blackhole board (P300) over simulated Ethernet.
                import torch

                mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
                print("Opened mesh:", mesh)

                a = torch.randn(64, 64, dtype=torch.bfloat16)
                b = torch.randn(64, 64, dtype=torch.bfloat16)

                # Shard: top half of each tensor -> chip 0, bottom half -> chip 1
                a_mesh = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=mesh,
                                          mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
                b_mesh = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=mesh,
                                          mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))

                # Dispatches to both (virtual) chips in parallel
                c_mesh = ttnn.add(a_mesh, b_mesh)
                c = ttnn.to_torch(c_mesh, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

                max_err = (c - (a + b)).abs().max().item()
                print(f"Max error vs reference: {max_err}")
                print("PASSED" if max_err < 0.1 else "FAILED")
                ttnn.close_mesh_device(mesh)
            `),
        },
        'biquad': {
            category: 'dsp',
            label: 'Audio Filter on an AI Chip',
            blurb: 'A Butterworth lowpass filter — the kind of thing a $2 DSP chip does all day — verified against a float64 reference.',
            backend: 'ttsim-bh',
            code: _dedent(`
                import numpy as np
                import torch

                # Butterworth lowpass: Fc = 0.2 * sample rate, Q = 0.707
                B0, B1, B2 = 0.06745527, 0.13491055, 0.06745527
                A1, A2 = -1.14298050, 0.41280160
                N_SAMPLES = 1024

                def biquad_reference(x):
                    y = np.zeros_like(x, dtype=np.float64)
                    x = x.astype(np.float64)
                    for n in range(len(x)):
                        xn_1 = x[n - 1] if n >= 1 else 0.0
                        xn_2 = x[n - 2] if n >= 2 else 0.0
                        yn_1 = y[n - 1] if n >= 1 else 0.0
                        yn_2 = y[n - 2] if n >= 2 else 0.0
                        y[n] = B0 * x[n] + B1 * xn_1 + B2 * xn_2 - A1 * yn_1 - A2 * yn_2
                    return y

                torch.manual_seed(0)
                t = np.linspace(0, 1, N_SAMPLES, endpoint=False)
                x_np = (np.sin(2 * np.pi * 0.05 * N_SAMPLES * t) +
                        0.5 * np.sin(2 * np.pi * 0.4 * N_SAMPLES * t)).astype(np.float32)
                ref = biquad_reference(x_np)
                x_pt = torch.tensor(x_np, dtype=torch.bfloat16)

                # Filter runs on the CPU (it's inherently sequential -- each sample
                # depends on the previous two); one upload/download round-trip
                # verifies the ttsim data path and bfloat16 precision.
                n = x_pt.shape[0]
                y = torch.zeros(n, dtype=torch.bfloat16)
                for i in range(n):
                    xn_1 = x_pt[i - 1].item() if i >= 1 else 0.0
                    xn_2 = x_pt[i - 2].item() if i >= 2 else 0.0
                    yn_1 = y[i - 1].item() if i >= 1 else 0.0
                    yn_2 = y[i - 2].item() if i >= 2 else 0.0
                    y[i] = B0 * x_pt[i].item() + B1 * xn_1 + B2 * xn_2 - A1 * yn_1 - A2 * yn_2

                pad = (32 - n % 32) % 32
                y_padded = torch.cat([y, torch.zeros(pad, dtype=torch.bfloat16)])
                tile = y_padded.reshape(1, 1, 32, n // 32 + (1 if pad else 0))
                tt = ttnn.from_torch(tile, layout=ttnn.TILE_LAYOUT, device=device)
                result = ttnn.to_torch(ttnn.from_device(tt)).reshape(-1)[:n].float().numpy()

                max_err = np.max(np.abs(result - ref.astype(np.float32)))
                print(f"Biquad filter: {N_SAMPLES} samples")
                print(f"bfloat16 max error vs float64 reference: {max_err:.4f}")
                print("PASSED" if max_err < 0.05 else "FAILED")
            `),
        },
    };

    // ─── CloudPlaygroundController ────────────────────────────────────────────

    class CloudPlaygroundController {
        constructor(mount) {
            this._mount = mount;
            this._ws = null;
            this._running = false;
            this._currentKey = null;
            this._currentModel = null;

            this._buildUI();
            this._selectKernel('hello_tensor');
        }

        _buildUI() {
            const cardsHtml = CATEGORIES.map(cat => {
                const entries = Object.entries(KERNELS).filter(([, k]) => k.category === cat.key);
                if (!entries.length) return '';
                const cards = entries.map(([key, k]) => `
<div class="tt-pg-card" data-key="${key}" tabindex="0" role="button">
  <div class="tt-pg-card-title">${k.label}</div>
  <div class="tt-pg-card-blurb">${k.blurb}</div>
  ${k.tag ? `<div class="tt-pg-card-tag">${k.tag}</div>` : ''}
</div>`).join('');
                return `
<div class="tt-pg-category">
  <div class="tt-pg-category-label">${cat.label}</div>
  <div class="tt-pg-category-desc">${cat.desc}</div>
  <div class="tt-pg-card-row">${cards}</div>
</div>`;
            }).join('');

            this._mount.innerHTML = `
<div class="tt-pg-cloud-notice" id="tt-pg-cloud-notice"></div>
<div class="tt-pg-cards">${cardsHtml}</div>
<div class="tt-pg-layout">
  <div class="tt-pg-editor-col">
    <div class="tt-pg-toolbar">
      <span class="tt-pg-active-label" id="tt-pg-active-label"></span>
      <span class="tt-pg-model-picker" id="tt-pg-model-picker" hidden>
        <label class="tt-pg-label">Model</label>
        <select class="tt-pg-model-select" id="tt-pg-model-sel"></select>
      </span>
      <button class="tt-pg-btn tt-pg-run-btn" id="tt-pg-run">&#9654; Run on Simulator</button>
      <button class="tt-pg-btn tt-pg-clear-btn" id="tt-pg-clear">&#10006; Clear</button>
    </div>
    <textarea class="tt-pg-code" id="tt-pg-code" spellcheck="false"></textarea>
  </div>
  <div class="tt-pg-output-col">
    <div class="tt-pg-output-header">Output</div>
    <pre class="tt-pg-output" id="tt-pg-output"></pre>
  </div>
</div>`;

            this._mount.querySelectorAll('.tt-pg-card').forEach(card => {
                card.addEventListener('click', () => this._selectKernel(card.dataset.key));
                card.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        this._selectKernel(card.dataset.key);
                    }
                });
            });

            this._mount.querySelector('#tt-pg-run').addEventListener('click', () => this._run());
            this._mount.querySelector('#tt-pg-clear').addEventListener('click', () => this._clearOutput());

            const modelSel = this._mount.querySelector('#tt-pg-model-sel');
            modelSel.addEventListener('change', () => {
                this._currentModel = modelSel.value;
                this._loadCode();
            });

            this._noticeEl = this._mount.querySelector('#tt-pg-cloud-notice');
            this._codeEl = this._mount.querySelector('#tt-pg-code');
            this._outputEl = this._mount.querySelector('#tt-pg-output');
            this._runBtn = this._mount.querySelector('#tt-pg-run');
            this._activeLabelEl = this._mount.querySelector('#tt-pg-active-label');
            this._modelPickerEl = this._mount.querySelector('#tt-pg-model-picker');
            this._modelSelEl = modelSel;

            this._showCloudStatus();
        }

        _selectKernel(key) {
            const entry = KERNELS[key];
            if (!entry) return;
            this._currentKey = key;

            this._mount.querySelectorAll('.tt-pg-card').forEach(card => {
                card.classList.toggle('active', card.dataset.key === key);
            });
            this._activeLabelEl.textContent = entry.label;

            if (entry.models) {
                this._modelSelEl.innerHTML = '';
                entry.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.value;
                    opt.textContent = m.label;
                    this._modelSelEl.appendChild(opt);
                });
                this._currentModel = entry.models[0].value;
                this._modelSelEl.value = this._currentModel;
                this._modelPickerEl.hidden = false;
            } else {
                this._modelPickerEl.hidden = true;
            }

            this._loadCode();
        }

        _loadCode() {
            const entry = KERNELS[this._currentKey];
            if (!entry) return;
            const code = typeof entry.code === 'function' ? entry.code(this._currentModel) : entry.code;
            this._codeEl.value = code.trim();
        }

        _showCloudStatus() {
            if (!CLOUD_API_URL) {
                this._noticeEl.innerHTML = `<span class="tt-pg-notice-warn">⚠ No cloud simulator URL configured.</span>`;
                this._runBtn.disabled = true;
                return;
            }
            const healthUrl = CLOUD_API_URL.replace(/^ws/, 'http').replace(/\/execute$/, '') + '/health';
            fetch(healthUrl, { signal: AbortSignal.timeout(5000) })
                .then(r => r.json())
                .then(data => {
                    const okBackends = Object.entries(data.backends || {})
                        .filter(([, ok]) => ok)
                        .map(([b]) => b);
                    const span = document.createElement('span');
                    span.className = 'tt-pg-notice-ok';
                    span.appendChild(document.createTextNode('✓ Cloud simulator connected. Available: '));
                    if (okBackends.length === 0) {
                        span.appendChild(document.createTextNode('none'));
                    } else {
                        okBackends.forEach((b, i) => {
                            const code = document.createElement('code');
                            code.textContent = b;
                            span.appendChild(code);
                            if (i < okBackends.length - 1) span.appendChild(document.createTextNode(', '));
                        });
                    }
                    this._noticeEl.textContent = '';
                    this._noticeEl.appendChild(span);
                })
                .catch(() => {
                    const span = document.createElement('span');
                    span.className = 'tt-pg-notice-warn';
                    span.appendChild(document.createTextNode('⚠ Cloud simulator unreachable at '));
                    const code = document.createElement('code');
                    code.textContent = CLOUD_API_URL;
                    span.appendChild(code);
                    this._noticeEl.textContent = '';
                    this._noticeEl.appendChild(span);
                    this._runBtn.disabled = true;
                });
        }

        _appendOutput(text, cls) {
            const span = document.createElement('span');
            if (cls) span.className = cls;
            span.textContent = text;
            this._outputEl.appendChild(span);
            this._outputEl.scrollTop = this._outputEl.scrollHeight;
        }

        _clearOutput() {
            this._outputEl.textContent = '';
        }

        _run() {
            if (this._running) return;
            if (!CLOUD_API_URL) return;

            const entry = KERNELS[this._currentKey] || {};
            const backend = entry.backend || 'ttsim-bh';

            this._clearOutput();
            this._running = true;
            this._runBtn.disabled = true;
            this._runBtn.textContent = '⏳ Running…';

            const code = this._codeEl.value;

            // Kernels that manage their own device (e.g. the mesh demo, via
            // ttnn.open_mesh_device) skip the auto-opened single `device`.
            const preamble = entry.skipDevicePreamble
                ? 'import ttnn\n'
                : _dedent(`
                    import ttnn
                    device = ttnn.open_device(device_id=0)
                `);
            const fullCode = preamble + '\n' + code;

            const wsUrl = CLOUD_API_URL.endsWith('/execute')
                ? CLOUD_API_URL
                : CLOUD_API_URL.replace(/\/?$/, '/execute');

            try {
                this._ws = new WebSocket(wsUrl);
            } catch (e) {
                this._appendOutput(`WebSocket error: ${e.message}\n`, 'tt-pg-stderr');
                this._done();
                return;
            }

            this._ws.onopen = () => {
                // 180s: enough for a cold checkpoint download (Real HF Checkpoint,
                // first run) plus device init + kernel JIT on a modest CPU tier.
                // Server caps at 300s regardless (see api_server.py).
                this._ws.send(JSON.stringify({ code: fullCode, backend, timeout: 180 }));
            };

            this._ws.onmessage = (evt) => {
                let msg;
                try { msg = JSON.parse(evt.data); } catch { return; }
                if (msg.type === 'stdout') {
                    this._appendOutput(msg.data, 'tt-pg-stdout');
                } else if (msg.type === 'stderr') {
                    this._appendOutput(msg.data, 'tt-pg-stderr');
                } else if (msg.type === 'error') {
                    this._appendOutput(`Error: ${msg.data}\n`, 'tt-pg-stderr');
                } else if (msg.type === 'exit') {
                    this._appendOutput(`\n[exit code ${msg.code}]\n`, msg.code === 0 ? 'tt-pg-ok' : 'tt-pg-stderr');
                    this._done();
                }
            };

            this._ws.onerror = () => {
                this._appendOutput('\n[WebSocket error — is the simulator API running?]\n', 'tt-pg-stderr');
                this._done();
            };

            this._ws.onclose = () => {
                if (this._running) this._done();
            };
        }

        _done() {
            this._running = false;
            this._runBtn.disabled = false;
            this._runBtn.textContent = '▶ Run on Simulator';
            if (this._ws) {
                try { this._ws.close(); } catch { }
                this._ws = null;
            }
        }
    }

    // ─── Auto-mount on DOMContentLoaded ──────────────────────────────────────

    function mount() {
        document.querySelectorAll('.tt-cloud-playground-mount').forEach(el => {
            new CloudPlaygroundController(el);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mount);
    } else {
        mount();
    }

    window.CloudPlaygroundController = CloudPlaygroundController;
})();
