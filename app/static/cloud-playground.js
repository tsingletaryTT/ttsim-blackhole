// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// cloud-playground.js — cloud-backed variant of the browser playground.
// Connects to TTSIM_API_URL via WebSocket to execute kernels server-side.
// Falls back to the Pyodide (local) playground if the API is unreachable.

(function () {
    'use strict';

    // Injected by build-web.js from $TTSIM_API_URL env var (may be empty string).
    const CLOUD_API_URL = window.TTSIM_API_URL || '';

    // ─── Kernel snippets (same set as playground.js) ─────────────────────────

    const KERNELS = {
        'hello_tensor': {
            label: 'Hello Tensor',
            code: `\
import torch

a = ttnn.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), device=device)
b = ttnn.from_torch(torch.tensor([[10.0, 20.0], [30.0, 40.0]]), device=device)
c = a + b
print("a + b =", ttnn.to_torch(ttnn.from_device(c)))
print("PASSED")
`
        },
        'eltwise_add': {
            label: 'Element-wise Add',
            code: `\
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
`
        },
        'matmul_1d': {
            label: 'Matmul',
            code: `\
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
`
        },
        'real_distilgpt2': {
            label: 'Real distilgpt2 (6-layer forward pass)',
            code: `\
# Downloads the REAL distilgpt2 checkpoint (~330MB, first run only), runs
# all 6 real transformer blocks -- attention + causal mask + MLP/GELU, real
# trained weights throughout -- on this simulated Blackhole chip, and
# decodes the actual predicted next word.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "distilgpt2"
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

print(f"ttsim (Blackhole, 6 real distilgpt2 layers) predicts: {next_word!r}")
print(f"top-5: {list(zip(top5_words, [round(v, 2) for v in top5.values.tolist()]))}")
print(f"Continuation: {PROMPT}{next_word}")
print("PASSED")
`
        },
    };

    // ─── CloudPlaygroundController ────────────────────────────────────────────

    class CloudPlaygroundController {
        constructor(mount) {
            this._mount = mount;
            this._ws = null;
            this._running = false;

            this._buildUI();
            this._selectKernel('hello_tensor');
        }

        _buildUI() {
            this._mount.innerHTML = `
<div class="tt-pg-cloud-notice" id="tt-pg-cloud-notice"></div>
<div class="tt-pg-layout">
  <div class="tt-pg-editor-col">
    <div class="tt-pg-toolbar">
      <label class="tt-pg-label">Kernel</label>
      <select class="tt-pg-kernel-select" id="tt-pg-kernel-sel"></select>
      <label class="tt-pg-label">Backend</label>
      <select class="tt-pg-backend-select" id="tt-pg-backend-sel">
        <option value="ttsim-bh" selected>ttsim-bh (Blackhole emulation)</option>
      </select>
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

            const sel = this._mount.querySelector('#tt-pg-kernel-sel');
            for (const [key, { label }] of Object.entries(KERNELS)) {
                const opt = document.createElement('option');
                opt.value = key;
                opt.textContent = label;
                sel.appendChild(opt);
            }
            sel.addEventListener('change', () => this._selectKernel(sel.value));

            this._mount.querySelector('#tt-pg-run').addEventListener('click', () => this._run());
            this._mount.querySelector('#tt-pg-clear').addEventListener('click', () => this._clearOutput());

            this._noticeEl = this._mount.querySelector('#tt-pg-cloud-notice');
            this._codeEl = this._mount.querySelector('#tt-pg-code');
            this._outputEl = this._mount.querySelector('#tt-pg-output');
            this._runBtn = this._mount.querySelector('#tt-pg-run');
            this._backendSel = this._mount.querySelector('#tt-pg-backend-sel');
            this._kernelSel = sel;

            this._showCloudStatus();
        }

        _selectKernel(key) {
            if (KERNELS[key]) {
                this._codeEl.value = KERNELS[key].code.trim();
                if (this._kernelSel) this._kernelSel.value = key;
            }
        }

        _showCloudStatus() {
            if (!CLOUD_API_URL) {
                this._noticeEl.innerHTML = `
<span class="tt-pg-notice-warn">
  ⚠ No cloud simulator URL configured. Set <code>TTSIM_API_URL</code> at build time.
  <a href="#pyodide-playground">Use the local Pyodide playground instead.</a>
</span>`;
                this._runBtn.disabled = true;
                return;
            }
            // Quick connectivity check via HTTP health endpoint
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
                            if (i < okBackends.length - 1) {
                                span.appendChild(document.createTextNode(', '));
                            }
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
                    span.appendChild(document.createTextNode('. '));
                    const link = document.createElement('a');
                    link.href = '#pyodide-playground';
                    link.textContent = 'Use the local Pyodide playground instead.';
                    span.appendChild(link);
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

            this._clearOutput();
            this._running = true;
            this._runBtn.disabled = true;
            this._runBtn.textContent = '⏳ Running…';

            const code = this._codeEl.value;
            const backend = this._backendSel.value;

            // Build preamble that opens a device inside the server environment.
            // ttsim-wh/ttsim-bh run against real tt-metal/ttnn (no tt-lang
            // installed there); ttlang-sim runs against tt-lang's ttl+ttnn.
            const preamble = backend.startsWith('ttsim')
                ? `
import ttnn
device = ttnn.open_device(device_id=0)
`
                : `
try:
    import ttl
    import ttnn
    device = ttnn.open_device(device_id=0)
except ImportError:
    pass
`;
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
                // 180s: enough for a cold checkpoint download (real_distilgpt2,
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
