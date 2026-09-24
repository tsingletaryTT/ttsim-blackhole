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

    // `slug` is the deep-link fragment for this category, as #sec-<slug>
    // (e.g. #sec-real). Kept separate from `key` so the URL can stay short
    // and readable even where the internal key isn't (key 'model' -> #sec-real).
    const CATEGORIES = [
        { key: 'start', slug: 'start', label: 'Start Here', desc: 'The smallest useful programs. No AI, no chip lore required.' },
        { key: 'ops', slug: 'ops', label: 'Op Zoo', desc: 'More of the primitives a transformer leans on, isolated and checked one at a time.' },
        { key: 'model', slug: 'real', label: 'Run a Real Model', desc: 'An actual HuggingFace checkpoint, real weights, a real prediction.' },
        { key: 'scale', slug: 'scale', label: 'Bigger Compute', desc: 'Same primitives, turned up: a deeper model, real tensor parallelism across chips.' },
        { key: 'precision', slug: 'precision', label: 'Precision & Numbers', desc: 'bfloat16 saves memory. It is not free. See what you gain and lose.' },
        { key: 'push', slug: 'push', label: 'Push the Simulator', desc: 'ttsim is deliberately stricter than silicon. See what that means.' },
        { key: 'mesh', slug: 'mesh', label: 'Multi-Chip', desc: 'Two virtual chips, one program — no second card required.' },
        { key: 'dsp', slug: 'dsp', label: 'Signal Processing', desc: 'An AI accelerator, repurposed as an audio and image filter.' },
    ];

    // ─── Kernel snippets ───────────────────────────────────────────────────────
    // Each entry: { category, label, blurb, tag?, complexity (1-3), backend,
    //               skipDevicePreamble?, models? (for a model picker),
    //               code: string | (model) => string }

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

    const TNT_DEFAULT_PROMPT = 'Once upon a time, there was a little';
    const TNT_MAX_PROMPT_LEN = 80;

    // Embeds a user-supplied prompt into a Python double-quoted string
    // literal safely: escape backslashes/quotes/newlines so the value can
    // only ever be DATA inside PROMPT = "...", never code that escapes the
    // literal, however it's crafted (this playground already lets anyone
    // type and run arbitrary Python in the editor -- the point here isn't
    // sandboxing, it's making sure a shared #demo-tt-tnt&prompt=... link
    // can't be crafted to make someone else's browser submit code other
    // than the prompt the link claims to carry).
    function escapePythonStringLiteral(s) {
        return s
            .slice(0, TNT_MAX_PROMPT_LEN)
            .replace(/\\/g, '\\\\')
            .replace(/"/g, '\\"')
            .replace(/\r?\n/g, '\\n');
    }

    const REAL_TNT_CODE = (_model, prompt) => _dedent(`
        # tt-tnt: a 123M-parameter Llama-3-style model trained from random
        # initialization ON Blackhole (via ttml/tt-train) -- not merely
        # trained elsewhere and ported here. Real weights, real architecture:
        # RMSNorm, rotary position embeddings, grouped-query attention
        # (16 query heads sharing 4 KV heads), SwiGLU MLP. Every one of those
        # pieces is written out below and run through ttnn on this simulated
        # chip, the same way the GPT-2 kernel does for its architecture.
        #
        # Generation is greedy (no sampling) and capped at a few tokens --
        # ttsim functionally simulates every instruction the chip actually
        # executes, so it's accurate but slow; each of these forward passes
        # through all 8 layers costs real wall-clock seconds, and that cost
        # compounds with every additional token. A handful of tokens is a
        # deliberate choice, not a limitation of the architecture.
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer

        MODEL_ID = "episod/tt-tnt-1024"
        PROMPT = "${escapePythonStringLiteral(prompt || TNT_DEFAULT_PROMPT)}"
        MAX_NEW_TOKENS = 6


        def tt_matmul(x, w_T):
            xt = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            wt = ttnn.from_torch(w_T.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            return ttnn.to_torch(ttnn.from_device(ttnn.matmul(xt, wt))).float()


        def tt_rms_norm(x, w, eps):
            xt = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            wt = ttnn.from_torch(w.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            y = ttnn.rms_norm(xt, weight=wt, epsilon=eps)
            return ttnn.to_torch(ttnn.from_device(y)).float()


        def tt_silu(x):
            xt = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            return ttnn.to_torch(ttnn.from_device(ttnn.silu(xt))).float()


        def tt_multiply(a, b):
            at = ttnn.from_torch(a.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            bt = ttnn.from_torch(b.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
            return ttnn.to_torch(ttnn.from_device(ttnn.multiply(at, bt))).float()


        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)


        def apply_rope(q, k, cos, sin):
            return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


        def repeat_kv(x, n_rep):
            # Grouped-query attention: each of the 4 KV heads is shared by
            # 4 query heads, so K/V get duplicated to match Q's head count
            # before the attention matmul.
            kvh, seq, hd = x.shape
            x = x[:, None, :, :].expand(kvh, n_rep, seq, hd)
            return x.reshape(kvh * n_rep, seq, hd)


        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        model.eval()

        cfg = model.config
        n_layer, n_head, n_kv_head = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        head_dim = cfg.hidden_size // n_head
        eps = cfg.rms_norm_eps
        # transformers >=5 moved rope_theta off the config into a nested
        # rope_parameters dict; this reads either shape.
        theta = getattr(cfg, "rope_theta", None)
        if theta is None:
            theta = cfg.rope_parameters["rope_theta"]
        sd = model.state_dict()

        emb = sd["model.embed_tokens.weight"]


        def forward_last_logits(ids_1d):
            # No KV cache: each call recomputes the full sequence, same as
            # the single-shot GPT-2 kernel -- simplicity over speed. RoPE
            # tables and the causal mask are rebuilt each call since the
            # sequence length grows by one every generation step.
            seq_len = len(ids_1d)
            h = emb[torch.tensor(ids_1d)]

            positions = torch.arange(seq_len)
            inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            freqs = torch.outer(positions.float(), inv_freq)
            emb_ang = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb_ang.cos(), emb_ang.sin()

            mask = torch.triu(torch.full((seq_len, seq_len), float("-1e4")), diagonal=1)
            mask = mask.unsqueeze(0).expand(n_head, seq_len, seq_len).contiguous()

            for i in range(n_layer):
                p = f"model.layers.{i}."
                ln1 = tt_rms_norm(h, sd[p + "input_layernorm.weight"], eps)
                q = tt_matmul(ln1, sd[p + "self_attn.q_proj.weight"].T.contiguous())
                k = tt_matmul(ln1, sd[p + "self_attn.k_proj.weight"].T.contiguous())
                v = tt_matmul(ln1, sd[p + "self_attn.v_proj.weight"].T.contiguous())

                q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
                k = k.view(seq_len, n_kv_head, head_dim).transpose(0, 1)
                v = v.view(seq_len, n_kv_head, head_dim).transpose(0, 1)

                q, k = apply_rope(q, k, cos, sin)
                k = repeat_kv(k, n_head // n_kv_head)
                v = repeat_kv(v, n_head // n_kv_head)

                scale = head_dim ** -0.5
                qt = ttnn.from_torch(q.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
                kt = ttnn.from_torch(k.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
                vt = ttnn.from_torch(v.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
                maskt = ttnn.from_torch(mask.bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
                scores = ttnn.matmul(qt, ttnn.permute(kt, (0, 2, 1))) * scale
                scores = scores + maskt
                attn = ttnn.softmax(scores, dim=-1)
                out_h = ttnn.to_torch(ttnn.from_device(ttnn.matmul(attn, vt))).float()

                out = out_h.transpose(0, 1).contiguous().view(seq_len, -1)
                out = tt_matmul(out, sd[p + "self_attn.o_proj.weight"].T.contiguous())
                h = h + out

                ln2 = tt_rms_norm(h, sd[p + "post_attention_layernorm.weight"], eps)
                gate = tt_matmul(ln2, sd[p + "mlp.gate_proj.weight"].T.contiguous())
                up = tt_matmul(ln2, sd[p + "mlp.up_proj.weight"].T.contiguous())
                mlp_out = tt_matmul(tt_multiply(tt_silu(gate), up), sd[p + "mlp.down_proj.weight"].T.contiguous())
                h = h + mlp_out

            h_final = tt_rms_norm(h, sd["model.norm.weight"], eps)
            # tie_word_embeddings: the output projection IS the embedding table.
            return tt_matmul(h_final[-1:], emb.T.contiguous())[0]


        input_ids = tok(PROMPT, return_tensors="pt").input_ids[0]
        print(f"Model: {MODEL_ID}  ({n_layer} layers, {n_head} heads / {n_kv_head} KV heads, trained on real Blackhole hardware)")
        print(f"Prompt: {PROMPT!r} ({len(input_ids)} tokens)")

        generated = input_ids.tolist()
        new_ids = []
        for step in range(MAX_NEW_TOKENS):
            logits = forward_last_logits(generated)
            next_id = int(logits.argmax().item())
            generated.append(next_id)
            new_ids.append(next_id)
            print(f"  + {tok.decode([next_id])!r}", flush=True)
            if next_id == tok.eos_token_id:
                break

        print(f"ttsim (Blackhole, {n_layer} real layers), greedy, {len(new_ids)} tokens:")
        print(f"Continuation: {tok.decode(generated, skip_special_tokens=True)!r}")
        print("PASSED")
    `);

    // ─── Output-line classifiers ───────────────────────────────────────────────
    // These recognize logging formats that show up in real ttnn runs but don't
    // carry spdlog's own "| LEVEL |" shape: TensorFlow/absl's own bracketed
    // logger (pulled in transitively by transformers), Python's warnings.warn()
    // traceback line, and a bare "Config{...}" line -- ttnn logs its startup
    // config as logger.debug(f"...\n{CONFIG}"), a single spdlog call whose
    // message contains an embedded newline, so the payload lands as its own
    // line with no level tag of its own.
    const ABSL_RE = /^[EWIF]\d{4}\s/;
    const TF_RE = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\.\d+:\s*[EWIF]\b/;
    const BARE_LEVEL_RE = /^\s*(WARNING|ERROR)\s*:/;
    const PYWARN_RE = /^\S+\.py:\d+:\s*\w*Warning:/;
    const CONFIG_DUMP_RE = /^Config\{/;

    // ─── Deep links ─────────────────────────────────────────────────────────
    // #sec-<category slug>  -> jumps to that category's tab
    // #demo-<kernel slug>   -> jumps to that kernel's card and selects it
    // Kernel slugs are just the key with underscores swapped for hyphens
    // (no kernel key contains a literal hyphen, so this reverses cleanly);
    // category slugs are explicit (see CATEGORIES) so the URL can stay short
    // even where the internal key isn't.
    function kernelSlug(key) {
        return key.replace(/_/g, '-');
    }
    function kernelKeyFromSlug(slug) {
        return slug.replace(/-/g, '_');
    }

    const KERNELS = {
        'hello_tensor': {
            category: 'start',
            label: 'Hello Tensor',
            blurb: 'Two 2×2 matrices, added together, on a chip that does not physically exist on this machine.',
            complexity: 1,
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
            complexity: 1,
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
            complexity: 1,
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
            complexity: 2,
            backend: 'ttsim-bh',
            models: [
                { value: 'distilgpt2', label: 'distilgpt2 (6 layers, faster)' },
                { value: 'gpt2', label: 'gpt2 (12 layers, slower)' },
                { value: 'gpt2-medium', label: 'gpt2-medium (24 layers, slowest)' },
            ],
            code: REAL_MODEL_CODE,
        },
        'tt_tnt': {
            category: 'model',
            label: 'tt-tnt: Tenstorrent’s Own Model',
            blurb: 'A 123M-param Llama-3-style model trained from scratch ON Blackhole, not just deployed to it. RMSNorm, RoPE, and grouped-query attention, all hand-written and run through ttnn here.',
            tag: '~250MB download, first run',
            complexity: 3,
            backend: 'ttsim-bh',
            promptable: true,
            // ttsim is a functional (instruction-accurate) simulator, not a
            // performance one -- 6 real forward passes through all 8 layers
            // genuinely costs minutes of wall-clock time. Overrides the
            // default 180s the client normally requests; the server still
            // caps every request at 300s regardless (see api_server.py).
            timeout: 280,
            code: REAL_TNT_CODE,
        },
        'softmax_only': {
            category: 'ops',
            label: 'Softmax',
            blurb: 'Raw scores in, a probability distribution out — the last step of every attention head, isolated.',
            complexity: 1,
            backend: 'ttsim-bh',
            code: _dedent(`
                import numpy as np
                import torch

                # Softmax turns a row of raw scores into a probability
                # distribution -- the operation that decides "how much
                # attention" each token gets, run here on its own.
                dim = 64
                x_np = np.random.randn(dim, dim).astype(np.float32)
                ref = np.exp(x_np - x_np.max(axis=-1, keepdims=True))
                ref = ref / ref.sum(axis=-1, keepdims=True)

                x = ttnn.from_torch(torch.from_numpy(x_np), layout=ttnn.TILE_LAYOUT, device=device)
                y = ttnn.softmax(x, dim=-1)
                result = ttnn.to_torch(ttnn.from_device(y)).numpy()

                max_err = float(np.abs(result - ref).max())
                print(f"softmax  dim={dim}x{dim}  max_err={max_err:.6f}")
                print("PASSED" if max_err < 1e-2 else "FAILED")
            `),
        },
        'reduction_ops': {
            category: 'ops',
            label: 'Row Reductions',
            blurb: 'Sum, mean, and max along a row — the quiet arithmetic underneath every normalization and pooling layer.',
            complexity: 1,
            backend: 'ttsim-bh',
            code: _dedent(`
                import numpy as np
                import torch

                # Row-wise reductions -- sum, mean, max along the last axis.
                # Every attention and normalization op leans on one of these.
                dim = 64
                x_np = np.random.rand(dim, dim).astype(np.float32)
                x = ttnn.from_torch(torch.from_numpy(x_np), layout=ttnn.TILE_LAYOUT, device=device)

                def reduce_row(op, ref_fn):
                    y = ttnn.to_torch(ttnn.from_device(op(x, dim=-1))).numpy().reshape(-1)[:dim]
                    ref = ref_fn(x_np, axis=-1)
                    return float(np.abs(y - ref).max())

                errs = {
                    "sum": reduce_row(ttnn.sum, np.sum),
                    "mean": reduce_row(ttnn.mean, np.mean),
                    "max": reduce_row(ttnn.max, np.max),
                }
                for name, err in errs.items():
                    print(f"{name}: max_err={err:.4f}")
                print("PASSED" if all(e < 0.5 for e in errs.values()) else "FAILED")
            `),
        },
        'embedding_lookup': {
            category: 'ops',
            label: 'Embedding Lookup',
            blurb: 'Token IDs in, dense vectors out — the very first operation in every transformer forward pass.',
            complexity: 2,
            backend: 'ttsim-bh',
            code: _dedent(`
                import numpy as np
                import torch

                # Embedding lookup: token IDs -> dense vectors, the very
                # first operation in every transformer forward pass.
                # ttnn.embedding requires uint32 ids and a bfloat16 table --
                # int32/float32 fail with a TT_FATAL dtype assertion.
                vocab, d_model, n_tokens = 256, 64, 16
                torch.manual_seed(0)
                table_np = np.random.randn(vocab, d_model).astype(np.float32)
                ids_np = np.random.randint(0, vocab, size=(1, n_tokens)).astype(np.int32)

                table = ttnn.from_torch(torch.from_numpy(table_np).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
                ids = ttnn.from_torch(torch.from_numpy(ids_np), dtype=ttnn.uint32, device=device)
                y = ttnn.embedding(ids, table)
                result = ttnn.to_torch(ttnn.from_device(y)).float().numpy().reshape(n_tokens, d_model)

                ref = table_np[ids_np[0]]
                max_err = float(np.abs(result - ref).max())
                print(f"embedding lookup: {n_tokens} tokens -> {d_model}-dim vectors")
                print(f"max_err={max_err:.6f}")
                print("PASSED" if max_err < 1e-2 else "FAILED")
            `),
        },
        'wide_matmul_regression': {
            category: 'precision',
            label: 'The Bug That Used To Break This',
            blurb: 'The exact 768→3072 matmul shape that aborted ttsim for months. Fixed as of the pairing this Space now runs.',
            tag: 'v1.10.7+ regression fix',
            complexity: 2,
            backend: 'ttsim-bh',
            code: _dedent(`
                import numpy as np
                import torch

                # GPT-2-style MLP up-projection: 768 -> 3072 output free dim.
                # This exact shape aborted ttsim v1.10.3-v1.10.6 with
                # UnsupportedFunctionality: tensix_pacr: Disable_pack_zero_flags
                # -- see this Space's Dockerfile for the full history. Fixed as
                # of the tt-metal/ttnn pairing this Space now runs.
                seq_len, d_model, d_ff = 32, 768, 3072
                x_np = np.random.rand(seq_len, d_model).astype(np.float32)
                w_np = np.random.rand(d_model, d_ff).astype(np.float32)
                ref = x_np @ w_np

                x = ttnn.from_torch(torch.from_numpy(x_np).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
                w = ttnn.from_torch(torch.from_numpy(w_np).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
                y = ttnn.matmul(x, w)
                result = ttnn.to_torch(ttnn.from_device(y)).float().numpy()

                max_err = float(np.abs(result - ref).max())
                print(f"wide matmul {seq_len}x{d_model} @ {d_model}x{d_ff} -> {seq_len}x{d_ff}")
                print(f"max_err={max_err:.3f}")
                print("PASSED" if max_err < ref.max() * 0.05 else "FAILED")
            `),
        },
        'fp32_vs_bf16': {
            category: 'precision',
            label: 'Full Precision vs bfloat16',
            blurb: 'The same matmul, run twice — once at fp32, once at the bfloat16 the chip actually stores tensors in.',
            complexity: 2,
            backend: 'ttsim-bh',
            code: _dedent(`
                import numpy as np
                import torch

                # bfloat16 keeps fp32's exponent range but only 7 mantissa
                # bits -- about 2-3 decimal digits. Every ttnn program on
                # real silicon makes this tradeoff; here it's made twice,
                # side by side.
                dim = 128
                a_np = np.random.rand(dim, dim).astype(np.float32)
                b_np = np.random.rand(dim, dim).astype(np.float32)
                ref = a_np @ b_np

                def run(cast):
                    a = ttnn.from_torch(cast(torch.from_numpy(a_np)), layout=ttnn.TILE_LAYOUT, device=device)
                    b = ttnn.from_torch(cast(torch.from_numpy(b_np)), layout=ttnn.TILE_LAYOUT, device=device)
                    return ttnn.to_torch(ttnn.from_device(ttnn.matmul(a, b))).float().numpy()

                fp32_result = run(lambda t: t.float())
                bf16_result = run(lambda t: t.bfloat16())

                fp32_err = float(np.abs(fp32_result - ref).max())
                bf16_err = float(np.abs(bf16_result - ref).max())
                print(f"matmul {dim}x{dim}, vs a float32 numpy reference")
                print(f"fp32  max_err={fp32_err:.6f}")
                print(f"bf16  max_err={bf16_err:.6f}  ({bf16_err / max(fp32_err, 1e-9):.0f}x larger)")
                print("PASSED")
            `),
        },
        'tensor_parallel_matmul': {
            category: 'scale',
            label: 'Tensor-Parallel Matmul',
            blurb: 'A weight matrix split by column across two virtual chips — real tensor parallelism, the trick that fits bigger models on a cluster.',
            tag: '2× Blackhole, tensor-parallel',
            complexity: 3,
            backend: 'ttsim-bh-x2',
            skipDevicePreamble: true,
            code: _dedent(`
                # This one manages its own (mesh) device -- ttsim-bh-x2
                # simulates a 2-chip Blackhole board (P300) over simulated
                # Ethernet.
                import torch

                # Tensor parallelism: split a big weight matrix by COLUMN
                # across two virtual chips, so each chip only computes its
                # own shard of the output -- the same trick real multi-chip
                # TT clusters use to fit a bigger model's weights.
                mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
                print("Opened mesh:", mesh)

                d_model, d_ff = 256, 512  # d_ff splits 256/256 across the two chips
                torch.manual_seed(0)
                x = torch.randn(64, d_model, dtype=torch.bfloat16)
                w = torch.randn(d_model, d_ff, dtype=torch.bfloat16)

                # Activation: replicated (every chip needs the full input row)
                x_mesh = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=mesh,
                                          mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                # Weight: column-sharded (chip 0 gets w[:, :256], chip 1 gets w[:, 256:])
                w_mesh = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=mesh,
                                          mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))

                # Each chip computes its own output shard, in parallel
                y_mesh = ttnn.matmul(x_mesh, w_mesh)
                y = ttnn.to_torch(y_mesh, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))

                ref = x.float() @ w.float()
                max_err = (y.float() - ref).abs().max().item()
                print(f"Tensor-parallel matmul: {tuple(x.shape)} @ {tuple(w.shape)} -> {tuple(y.shape)}, split across 2 chips")
                print(f"Max error vs reference: {max_err:.3f}")
                print("PASSED" if max_err < ref.abs().max().item() * 0.1 else "FAILED")
                ttnn.close_mesh_device(mesh)
            `),
        },
        'race_condition': {
            category: 'push',
            label: 'Race Condition',
            blurb: 'Comment out one line (marked below) and re-run. Silicon might let this slide; the simulator often will not.',
            complexity: 2,
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
            complexity: 2,
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
            complexity: 2,
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
        'dft_matmul': {
            category: 'dsp',
            label: 'Fourier Transform via Matmul',
            blurb: 'A DFT is just a matrix multiply against a fixed sine/cosine basis — no dedicated FFT hardware, only the same ttnn.matmul running the rest of this Space.',
            complexity: 2,
            backend: 'ttsim-bh',
            code: _dedent(`
                import numpy as np
                import torch

                # A Discrete Fourier Transform is a matrix multiply against a
                # fixed basis matrix of sines and cosines. Real ttnn matmul
                # is real-valued only, so the complex DFT matrix is split
                # into its real and imaginary halves and run as two matmuls.
                N = 64
                n = np.arange(N)
                k = n.reshape(-1, 1)
                dft_matrix = np.exp(-2j * np.pi * k * n / N)

                t = np.linspace(0, 1, N, endpoint=False)
                signal = (np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)).astype(np.float32)
                ref = np.fft.fft(signal)

                sig_t = ttnn.from_torch(torch.from_numpy(signal).reshape(N, 1), layout=ttnn.TILE_LAYOUT, device=device)
                real_mat = ttnn.from_torch(torch.from_numpy(dft_matrix.real.astype(np.float32)), layout=ttnn.TILE_LAYOUT, device=device)
                imag_mat = ttnn.from_torch(torch.from_numpy(dft_matrix.imag.astype(np.float32)), layout=ttnn.TILE_LAYOUT, device=device)

                real_out = ttnn.to_torch(ttnn.from_device(ttnn.matmul(real_mat, sig_t))).numpy().reshape(-1)[:N]
                imag_out = ttnn.to_torch(ttnn.from_device(ttnn.matmul(imag_mat, sig_t))).numpy().reshape(-1)[:N]

                magnitude = np.sqrt(real_out ** 2 + imag_out ** 2)
                ref_magnitude = np.abs(ref)

                max_err = float(np.abs(magnitude - ref_magnitude).max())
                peak_bin = int(magnitude[:N // 2].argmax())
                print(f"DFT via matmul: {N}-sample signal, peak frequency bin = {peak_bin} (expected 5)")
                print(f"max_err vs np.fft.fft magnitude: {max_err:.3f}")
                print("PASSED" if max_err < ref_magnitude.max() * 0.1 else "FAILED")
            `),
        },
        'sobel_edge': {
            category: 'dsp',
            label: 'Sobel Edge Detection',
            blurb: "A 2D convolution, expressed as im2col + one big matmul — the same trick tt-metal's own conv2d op compiles down to. Edges rendered as ASCII art.",
            complexity: 2,
            backend: 'ttsim-bh',
            code: _dedent(`
                import numpy as np
                import torch

                # Convolution as matmul: every "3x3 patch dot Sobel kernel"
                # at every pixel, batched into one ttnn.matmul call. This IS
                # how tt-metal's own conv2d op is compiled under the hood --
                # im2col (unroll patches into rows), then matmul.
                H, W = 16, 24
                img = np.zeros((H, W), dtype=np.float32)
                img[:, W // 2:] = 1.0  # a hard vertical edge down the middle

                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

                padded = np.pad(img, 1, mode='edge')
                patches = np.stack([
                    padded[i:i + H, j:j + W].reshape(-1)
                    for i in range(3) for j in range(3)
                ], axis=1)  # (H*W, 9)

                ref = (patches @ sobel_x.reshape(-1)).reshape(H, W)

                patches_t = ttnn.from_torch(torch.from_numpy(patches), layout=ttnn.TILE_LAYOUT, device=device)
                kernel_t = ttnn.from_torch(torch.from_numpy(sobel_x.reshape(9, 1)), layout=ttnn.TILE_LAYOUT, device=device)
                out = ttnn.matmul(patches_t, kernel_t)
                result = ttnn.to_torch(ttnn.from_device(out)).numpy().reshape(-1)[:H * W].reshape(H, W)

                max_err = float(np.abs(result - ref).max())

                def render(mat, chars=" .:-=+*#%@"):
                    m = np.abs(mat)
                    m = m / (m.max() + 1e-9)
                    return "\\n".join(
                        "".join(chars[min(int(v * (len(chars) - 1)), len(chars) - 1)] for v in row)
                        for row in m
                    )

                print("Input image:")
                print(render(img))
                print()
                print("Sobel edge response (one ttnn.matmul over image patches):")
                print(render(result))
                print()
                print(f"max_err vs numpy reference: {max_err:.4f}")
                print("PASSED" if max_err < 0.5 else "FAILED")
            `),
        },
        'game_of_life': {
            category: 'dsp',
            label: "Conway's Game of Life",
            blurb: 'Neighbor counts computed by the same im2col + matmul trick as Sobel edge detection -- six generations of a cellular automaton, rendered as ASCII.',
            complexity: 2,
            backend: 'ttsim-bh',
            code: _dedent(`
                import numpy as np
                import torch

                # Conway's Game of Life: still just im2col + matmul (the
                # same trick as the Sobel Edge Detection kernel), with a
                # different 3x3 kernel -- "count the 8 neighbors" -- and an
                # elementwise birth/death rule applied on the host.
                H, W = 16, 32
                GENERATIONS = 6

                rng = np.random.RandomState(0)
                grid = (rng.rand(H, W) < 0.35).astype(np.float32)  # random soup seed

                neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
                kernel_t = ttnn.from_torch(torch.from_numpy(neighbor_kernel.reshape(9, 1)), layout=ttnn.TILE_LAYOUT, device=device)

                def im2col(mat):
                    padded = np.pad(mat, 1, mode='constant')
                    return np.stack([
                        padded[i:i + H, j:j + W].reshape(-1)
                        for i in range(3) for j in range(3)
                    ], axis=1)

                def render(mat):
                    return "\\n".join("".join("#" if v > 0.5 else "." for v in row) for row in mat)

                print(f"Generation 0 ({int(grid.sum())} live cells):")
                print(render(grid))

                max_err = 0.0
                for gen in range(1, GENERATIONS + 1):
                    patches = im2col(grid)
                    patches_t = ttnn.from_torch(torch.from_numpy(patches), layout=ttnn.TILE_LAYOUT, device=device)
                    neighbors = ttnn.to_torch(ttnn.from_device(ttnn.matmul(patches_t, kernel_t))).numpy().reshape(-1)[:H * W].reshape(H, W)

                    ref_neighbors = (patches @ neighbor_kernel.reshape(-1)).reshape(H, W)
                    max_err = max(max_err, float(np.abs(neighbors - ref_neighbors).max()))

                    # B3/S23: a live cell with 2 or 3 neighbors survives; a
                    # dead cell with exactly 3 neighbors is born.
                    alive = grid > 0.5
                    grid = (((alive) & ((neighbors == 2) | (neighbors == 3))) | ((~alive) & (neighbors == 3))).astype(np.float32)
                    print(f"\\nGeneration {gen} ({int(grid.sum())} live cells):")
                    print(render(grid))

                print(f"\\nmax_err (ttnn matmul vs numpy neighbor count) across {GENERATIONS} generations: {max_err:.4f}")
                print("PASSED" if max_err < 0.5 else "FAILED")
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
            this._currentPrompt = null;
            this._activeCategory = KERNELS['hello_tensor'].category;
            this._activeOutputTab = 'logs';
            this._paneLineCounts = { output: 0, logs: 0 };
            this._pendingLineEl = { stdout: null, stderr: null };
            this._pendingLineText = { stdout: '', stderr: '' };

            this._buildUI();
            if (!this._applyHash(window.location.hash)) {
                this._setCategory(this._activeCategory);
                this._selectKernel('hello_tensor');
            }

            // Back/forward and pasted/edited links. Our own hash writes use
            // history.replaceState (see _setCategory/_selectKernel), which
            // does not fire hashchange, so this only ever reacts to an
            // external change -- no risk of feedback loops.
            window.addEventListener('hashchange', () => this._applyHash(window.location.hash));
        }

        // Returns true if the hash matched a known category or kernel.
        // Format: #demo-<slug>[&key=value...] or #sec-<slug>. The part
        // after '&' isn't real query-string syntax (it's still inside the
        // fragment), just a convention parsed the same way, so links stay
        // readable and copy/pasteable.
        _applyHash(hash) {
            const raw = (hash || '').replace(/^#/, '');
            if (!raw) return false;
            const [primary, ...rest] = raw.split('&');
            const params = {};
            for (const part of rest) {
                const eq = part.indexOf('=');
                if (eq === -1) continue;
                try {
                    params[decodeURIComponent(part.slice(0, eq))] = decodeURIComponent(part.slice(eq + 1));
                } catch {
                    // malformed percent-encoding in a hand-edited link -- ignore that param
                }
            }

            if (primary.startsWith('demo-')) {
                const key = kernelKeyFromSlug(primary.slice('demo-'.length));
                if (!KERNELS[key]) return false;
                this._setCategory(KERNELS[key].category);
                this._selectKernel(key, { prompt: params.prompt });
                return true;
            }
            if (primary.startsWith('sec-')) {
                const slug = primary.slice('sec-'.length);
                const cat = CATEGORIES.find(c => c.slug === slug);
                if (!cat) return false;
                this._setCategory(cat.key);
                return true;
            }
            return false;
        }

        _buildUI() {
            const tabsHtml = CATEGORIES.map(cat => `
<button class="tt-pg-tab" data-cat="${cat.key}" type="button">${cat.label}</button>`).join('');

            this._mount.innerHTML = `
<div class="tt-pg-cloud-notice" id="tt-pg-cloud-notice"></div>
<div class="tt-pg-tabbar" id="tt-pg-tabbar">${tabsHtml}</div>
<div class="tt-pg-workspace">
  <div class="tt-pg-browse-col">
    <div class="tt-pg-browse-header">
      <div class="tt-pg-category-desc" id="tt-pg-category-desc"></div>
      <button class="tt-pg-btn tt-pg-surprise-btn" id="tt-pg-surprise" type="button" title="Jump to a random kernel">&#127922; Surprise me</button>
    </div>
    <div class="tt-pg-card-grid" id="tt-pg-card-grid"></div>
  </div>
  <div class="tt-pg-do-col">
    <div class="tt-pg-toolbar">
      <span class="tt-pg-active-label" id="tt-pg-active-label"></span>
      <span class="tt-pg-model-picker" id="tt-pg-model-picker" hidden>
        <label class="tt-pg-label">Model</label>
        <select class="tt-pg-model-select" id="tt-pg-model-sel"></select>
      </span>
      <button class="tt-pg-btn tt-pg-run-btn" id="tt-pg-run">&#9654; Run on Simulator</button>
      <button class="tt-pg-btn tt-pg-clear-btn" id="tt-pg-clear">&#10006; Clear</button>
    </div>
    <div class="tt-pg-layout">
      <div class="tt-pg-editor-col">
        <textarea class="tt-pg-code" id="tt-pg-code" spellcheck="false"></textarea>
      </div>
      <div class="tt-pg-output-col">
        <div class="tt-pg-output-header">
          <button class="tt-pg-output-tab" data-pane="output" type="button">Output</button>
          <button class="tt-pg-output-tab active" data-pane="logs" type="button">Logs</button>
        </div>
        <pre class="tt-pg-output" id="tt-pg-output-output" hidden></pre>
        <pre class="tt-pg-output" id="tt-pg-output-logs"></pre>
      </div>
    </div>
  </div>
</div>`;

            this._tabbarEl = this._mount.querySelector('#tt-pg-tabbar');
            this._gridEl = this._mount.querySelector('#tt-pg-card-grid');
            this._categoryDescEl = this._mount.querySelector('#tt-pg-category-desc');

            this._tabbarEl.querySelectorAll('.tt-pg-tab').forEach(tab => {
                tab.addEventListener('click', () => this._setCategory(tab.dataset.cat));
            });

            this._gridEl.addEventListener('keydown', (e) => {
                if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
                e.preventDefault();
                this._moveCardFocus(e.key);
            });

            this._mount.querySelector('#tt-pg-surprise').addEventListener('click', () => this._surpriseMe());
            this._mount.querySelector('#tt-pg-run').addEventListener('click', () => this._run());
            this._mount.querySelector('#tt-pg-clear').addEventListener('click', () => this._clearOutput());

            const modelSel = this._mount.querySelector('#tt-pg-model-sel');
            modelSel.addEventListener('change', () => {
                this._currentModel = modelSel.value;
                this._loadCode();
            });

            this._noticeEl = this._mount.querySelector('#tt-pg-cloud-notice');
            this._codeEl = this._mount.querySelector('#tt-pg-code');
            this._outputPanes = {
                output: this._mount.querySelector('#tt-pg-output-output'),
                logs: this._mount.querySelector('#tt-pg-output-logs'),
            };
            this._outputTabEls = this._mount.querySelectorAll('.tt-pg-output-tab');
            this._outputTabEls.forEach(tab => {
                tab.addEventListener('click', () => this._setOutputTab(tab.dataset.pane));
            });
            this._runBtn = this._mount.querySelector('#tt-pg-run');
            this._activeLabelEl = this._mount.querySelector('#tt-pg-active-label');
            this._modelPickerEl = this._mount.querySelector('#tt-pg-model-picker');
            this._modelSelEl = modelSel;

            // Syntax highlighting via CodeMirror, loaded from CDN. Falls back
            // to the plain textarea (still fully functional) if it failed to
            // load -- e.g. offline, or a blocked third-party script.
            this._cm = (typeof window.CodeMirror !== 'undefined')
                ? window.CodeMirror.fromTextArea(this._codeEl, {
                    mode: 'python',
                    indentUnit: 4,
                    tabSize: 4,
                    indentWithTabs: false,
                    lineNumbers: true,
                    viewportMargin: Infinity,
                })
                : null;

            this._showCloudStatus();
        }

        // Renders the card grid for the given category key, preserving which
        // card (if any) is currently active.
        _renderGrid(categoryKey) {
            const cat = CATEGORIES.find(c => c.key === categoryKey) || CATEGORIES[0];
            this._categoryDescEl.textContent = cat.desc;

            const entries = Object.entries(KERNELS).filter(([, k]) => k.category === categoryKey);
            this._gridEl.innerHTML = entries.map(([key, k]) => {
                const dots = [1, 2, 3].map(n => `<span class="tt-pg-dot${n <= (k.complexity || 1) ? ' filled' : ''}"></span>`).join('');
                return `
<div class="tt-pg-card${key === this._currentKey ? ' active' : ''}" data-key="${key}" tabindex="0" role="button">
  <div class="tt-pg-card-title">${k.label}</div>
  <div class="tt-pg-card-blurb">${k.blurb}</div>
  <div class="tt-pg-card-footer">
    <span class="tt-pg-card-complexity" title="Complexity">${dots}</span>
    ${k.tag ? `<span class="tt-pg-card-tag">${k.tag}</span>` : ''}
  </div>
</div>`;
            }).join('');

            this._gridEl.querySelectorAll('.tt-pg-card').forEach(card => {
                card.addEventListener('click', () => this._selectKernel(card.dataset.key));
                card.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        this._selectKernel(card.dataset.key);
                    }
                });
            });
        }

        _setCategory(categoryKey) {
            this._activeCategory = categoryKey;
            this._tabbarEl.querySelectorAll('.tt-pg-tab').forEach(tab => {
                tab.classList.toggle('active', tab.dataset.cat === categoryKey);
            });
            this._renderGrid(categoryKey);

            const cat = CATEGORIES.find(c => c.key === categoryKey);
            if (cat) window.history.replaceState(null, '', `#sec-${cat.slug}`);
        }

        _moveCardFocus(key) {
            const cards = Array.from(this._gridEl.querySelectorAll('.tt-pg-card'));
            if (!cards.length) return;
            const focused = this._mount.ownerDocument.activeElement;
            let idx = cards.indexOf(focused);
            if (idx === -1) idx = 0;
            else if (key === 'ArrowLeft' || key === 'ArrowUp') idx = Math.max(0, idx - 1);
            else idx = Math.min(cards.length - 1, idx + 1);
            cards[idx].focus();
        }

        _surpriseMe() {
            const keys = Object.keys(KERNELS).filter(k => k !== this._currentKey);
            const pick = keys[Math.floor(Math.random() * keys.length)];
            if (!pick) return;
            const entry = KERNELS[pick];
            this._setCategory(entry.category);
            this._selectKernel(pick);
            const card = this._gridEl.querySelector(`.tt-pg-card[data-key="${pick}"]`);
            if (card) card.focus();
        }

        // `opts.prompt` restores a custom prompt from a deep link
        // (#demo-tt-tnt&prompt=...); only kernels marked `promptable` use
        // it, and any other selection path (a plain card click, Surprise
        // me) always resets to that kernel's own default prompt.
        _selectKernel(key, opts = {}) {
            const entry = KERNELS[key];
            if (!entry) return;
            this._currentKey = key;
            this._currentPrompt = (entry.promptable && opts.prompt) ? opts.prompt : null;

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

            const promptParam = (entry.promptable && this._currentPrompt)
                ? `&prompt=${encodeURIComponent(this._currentPrompt)}`
                : '';
            window.history.replaceState(null, '', `#demo-${kernelSlug(key)}${promptParam}`);
            this._loadCode();
        }

        _loadCode() {
            const entry = KERNELS[this._currentKey];
            if (!entry) return;
            const code = typeof entry.code === 'function' ? entry.code(this._currentModel, this._currentPrompt) : entry.code;
            if (this._cm) this._cm.setValue(code.trim());
            else this._codeEl.value = code.trim();
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

        _setOutputTab(pane) {
            this._activeOutputTab = pane;
            this._outputTabEls.forEach(tab => tab.classList.toggle('active', tab.dataset.pane === pane));
            Object.entries(this._outputPanes).forEach(([key, el]) => { el.hidden = key !== pane; });
        }

        // Updates a tab's label to "Output (5)" / "Logs (12)" once that pane
        // has lines in it (CSS uppercases the text, so this stays lowercase
        // here); reverts to the bare label at zero.
        _updateTabLabel(pane) {
            const tab = this._mount.querySelector(`.tt-pg-output-tab[data-pane="${pane}"]`);
            if (!tab) return;
            const base = pane === 'logs' ? 'Logs' : 'Output';
            const count = this._paneLineCounts[pane];
            tab.textContent = count > 0 ? `${base} (${count})` : base;
        }

        // A one-off system message (run errors, exit status) -- shown in
        // both panes so the pass/fail result is visible regardless of which
        // tab is active.
        _appendOutput(text, cls) {
            Object.values(this._outputPanes).forEach(pane => {
                const span = document.createElement('span');
                if (cls) span.className = cls;
                span.textContent = text;
                pane.appendChild(span);
                pane.scrollTop = pane.scrollHeight;
            });
        }

        // Classifies a line of stdout/stderr the way ttnn/spdlog would color
        // it in a real terminal -- our subprocess captures a plain pipe, not
        // a tty, so spdlog itself never emits ANSI codes to pass through;
        // this reproduces the same severity coloring from the level tag
        // ttnn's logger already prints in every line ("... | warning | ...").
        // A matched level tag (or one of the other known noise shapes above)
        // means this is framework log noise, not the kernel's own print()
        // output -- routed to the "logs" pane so "output" stays just the
        // program's own result.
        _classifyLine(line, streamKey) {
            const m = line.match(/\|\s*(TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|ERR|CRITICAL)\s*\|/i);
            if (m) {
                const lvl = m[1].toLowerCase();
                let cls;
                if (lvl.startsWith('warn')) cls = 'tt-pg-log-warn';
                else if (lvl === 'err' || lvl === 'error') cls = 'tt-pg-log-error';
                else if (lvl === 'critical') cls = 'tt-pg-log-critical';
                else if (lvl === 'debug') cls = 'tt-pg-log-debug';
                else if (lvl === 'trace') cls = 'tt-pg-log-trace';
                else cls = 'tt-pg-log-info';
                return { cls, pane: 'logs' };
            }
            if (CONFIG_DUMP_RE.test(line)) {
                return { cls: 'tt-pg-log-debug', pane: 'logs' };
            }
            const bare = line.match(BARE_LEVEL_RE);
            if (bare) {
                return { cls: bare[1] === 'ERROR' ? 'tt-pg-log-error' : 'tt-pg-log-warn', pane: 'logs' };
            }
            if (PYWARN_RE.test(line)) {
                return { cls: 'tt-pg-log-warn', pane: 'logs' };
            }
            const tf = line.match(TF_RE) || line.match(ABSL_RE);
            if (tf) {
                const letter = (tf[0].match(/[EWIF]/) || ['I'])[0];
                const cls = letter === 'E' ? 'tt-pg-log-error' : letter === 'W' ? 'tt-pg-log-warn' : 'tt-pg-log-info';
                return { cls, pane: 'logs' };
            }
            return { cls: streamKey === 'stderr' ? 'tt-pg-stderr' : 'tt-pg-stdout', pane: 'output' };
        }

        // Appends streamed text line-by-line, classifying each line as soon
        // as it's complete (a chunk boundary rarely lands mid-line, so this
        // stays effectively real-time) while still growing the in-progress
        // line's span incrementally for live streaming feel. Re-parenting an
        // in-progress span into its (possibly newly decided) pane on every
        // update also correctly handles the rare case where a chunk splits
        // before the log-level tag is visible yet.
        //
        // A line with none of _classifyLine's recognized shapes, but which
        // immediately follows a logs-pane line within this SAME chunk, is
        // treated as a continuation of that message rather than the kernel's
        // own output -- e.g. Python's warnings.warn() can print several
        // unmarked lines of message body after its ".py:NNN: XWarning:"
        // line, all in one process write() (and so, in practice, one chunk).
        // Scoping this to "same chunk" only (never persisted across
        // separate reads) keeps a real print() that happens to follow
        // framework noise in a *later* chunk from being misclassified.
        _appendStreamText(streamKey, text) {
            const parts = text.split('\n');
            let lastPaneThisChunk = null;
            for (let i = 0; i < parts.length; i++) {
                if (!this._pendingLineEl[streamKey]) {
                    this._pendingLineEl[streamKey] = document.createElement('span');
                    this._pendingLineText[streamKey] = '';
                }
                this._pendingLineText[streamKey] += parts[i];
                const el = this._pendingLineEl[streamKey];
                let { cls, pane } = this._classifyLine(this._pendingLineText[streamKey], streamKey);
                if (pane === 'output' && lastPaneThisChunk === 'logs') {
                    cls = 'tt-pg-log-debug';
                    pane = 'logs';
                }
                el.textContent = this._pendingLineText[streamKey];
                el.className = cls;
                const paneEl = this._outputPanes[pane];
                if (el.parentNode !== paneEl) paneEl.appendChild(el);

                if (i < parts.length - 1) {
                    paneEl.appendChild(document.createTextNode('\n'));
                    this._pendingLineEl[streamKey] = null;
                    this._pendingLineText[streamKey] = '';
                    this._paneLineCounts[pane]++;
                    this._updateTabLabel(pane);
                    lastPaneThisChunk = pane;
                }
            }
            Object.values(this._outputPanes).forEach(pane => { pane.scrollTop = pane.scrollHeight; });
        }

        _clearOutput() {
            Object.values(this._outputPanes).forEach(pane => { pane.textContent = ''; });
            this._pendingLineEl = { stdout: null, stderr: null };
            this._pendingLineText = { stdout: '', stderr: '' };
            this._paneLineCounts = { output: 0, logs: 0 };
            this._updateTabLabel('output');
            this._updateTabLabel('logs');
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

            const code = this._cm ? this._cm.getValue() : this._codeEl.value;

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
                // 180s default: enough for a cold checkpoint download (Real HF
                // Checkpoint, first run) plus device init + kernel JIT on a
                // modest CPU tier. A kernel can request more (see tt-tnt's
                // multi-step generation loop); the server still caps every
                // request at 300s regardless (see api_server.py).
                const timeout = entry.timeout || 180;
                this._ws.send(JSON.stringify({ code: fullCode, backend, timeout }));
            };

            this._ws.onmessage = (evt) => {
                let msg;
                try { msg = JSON.parse(evt.data); } catch { return; }
                if (msg.type === 'stdout') {
                    this._appendStreamText('stdout', msg.data);
                } else if (msg.type === 'stderr') {
                    this._appendStreamText('stderr', msg.data);
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
