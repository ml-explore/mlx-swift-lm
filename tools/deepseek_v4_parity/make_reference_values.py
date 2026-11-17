"""Writes Tests/MLXLMTests/DeepseekV4ReferenceValues.swift from DeepSeek's own DeepSeek-V4
reference implementation (`inference/model.py`), run unmodified on the CPU.

Usage, from the repository root, with a python that has torch and numpy (no MLX needed):

    python -B tools/deepseek_v4_parity/make_reference_values.py --reference-dir <inference folder>

`<inference folder>` is DeepSeek-V4's `inference/` directory (model.py, ...). It is imported, not
copied; the sha256 of its model.py is written to the output. `reference_shim/` is put first on
sys.path, so model.py's `from kernel import ...` gets reference_shim/kernel.py (a float32 CPU
version of kernel.py's sparse_attn and hc_split_sinkhorn, activation quantization identity) and
its `fast_hadamard_transform` import gets an identity (the rotation cancels in the indexer's dot
products without the quantization). Changes to the reference, all here: `rotate_activation`
without its bfloat16 assert; the model built in float32 (generate.py builds bfloat16); and in the
HISA runs only, each `Indexer.forward` replaced by a copy whose top-k is mlx-vlm's two-stage block
selection (`hisa_forward`; DeepSeek's reference has no HISA). Prefill is one call over the prompt
but its last token, then one-token decode steps.

The model: 4 layers (compress ratios 0, 4, 128, 4, one hash-routed layer, an 8-token window),
stored as the DeepSeek-V4-Flash MXFP4 MLX conversion stores it. Parameter `i` in sorted name
order, element `j` row-major, seed `s`:
    x = (s * 2**23 + i * 2**15 + j) % P,  h = (x * x % P) * 48271 % P,  P = 2**31 - 1
Float parameters: (shift + h % (2 * half + 1) - half) / denom, integers then one float32 division,
(shift, half, denom) = (128, 20, 128) for norm weights, (0, 20, 256) for the router bias,
(64, 20, 128) for hyper-connection scales, (0, 32, 128) otherwise; all exact in bfloat16, stored
as float32 where the reference keeps float32 and as bfloat16 otherwise. Routed experts (MXFP4)
and attention projections, indexer `wq_b` and shared expert (MXFP8) are packed codes: element j
is code h % 16 (MXFP4) or the e4m3 code of h % 33 - 16 (MXFP8); group g's scale exponent, from h
at j = (element count) + g, is -5 + h % 2 (MXFP4) or -7 + h % 2 (MXFP8). A hash-routing row picks
its experts from the sorted unused ones at index h % (number left).

The seed is kept only if every discrete selection (routed experts, indexer top-k, HISA blocks,
greedy argmax) has a margin at least MARGIN_FACTOR times the largest change of its scores when
every product (Linear, einsum, the shim's attention products, the hyper-connection update) is
replaced by its float64 value rounded to float32 and moved by random -2..2 ulps. That check runs
after the stored values are taken, with those products patched; SEED is the first seed from 11
that passes it, and the script exits with an error if it does not pass.
"""

import argparse
import functools
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

SEED = 149
DECODE, TOP_K, MARGIN_FACTOR, MARGIN_FLOOR = 24, 4, 20.0, 1e-5
CASES = [(250, 64), (250, 2048), (45, 16)]  # (prompt length, Swift prefill step)
INDEXER = {"hisa": (4, 2), "flat": (0, 0)}  # (index_block, index_keep)
TOOLS = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(TOOLS, "..", "..", "Tests", "MLXLMTests", "DeepseekV4ReferenceValues.swift")
CONFIG = {
    "model_type": "deepseek_v4", "vocab_size": 128, "hidden_size": 64,
    "moe_intermediate_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4,
    "num_key_value_heads": 1, "n_shared_experts": 1, "n_routed_experts": 8,
    "num_experts_per_tok": 2, "routed_scaling_factor": 1.5, "q_lora_rank": 32,
    "qk_rope_head_dim": 16, "head_dim": 64, "norm_topk_prob": True, "rms_norm_eps": 1e-6,
    "rope_theta": 10000, "compress_rope_theta": 160000,
    "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
                     "beta_fast": 32, "beta_slow": 1},
    "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
    "compress_ratios": [0, 4, 128, 4, 0], "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
    "num_hash_layers": 1, "swiglu_limit": 10.0, "sliding_window": 8, "o_groups": 2,
    "o_lora_rank": 32,
    # Sixteen heads: with two, a quarter of the indexer scores are relu-clamped to exactly zero.
    "index_n_heads": 16, "index_head_dim": 32, "index_topk": 4, "index_block": 4, "index_keep": 2,
    "tie_word_embeddings": False, "bos_token_id": 0, "eos_token_id": 1,
}
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
E4M3 = np.array([(-1) ** (c >> 7) * ((c & 7) / 8 * 2.0 ** -6 if (c >> 3) & 15 == 0
                                     else (1 + (c & 7) / 8) * 2.0 ** (((c >> 3) & 15) - 7))
                 for c in range(256)])
ref = kernel = None


def load_reference(reference_dir):
    global ref, kernel
    sys.path[:0] = [os.path.join(TOOLS, "reference_shim"), os.path.abspath(reference_dir)]
    torch.set_default_dtype(torch.float32)
    import fast_hadamard_transform
    import kernel
    import model as ref
    if not os.path.samefile(os.path.dirname(ref.__file__), reference_dir):
        raise SystemExit(f"model.py did not come from {reference_dir}")
    ref.rotate_activation = lambda x: fast_hadamard_transform.hadamard_transform(
        x, scale=x.size(-1) ** -0.5)


def model_args():
    c, rs = CONFIG, CONFIG["rope_scaling"]
    return ref.ModelArgs(
        max_batch_size=1, max_seq_len=512, dtype="bf16", expert_dtype=None,
        vocab_size=c["vocab_size"], dim=c["hidden_size"], moe_inter_dim=c["moe_intermediate_size"],
        n_layers=c["num_hidden_layers"], n_hash_layers=c["num_hash_layers"], n_mtp_layers=0,
        n_heads=c["num_attention_heads"], n_routed_experts=c["n_routed_experts"],
        n_shared_experts=c["n_shared_experts"], n_activated_experts=c["num_experts_per_tok"],
        score_func=c["scoring_func"], route_scale=c["routed_scaling_factor"],
        swiglu_limit=c["swiglu_limit"], q_lora_rank=c["q_lora_rank"], head_dim=c["head_dim"],
        rope_head_dim=c["qk_rope_head_dim"], norm_eps=c["rms_norm_eps"], o_groups=c["o_groups"],
        o_lora_rank=c["o_lora_rank"], window_size=c["sliding_window"],
        compress_ratios=tuple(c["compress_ratios"]), compress_rope_theta=c["compress_rope_theta"],
        original_seq_len=rs["original_max_position_embeddings"], rope_theta=c["rope_theta"],
        rope_factor=rs["factor"], beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"],
        index_n_heads=c["index_n_heads"], index_head_dim=c["index_head_dim"],
        index_topk=c["index_topk"], hc_mult=c["hc_mult"], hc_sinkhorn_iters=c["hc_sinkhorn_iters"],
        hc_eps=c["hc_eps"])


# ------------------------------------------------------------------ weights
def storage(name):
    if ".ffn.experts." in name:
        return "mxfp4"
    if ".ffn.shared_experts." in name or ".attn.w" in name or ".attn.indexer.wq" in name:
        return "mxfp8"
    if name.endswith("tid2eid"):
        return "int32"
    if ".hc_" in name or name.startswith("hc_") or name.endswith(("attn_sink", ".ape", ".gate.bias")):
        return "float32"
    return "bfloat16"


def weights(seed):
    """The checkpoint (name -> numpy array), the sorted (name, shape, storage) table and the
    reference state dict with the values the checkpoint decodes to."""
    shapes = sorted((k, tuple(v.shape)) for k, v in ref.Transformer(model_args()).named_parameters())
    checkpoint, table, state = {}, [], {}
    for i, (name, shape) in enumerate(shapes):
        n, kind = math.prod(shape), storage(name)
        x = (seed * 2**23 + i * 2**15 + np.arange(n + n // 32, dtype=np.int64)) % (2**31 - 1)
        h = (x * x % (2**31 - 1)) * 48271 % (2**31 - 1)
        if kind == "int32":
            value = np.zeros(shape, dtype=np.int32)
            for r in range(shape[0]):
                left = list(range(CONFIG["n_routed_experts"]))
                for c in range(shape[1]):
                    value[r, c] = left.pop(int(h[r * shape[1] + c]) % len(left))
            checkpoint[name] = value
        elif kind in ("mxfp4", "mxfp8"):
            bits = 4 if kind == "mxfp4" else 8
            if bits == 4:
                codes = h[:n] % 16
            else:
                m = h[:n] % 33 - 16
                a = np.abs(m)
                e = np.floor(np.log2(np.maximum(a, 1))).astype(np.int64)
                codes = np.where(a == 0, 0, ((e + 7) << 3) | ((a << 3 >> e) - 8)) | (m < 0) * 128
            scales = (127 - (5 if bits == 4 else 7) + h[n:] % 2).astype(np.uint8)
            per = 32 // bits
            words = codes.astype(np.uint32).reshape(-1, per) << (np.arange(per, dtype=np.uint32) * bits)
            checkpoint[name] = np.bitwise_or.reduce(words, axis=1).reshape(shape[0], -1)
            checkpoint[name[: -len(".weight")] + ".scales"] = scales.reshape(shape[0], -1)
            value = ((E2M1 if bits == 4 else E4M3)[codes]
                     * np.repeat(2.0 ** (scales.astype(np.float64) - 127), 32)).astype(np.float32)
        else:
            shift, half, denom = (
                (128, 20, 128) if name.endswith("norm.weight") else (0, 20, 256)
                if name.endswith(".gate.bias") else (64, 20, 128) if name.endswith("_scale")
                else (0, 32, 128))
            value = (shift + h[:n] % (2 * half + 1) - half).astype(np.float32) / np.float32(denom)
            checkpoint[name] = value.reshape(shape)
        state[name] = torch.from_numpy(np.ascontiguousarray(value).reshape(shape))
        table.append((name, shape, kind))
    return checkpoint, table, state


def checksum(checkpoint):
    """h = h * 1000003 + v (mod 2**64) over the tensors in sorted name order, v each element's
    float32 bit pattern or integer value."""
    h = 0
    for name in sorted(checkpoint):
        a = np.ascontiguousarray(checkpoint[name]).reshape(-1)
        for v in (a.view(np.uint32) if a.dtype in (np.float32, np.int32) else a).tolist():
            h = (h * 1000003 + v) & (2**64 - 1)
    return h


# ------------------------------------------------------------------ runs
REC = {"events": None, "layer": None, "step": -1, "ctx": None}


def record(kind, scores, k, selected):
    """A selection's scores (rows, N), its k picks and the margin k-th minus (k+1)-th largest of
    the unmasked scores (inf when nothing is rejected)."""
    if REC["events"] is None:
        return
    rows = np.asarray(scores, dtype=np.float64).reshape(-1, scores.shape[-1])
    margins = [math.inf if (v := np.sort(r[r > -1e29])[::-1]).size <= k else v[k - 1] - v[k]
               for r in rows]
    REC["events"].append({"kind": kind, "layer": REC["layer"], "step": REC["step"], "scores": rows,
                          "margins": np.array(margins),
                          "selected": np.sort(np.asarray(selected).reshape(len(rows), -1), -1)})


def hisa_forward(self, block, keep, x, qr, start_pos, offset):
    """model.py's Indexer.forward with mlx-vlm's HISA top-k: pooled entries in blocks of `block`,
    blocks ranked by the score of their mean entry, the best `keep` blocks scored in full."""
    bsz, seqlen, _ = x.size()
    ratio, end_pos = self.compress_ratio, start_pos + seqlen
    if self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache
        self.compressor.freqs_cis = self.freqs_cis
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
    ref.apply_rotary_emb(q[..., -self.rope_head_dim:], self.freqs_cis[start_pos:end_pos])
    q = ref.rotate_activation(q)
    ref.fp4_act_quant(q, ref.fp4_block_size, True)
    self.compressor(x, start_pos)
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
    pool = end_pos // ratio
    kv, k = self.kv_cache[:bsz, :pool], min(self.index_topk, pool)
    visible = (torch.arange(1, seqlen + 1) // ratio if start_pos == 0
               else torch.full((seqlen,), pool))[None, :, None]
    if pool >= block * keep and block * keep >= k:
        blocks = pool // block
        mean = kv[:, :blocks * block].unflatten(1, (blocks, block)).mean(2)
        coarse = (torch.einsum("bshd,btd->bsht", q, mean).relu() * weights.unsqueeze(-1)).sum(dim=2)
        coarse = torch.where((torch.arange(blocks) * block)[None, None] < visible, coarse, -1e30)
        top = coarse.topk(keep, dim=-1)[1]
        record("coarse", coarse.double().numpy(), keep, top.numpy())
        positions = (top[..., None] * block + torch.arange(block)).flatten(-2)
        candidates = torch.gather(kv[:, None].expand(bsz, seqlen, pool, kv.size(-1)), 2,
                                  positions[..., None].expand(*positions.shape, kv.size(-1)))
        kind = "fine"
    else:  # too few entries for HISA: every entry is a candidate
        positions = torch.arange(pool).expand(bsz, seqlen, pool)
        candidates, kind = kv[:, None].expand(bsz, seqlen, pool, kv.size(-1)), "flat"
    score = (torch.einsum("bshd,bscd->bshc", q, candidates).relu() * weights.unsqueeze(-1)).sum(dim=2)
    score = torch.where(positions < visible, score, -1e30)
    chosen = score.topk(k, dim=-1)[1]
    record(kind, score.double().numpy(), k, chosen.numpy())
    idxs = torch.gather(positions, -1, chosen)
    if start_pos == 0:
        return torch.where(idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio, -1, idxs + offset)
    return idxs + offset


def run(state, prompt, hisa, forced=None):
    """One-shot prefill of prompt[:-1], then DECODE steps fed the last prompt token and then the
    greedy (or `forced`) tokens: per step the log-probs, the routed experts of each score-routed
    layer and the pooled entries each ratio-4 indexer picks."""
    model = ref.Transformer(model_args()).float()
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(f"state dict mismatch: {result}")
    model.eval()
    steps = []

    def gate_hook(module, args, out):
        if REC["step"] >= 0 and not module.hash:
            steps[-1][0].append(sorted(out[1][0].tolist()))

    def indexer_hook(module, args, out):
        if REC["step"] >= 0:
            steps[-1][1].append(sorted((out[0, 0] - args[3]).tolist()))

    def layer_hook(module, args):
        REC["layer"] = module.layer_id

    for block in model.layers:
        block.register_forward_pre_hook(layer_hook)
        block.ffn.gate.register_forward_hook(gate_hook)
        if (indexer := getattr(block.attn, "indexer", None)) is not None:
            indexer.register_forward_hook(indexer_hook)
            if hisa:
                indexer.forward = functools.partial(hisa_forward, indexer, *hisa)
    REC.update(step=-1, layer=None)  # -1: prefill
    model(torch.tensor([prompt[:-1]]), 0)
    token, logprobs = prompt[-1], []
    for t in range(DECODE):
        REC["step"] = t
        steps.append(([], []))
        logits = model(torch.tensor([[token]]), len(prompt) - 1 + t)[0].float()
        logprobs.append((logits - torch.logsumexp(logits, -1, keepdim=True)).numpy().copy())
        token = int(np.argmax(logprobs[-1])) if forced is None else forced[t]
    return logprobs, steps


def check_margins(state, prompt, hisa, logprobs, steps):
    """Reruns the case observed, then with every product re-rounded (canonically and moved by
    random -2..2 ulps, three seeds), teacher-forced; raises SystemExit if a selection changes.
    Returns per kind the smallest margin and the largest score change, and the largest log-prob
    change."""
    greedy = [int(np.argmax(lp)) for lp in logprobs]
    linear, einsum, hc_post, topk = (torch.nn.functional.linear, torch.einsum,
                                     ref.Block.hc_post, torch.Tensor.topk)
    pert = {"fn": None}

    def topk_hook(self, k, dim=-1, largest=True, sorted=True):
        out = topk(self, k, dim=dim, largest=largest, sorted=sorted)
        if REC["ctx"] is not None:
            record(REC["ctx"], self.double().numpy(), k, out[1].numpy())
        return out

    def gate_forward(self, x, input_ids=None):
        REC["ctx"] = "moe"
        try:
            return ref_gate(self, x, input_ids)
        finally:
            REC["ctx"] = None

    def indexer_forward(self, *args):
        REC["ctx"] = "flat"
        try:
            return ref_indexer(self, *args)
        finally:
            REC["ctx"] = None

    def product(fn, *args):
        tensors = [a for a in args if torch.is_tensor(a)]
        if pert["fn"] is None or not all(t.is_floating_point() for t in tensors):
            return fn(*args)
        exact = fn(*[a.double() if torch.is_tensor(a) else a for a in args])
        return pert["fn"](exact).to(tensors[0].dtype)

    def hc_post_hook(self, x, residual, post, comb):
        if pert["fn"] is None:
            return hc_post(self, x, residual, post, comb)
        y = post.double().unsqueeze(-1) * x.double().unsqueeze(-2) + torch.sum(
            comb.double().unsqueeze(-1) * residual.double().unsqueeze(-2), dim=2)
        return pert["fn"](y).type_as(x)

    def perturbation(ulps, seed):
        rng = np.random.default_rng(seed)

        def fn(exact):
            r = exact.to(torch.float32)
            if ulps:
                steps = torch.from_numpy(rng.integers(-ulps, ulps + 1, size=tuple(r.shape)).astype(np.int32))
                moved = (r.view(torch.int32) + torch.where(r > 0, steps, -steps)).view(torch.float32)
                r = torch.where((r != 0) & torch.isfinite(r), moved, r)
            return r
        return fn

    def observed(perturb):
        REC["events"] = []
        pert["fn"] = kernel.PERTURB = perturb
        try:
            lp, picked = run(state, prompt, hisa, forced=greedy)
        finally:
            pert["fn"] = kernel.PERTURB = None
        events, REC["events"] = REC["events"], None
        for t, row in enumerate(lp):
            a, b = np.sort(row)[-2:]
            events.append({"kind": "argmax", "layer": None, "step": t, "scores": row[None].astype(np.float64),
                           "margins": np.array([b - a]), "selected": np.array([[int(np.argmax(row))]])})
        return lp, picked, events

    ref_gate, ref_indexer = ref.Gate.forward, ref.Indexer.forward
    torch.Tensor.topk, torch.nn.functional.linear = topk_hook, functools.partial(product, linear)
    torch.einsum, ref.Block.hc_post = functools.partial(product, einsum), hc_post_hook
    ref.Gate.forward = gate_forward
    if hisa is None:
        ref.Indexer.forward = indexer_forward
    try:
        base_lp, base_picked, base = observed(None)
        if any(not np.array_equal(a, b) for a, b in zip(base_lp, logprobs)) or base_picked != steps:
            raise SystemExit("the observed run differs from the stored run")
        sensitivity, worst = {}, 0.0
        for ulps, seed in [(0, 0), (2, 1), (2, 2), (2, 3)]:
            lp, _, events = observed(perturbation(ulps, seed))
            for a, b in zip(base, events, strict=True):
                if not np.array_equal(a["selected"], b["selected"]):
                    raise SystemExit(f"{a['kind']} selection changes under re-rounding")
                both = (a["scores"] > -1e29) & (b["scores"] > -1e29)
                change = float(np.abs(a["scores"][both] - b["scores"][both]).max(initial=0))
                sensitivity[a["kind"]] = max(sensitivity.get(a["kind"], 0.0), change)
            worst = max(worst, max(float(np.abs(x - y).max()) for x, y in zip(lp, base_lp)))
    finally:
        torch.Tensor.topk, torch.nn.functional.linear, torch.einsum = topk, linear, einsum
        ref.Block.hc_post, ref.Gate.forward, ref.Indexer.forward = hc_post, ref_gate, ref_indexer
    margins = {k: min(float(e["margins"].min()) for e in base if e["kind"] == k) for k in sensitivity}
    return margins, sensitivity, worst


# ------------------------------------------------------------------ Swift literals
def json_lines(value, indent=0, key=None):
    head = " " * indent + (f"{json.dumps(key)}: " if key is not None else "")
    if not isinstance(value, dict) or len(head + json.dumps(value)) <= 90:
        return [head + json.dumps(value)]
    items = sorted(value.items())
    lines = [head + "{"]
    for n, (k, v) in enumerate(items):
        inner = json_lines(v, indent + 2, k)
        lines += inner[:-1] + [inner[-1] + ("," if n < len(items) - 1 else "")]
    return lines + [" " * indent + "}"]


def literal(values, per_line, indent, fmt=str):
    rows = [" " * (indent + 4) + ", ".join(fmt(v) for v in values[k:k + per_line]) + ","
            for k in range(0, len(values), per_line)]
    return ["["] + rows + [" " * indent + "]"]


def write_swift(config, table, digest, runs, sha):
    flt = lambda v: f"{float(np.float32(v)):.9g}"  # noqa: E731
    lines = [
        "// Generated by tools/deepseek_v4_parity/make_reference_values.py; do not edit values by hand.",
        "// DeepSeek-V4 inference/model.py, unmodified, on the CPU kernel shim, sha256",
        f"// {sha};",
        f"// PyTorch {torch.__version__} CPU float32, numpy {np.__version__}. Weight seed {SEED}.",
        "",
        "enum DeepseekV4ReferenceValues {",
        '    static let configuration = """',
        *["        " + line for line in json_lines(config)],
        '        """',
        "",
        "    enum Storage { case bfloat16, float32, int32, mxfp4, mxfp8 }",
        "",
        "    /// The reference's parameters in sorted name order with their unquantized shapes; the",
        "    /// weights come from the formula in the generator's docstring with this seed.",
        f"    static let weightSeed = {SEED}",
        "    static let parameters: [(name: String, shape: [Int], storage: Storage)] = [",
        *[f'        ("{name}", {list(shape)}, .{kind}),' for name, shape, kind in table],
        "    ]",
        "",
        "    /// h = h * 1000003 + v (mod 2^64) over the stored checkpoint, tensors in sorted name",
        "    /// order, v each element's float32 bit pattern or integer value.",
        f"    static let checkpointChecksum: UInt64 = {digest:_}",
        "",
        f"    static let decode = {DECODE}",
        "",
        "    /// A teacher-forced reference run: one-shot prefill of the prompt but its last token, then",
        "    /// `decode` steps fed the last prompt token and then `greedy`. Per decode step: the routed",
        "    /// experts of each score-routed layer and the pooled entries the indexer of each ratio-4",
        f"    /// layer picks (sorted, flattened), the log-probabilities of the reference's {TOP_K} most",
        "    /// likely tokens, and the whole log-softmax row of the last step.",
        "    struct Run {",
        "        let indexer: String",
        "        let indexBlock: Int",
        "        let indexKeep: Int",
        "        /// Prefill chunk sizes the Swift test runs against this one-shot reference run.",
        "        let prefillSteps: [Int]",
        "        let prompt: [Int]",
        "        let greedy: [Int]",
        "        let experts: [Int]",
        "        let indexerSelection: [Int]",
        "        let topTokens: [Int]",
        "        let topLogprobs: [Float]",
        "        let lastLogprobs: [Float]",
        "    }",
        "",
    ]
    prompts = {len(r["prompt"]): r["prompt"] for r in runs}
    for n, prompt in sorted(prompts.items()):
        body = literal(prompt, 16, 4)
        lines += [f"    static let prompt{n}: [Int] = {body[0]}", *body[1:], ""]
    lines.append("    static let runs: [Run] = [")
    for r in runs:
        fields = [("indexer", f'"{r["indexer"]}"'), ("indexBlock", str(r["block"])),
                  ("indexKeep", str(r["keep"])), ("prefillSteps", str(r["steps"])),
                  ("prompt", f"prompt{len(r['prompt'])}"), ("greedy", literal(r["greedy"], 16, 12)),
                  ("experts", literal(r["experts"], 16, 12)),
                  ("indexerSelection", literal(r["indexer_selection"], 16, 12)),
                  ("topTokens", literal(r["top_tokens"], 16, 12)),
                  ("topLogprobs", literal(r["top_logprobs"], 5, 12, flt)),
                  ("lastLogprobs", literal(r["last_logprobs"], 5, 12, flt))]
        lines.append("        Run(")
        for n, (key, value) in enumerate(fields):
            end = "," if n < len(fields) - 1 else ""
            if isinstance(value, list):
                lines += [f"            {key}: {value[0]}", *value[1:-1], value[-1] + end]
            else:
                lines.append(f"            {key}: {value}{end}")
        lines.append("        ),")
    lines += ["    ]", "}", ""]
    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference-dir", required=True, help="DeepSeek-V4's inference/ folder")
    reference_dir = parser.parse_args().reference_dir
    load_reference(reference_dir)
    checkpoint, table, state = weights(SEED)
    rng = np.random.default_rng(SEED)
    prompts = {n: rng.integers(2, CONFIG["vocab_size"], size=n).tolist() for n, _ in CASES}
    runs, margins, sensitivity, worst = [], {}, {}, 0.0
    for mode, (block, keep) in INDEXER.items():
        for n in prompts:
            hisa = (block, keep) if block else None
            logprobs, steps = run(state, prompts[n], hisa)
            case_margins, case_sensitivity, case_worst = check_margins(
                state, prompts[n], hisa, logprobs, steps)
            for kind in case_margins:
                margins[kind] = min(margins.get(kind, math.inf), case_margins[kind])
                sensitivity[kind] = max(sensitivity.get(kind, 0.0), case_sensitivity[kind])
            worst = max(worst, case_worst)
            top = [np.argsort(-lp, kind="stable")[:TOP_K] for lp in logprobs]
            runs.append({
                "indexer": mode, "block": block, "keep": keep, "prompt": prompts[n],
                "steps": [s for m, s in CASES if m == n],
                "greedy": [int(np.argmax(lp)) for lp in logprobs],
                "experts": [e for s in steps for layer in s[0] for e in layer],
                "indexer_selection": [e for s in steps for layer in s[1] for e in layer],
                "top_tokens": [int(t) for ts in top for t in ts],
                "top_logprobs": [float(lp[t]) for lp, ts in zip(logprobs, top) for t in ts],
                "last_logprobs": logprobs[-1].tolist(),
            })
    for kind in sorted(margins):
        threshold = max(MARGIN_FLOOR, MARGIN_FACTOR * sensitivity[kind])
        print(f"{kind}: smallest margin {margins[kind]:.3g}, threshold {threshold:.3g}")
        if margins[kind] < threshold:
            raise SystemExit(f"seed {SEED} rejected: {kind} margin below threshold")
    sha = hashlib.sha256(open(os.path.join(reference_dir, "model.py"), "rb").read()).hexdigest()
    config = dict(CONFIG, quantization={"group_size": 32, "bits": 4, "mode": "mxfp4", **{
        name[: -len(".weight")]: {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        for name, _, kind in table if kind == "mxfp8"}})
    write_swift(config, table, checksum(checkpoint), runs, sha)
    print(f"largest log-prob change under re-rounding: {worst:.3g}; wrote {os.path.normpath(OUTPUT)}")


if __name__ == "__main__":
    main()
