#!/usr/bin/env python3
"""Generate Swift FP32 reference values from DeepSeek's DeepSeek-V4 reference implementation.

Requires torch, numpy and huggingface_hub. Run from the repository root:
python tools/deepseek-v4-reference.py [--reference-dir <local copy of inference/>]
The pinned inference/model.py runs on the CPU in FP32 with its CUDA kernels replaced by
tools/deepseek-v4-reference/ (activation quantization and Hadamard rotation are identity).
Weights come from a hash, not LFM3's periodic formula, so router and indexer scores do not tie.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

REPO = "deepseek-ai/DeepSeek-V4-Flash"
REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
MODEL_SHA256 = "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71"
SHIM = Path(__file__).resolve().parent / "deepseek-v4-reference"
OUTPUT = Path(__file__).resolve().parents[1] / "Tests/MLXLMTests/DeepseekV4ReferenceValues.swift"
SEED, DECODE, TOP_K, TOLERANCE = 149, 24, 4, 2e-5
CASES = {"short": (45, [16]), "long": (250, [64, 2048])}  # prompt length, Swift prefill steps
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
    "index_n_heads": 16, "index_head_dim": 32, "index_topk": 4,
    "tie_word_embeddings": False, "bos_token_id": 0, "eos_token_id": 1,
}
DTYPES = ["bfloat16", "float32", "int32", "uint32", "uint8"]  # checksum codes, as in Swift
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
E4M3 = np.array([(-1) ** (c >> 7) * ((c & 7) / 8 * 2.0 ** -6 if (c >> 3) & 15 == 0
                                     else (1 + (c & 7) / 8) * 2.0 ** (((c >> 3) & 15) - 7))
                 for c in range(256)])
ref = None


def load_reference(directory):
    global ref
    sha = hashlib.sha256((directory / "model.py").read_bytes()).hexdigest()
    if sha != MODEL_SHA256:
        raise SystemExit(f"{directory / 'model.py'} has sha256 {sha}, expected {MODEL_SHA256}")
    sys.path[:0] = [str(SHIM), str(directory)]
    torch.set_default_dtype(torch.float32)
    import model as ref
    # Identity: without activation quantization the rotation cancels in the indexer's products.
    ref.rotate_activation = torch.clone


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


def weights():
    """The checkpoint as the MXFP4 MLX conversion stores it, the (name, shape, storage) table
    and the reference state dict it decodes to. Same formula as checkpoint() in the Swift test."""
    shapes = sorted((k, tuple(v.shape)) for k, v in ref.Transformer(model_args()).named_parameters())
    checkpoint, table, state = {}, [], {}
    for i, (name, shape) in enumerate(shapes):
        n, kind = math.prod(shape), storage(name)
        x = (SEED * 2**23 + i * 2**15 + np.arange(n + n // 32, dtype=np.int64)) % (2**31 - 1)
        h = (x * x % (2**31 - 1)) * 48271 % (2**31 - 1)
        if kind == "int32":  # hash routing: distinct experts per token
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
            else:  # the e4m3 code of an integer in -16...16
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
    """h = h * 1000003 + v (mod 2**64) over the tensors in sorted name order, v the dtype code,
    the shape, then each element's float32 bit pattern or integer value."""
    h = 0
    for name in sorted(checkpoint):
        a = checkpoint[name]
        dtype = storage(name) if a.dtype == np.float32 else str(a.dtype)
        flat = np.ascontiguousarray(a).reshape(-1)
        values = (flat.view(np.uint32) if a.dtype in (np.float32, np.int32) else flat).tolist()
        for v in [DTYPES.index(dtype), *a.shape, *values]:
            h = (h * 1000003 + v) & (2**64 - 1)
    return h


def margin(scores, k):
    """Smallest k-th minus (k+1)-th largest finite score over the rows (inf if none rejects)."""
    best = math.inf
    for row in scores.double().reshape(-1, scores.size(-1)):
        v = row[torch.isfinite(row)].sort(descending=True).values
        if v.numel() > k:
            best = min(best, float(v[k - 1] - v[k]))
    return best


def run(state, prompt, margins):
    """One-shot prefill of prompt[:-1], then DECODE greedy steps from the last prompt token.
    Returns the log-probs per step; lowers `margins` to each selection's smallest margin."""
    model = ref.Transformer(model_args()).float()
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(f"state dict mismatch: {result}")
    model.eval()

    def note(kind, scores, k, picked, expected):
        if not torch.equal(picked, expected):
            raise SystemExit(f"recomputed {kind} scores do not give the reference's selection")
        margins[kind] = min(margins.get(kind, math.inf), margin(scores, k))

    def gate_hook(gate, args, out):  # Gate.forward's selection scores
        if not gate.hash:
            scores = ref.F.softplus(ref.linear(args[0].float(), gate.weight.float())).sqrt() + gate.bias
            note("moe", scores, gate.topk, scores.topk(gate.topk)[1], out[1])

    def indexer_hook(indexer, args, out):  # Indexer.forward's index_score
        x, qr, start, offset = args
        n, ratio = x.size(1), indexer.compress_ratio
        q = indexer.wq_b(qr).unflatten(-1, (indexer.n_local_heads, indexer.head_dim))
        ref.apply_rotary_emb(q[..., -indexer.rope_head_dim:], indexer.freqs_cis[start:start + n])
        w = indexer.weights_proj(x) * (indexer.softmax_scale * indexer.n_heads ** -0.5)
        kv = indexer.kv_cache[:1, :(start + n) // ratio]
        scores = (torch.einsum("bshd,btd->bsht", q, kv).relu() * w.unsqueeze(-1)).sum(dim=2)
        visible = torch.arange(1, n + 1).unsqueeze(1) // ratio if start == 0 else kv.size(1)
        scores = torch.where(torch.arange(kv.size(1)) < visible, scores, -math.inf)
        k = min(indexer.index_topk, kv.size(1))
        picked = scores.topk(k)[1]
        picked = torch.where(picked >= visible, -1, picked + offset)
        note("indexer", scores, k, picked, out)

    for block in model.layers:
        block.ffn.gate.register_forward_hook(gate_hook)
        if (indexer := getattr(block.attn, "indexer", None)) is not None:
            indexer.register_forward_hook(indexer_hook)
    with torch.no_grad():
        model(torch.tensor([prompt[:-1]]), 0)
        token, logprobs = prompt[-1], []
        for t in range(DECODE):
            logits = model(torch.tensor([[token]]), len(prompt) - 1 + t)[0].float()
            logprobs.append(torch.log_softmax(logits, -1))
            token = int(logprobs[-1].argmax())
            margins["argmax"] = min(margins.get("argmax", math.inf), margin(logprobs[-1], 1))
    return logprobs


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
    return "[\n" + "\n".join(rows) + "\n" + " " * indent + "]"


def write_swift(config, table, digest, runs, margins, sha):
    flt = lambda v: f"{float(np.float32(v)):.9g}"  # noqa: E731
    lines = [
        "// Generated by tools/deepseek-v4-reference.py; do not edit values by hand.",
        f"// {REPO} {REVISION} inference/model.py, sha256",
        f"// {sha}; PyTorch {torch.__version__} CPU FP32.",
        "// Stubbed: kernel.py by tools/deepseek-v4-reference/kernel.py (FP32 sparse_attn and",
        "// hc_split_sinkhorn, act_quant and fp4_act_quant identity); rotate_activation identity.",
        "",
        "enum DeepseekV4ReferenceValues {",
        '    static let configuration = """',
        *["        " + line for line in json_lines(config)],
        '        """',
        "",
        "    enum Storage { case bfloat16, float32, int32, mxfp4, mxfp8 }",
        "",
        f"    static let weightSeed = {SEED}",
        "    static let parameters: [(name: String, shape: [Int], storage: Storage)] = [",
        *[f'        ("{name}", {list(shape)}, .{kind}),' for name, shape, kind in table],
        "    ]",
        f"    static let checkpointChecksum: UInt64 = {digest:_}",
        "",
        f"    static let decode = {DECODE}",
        f"    static let topK = {TOP_K}",
        "    /// Smallest top-k boundary margin of each selection in the reference runs.",
        "    static let selectionMargins: [String: Float] = [",
        *[f'        "{kind}": {flt(m)},' for kind, m in sorted(margins.items())],
        "    ]",
        "",
        "    struct Run {",
        "        let prompt: [Int]",
        "        let prefillSteps: [Int]",
        "        let greedy: [Int]",
        "        let topTokens: [Int]",
        "        let topLogprobs: [Float]",
        "    }",
    ]
    for name, r in runs.items():
        lines += [
            "",
            f"    static let {name} = Run(",
            f"        prompt: {literal(r['prompt'], 16, 8)},",
            f"        prefillSteps: {r['steps']},",
            f"        greedy: {literal(r['greedy'], 16, 8)},",
            f"        topTokens: {literal(r['top_tokens'], 16, 8)},",
            f"        topLogprobs: {literal(r['top_logprobs'], 4, 8, flt)})",
        ]
    lines += ["}", ""]
    OUTPUT.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-dir", type=Path, help="local copy of inference/ to use")
    directory = parser.parse_args().reference_dir
    with TemporaryDirectory(prefix="deepseek-v4-reference-") as source:
        if directory is None:
            from huggingface_hub import snapshot_download
            snapshot_download(REPO, revision=REVISION, allow_patterns=["inference/*"],
                              local_dir=source)
            directory = Path(source) / "inference"
        load_reference(directory.resolve())
    torch.set_num_threads(4)
    checkpoint, table, state = weights()
    rng = np.random.default_rng(SEED)
    runs, margins = {}, {}
    for name, (n, steps) in CASES.items():
        prompt = rng.integers(2, CONFIG["vocab_size"], size=n).tolist()
        logprobs = run(state, prompt, margins)
        top = [lp.argsort(descending=True, stable=True)[:TOP_K].tolist() for lp in logprobs]
        runs[name] = {
            "prompt": prompt, "steps": steps, "greedy": [int(lp.argmax()) for lp in logprobs],
            "top_tokens": [t for ts in top for t in ts],
            "top_logprobs": [float(lp[t]) for lp, ts in zip(logprobs, top) for t in ts],
        }
    for kind, m in sorted(margins.items()):
        print(f"{kind}: smallest margin {m:.3g}")
        if m <= 2 * TOLERANCE:
            raise SystemExit(f"seed {SEED} rejected: {kind} margin {m:.3g} <= {2 * TOLERANCE}")
    config = dict(CONFIG, quantization={"group_size": 32, "bits": 4, "mode": "mxfp4", **{
        name[: -len(".weight")]: {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        for name, _, kind in table if kind == "mxfp8"}})
    write_swift(config, table, checksum(checkpoint), runs, margins, MODEL_SHA256)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
