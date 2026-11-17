#!/usr/bin/env python3
"""Generate Tests/MLXLMTests/GLM5NextReferenceValues.swift from transformers' own glm5_next.

Requires torch, transformers==5.22.2, safetensors and numpy.
Run from the repository root: python -B tools/glm5-next-reference.py
The generated file's header states how the weights, seed and tolerance are chosen.
"""

import argparse
import contextlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from safetensors.numpy import save_file
from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration

OUTPUT = Path(__file__).resolve().parents[1] / "Tests/MLXLMTests/GLM5NextReferenceValues.swift"
TRANSFORMERS_VERSION = "5.22.2"

TEXT = {
    "model_type": "glm5_next_text", "vocab_size": 64, "hidden_size": 32,
    "intermediate_size": 64, "moe_intermediate_size": 16,
    "num_hidden_layers": 5, "num_attention_heads": 2, "num_key_value_heads": 2,
    "n_shared_experts": 1, "n_routed_experts": 4, "num_experts_per_tok": 2,
    "routed_scaling_factor": 2.5, "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
    "kv_lora_rank": 16, "q_lora_rank": 16, "qk_rope_head_dim": 0,
    "qk_nope_head_dim": 16, "v_head_dim": 16,
    "rms_norm_eps": 1e-5, "tie_word_embeddings": False, "attention_bias": False,
    # Eight indexer heads: with two, a quarter of the pool scores are ReLU-clamped to exactly 0
    # and the top-k boundary falls on ties.
    "index_topk": 8, "index_head_dim": 8, "index_n_heads": 8,
    "index_kpool": 4, "index_kpool_always_select_tail": True,
    "layer_types": ["linear_attention", "indexed_attention", "indexed_attention",
                    "linear_attention", "indexed_attention"],
    "mlp_layer_types": ["dense", "dense", "sparse", "sparse", "sparse"],
    "indexer_types": ["full", "full", "shared", "full", "full"],
    # Low enough that the clamps act on some activations.
    "swiglu_limit": 2.0,
    "linear_attn_config": {"num_heads": 2, "head_dim": 16, "short_conv_kernel_size": 4,
                           "gate_lower_bound": -5.0},
    # Four Sinkhorn passes (the released model uses 20) so each pass changes the mix.
    "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 4,
    "pad_token_id": None,
}
VISION = {"depth": 1, "hidden_size": 16, "num_heads": 2, "intermediate_size": 16,
          "out_hidden_size": 32, "projection_intermediate_size": 16, "patch_size": 2,
          "image_size": 8, "spatial_merge_size": 2, "temporal_patch_size": 2}

PROMPT_LEN = 32
DECODE = 6
PLANS = {
    "one_call": [32],
    "chunks_5_13_22_32": [5, 13, 22, 32],
    "chunks_of_7": [7, 14, 21, 28, 32],
    "token_by_token": list(range(1, 33)),
}
FIRST_SEED = 1
MAX_SEEDS = 200
MARGIN_FACTOR = 20.0
MARGIN_FLOOR = 1e-5
TOLERANCE_FACTOR = 10.0
NOISE_SEEDS = (1, 2)


# ------------------------------------------------------------------ model and weights
def weight_values(name, index, shape, seed):
    """Checkpoint tensor `index` (in sorted name order): a hash of the flat element index, the
    tensor index and the seed, mod 65521, centred, then one float32 division. Norm weights are
    1 +- 1/6, hyper-connection scales 0.5 +- 1/6, correction biases +- 1/6, the rest +- 1/4."""
    j = np.arange(int(np.prod(shape)), dtype=np.int64)
    h = (j * 7919 + index * 104729 + seed * 1299709 + 12345) % 65521
    h = (h * h + 7) % 65521
    h = (h * h + 13) % 65521
    numerator = h - 32760
    if name.endswith("norm.weight"):
        denominator = 196608
        numerator = numerator + denominator
    elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
        denominator = 196608
        numerator = numerator + denominator // 2
    elif name.endswith("e_score_correction_bias"):
        denominator = 196608
    else:
        denominator = 131072
    return (numerator.astype(np.float32) / np.float32(denominator)).reshape(shape)


def checkpoint_layout():
    """config.json text and sorted (name, shape) of what `save_pretrained` writes."""
    config = Glm5NextConfig(text_config=dict(TEXT), vision_config=dict(VISION))
    model = Glm5NextForConditionalGeneration(config).float()
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, safe_serialization=True)
        with safe_open(os.path.join(tmp, "model.safetensors"), "np") as f:
            layout = [(name, list(f.get_slice(name).get_shape())) for name in sorted(f.keys())]
        config_text = Path(tmp, "config.json").read_text()
    return config_text, layout


def checkpoint_weights(layout, seed):
    return {name: weight_values(name, index, shape, seed)
            for index, (name, shape) in enumerate(layout)}


def build(config_text, layout, seed):
    weights = checkpoint_weights(layout, seed)
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "config.json").write_text(config_text)
        save_file(weights, os.path.join(tmp, "model.safetensors"), metadata={"format": "pt"})
        model, info = Glm5NextForConditionalGeneration.from_pretrained(
            tmp, dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        sys.exit(f"checkpoint does not load cleanly: {info}")
    model.eval()
    prompt = np.random.default_rng(seed).integers(0, TEXT["vocab_size"], PROMPT_LEN).tolist()
    return model, prompt, weights


# ------------------------------------------------------------------ observation (patch 1)
# torch.topk is wrapped to record the indexer's and router's scores and picks; values untouched.
REC = {"on": False, "ctx": None, "layer": None, "start": 0, "rows": None}
_topk_fn, _topk_method, _linear = torch.topk, torch.Tensor.topk, torch.nn.functional.linear


def _record(scores, k, indices):
    kind = REC["ctx"]
    if kind == "moe" and scores.shape[-1] != TEXT["n_routed_experts"]:
        return  # the group pre-selection (one group)
    if scores.shape[-1] == 0:
        return  # no complete pool yet
    s = scores.detach().double().reshape(-1, scores.shape[-1]).numpy()
    picks = indices.detach().reshape(-1, indices.shape[-1]).numpy()
    for i, row in enumerate(s):
        valid = row > -1e30
        if not valid.any():
            continue
        chosen = sorted(int(p) for p in picks[i] if valid[p])
        REC["rows"][(kind, REC["layer"], REC["start"] + i)] = (row[valid].copy(), k, chosen)


def topk_fn(input, k, *args, **kwargs):
    out = _topk_fn(input, k, *args, **kwargs)
    if REC["on"] and REC["ctx"]:
        _record(input, k, out[1])
    return out


def topk_method(self, k, *args, **kwargs):
    out = _topk_method(self, k, *args, **kwargs)
    if REC["on"] and REC["ctx"]:
        _record(self, k, out[1])
    return out


torch.topk, torch.Tensor.topk = topk_fn, topk_method


def install_context_hooks(model):
    layers = model.model.language_model.layers
    for index, layer in enumerate(layers):
        for module, kind in ((getattr(layer.self_attn, "indexer", None), "indexer"),
                             (getattr(layer.mlp, "gate", None), "moe")):
            if module is None:
                continue

            def pre(_m, _a, kind=kind, index=index):
                REC["ctx"], REC["layer"] = kind, index

            def post(_m, _a, _o):
                REC["ctx"] = None

            module.register_forward_pre_hook(pre)
            module.register_forward_hook(post)


# ------------------------------------------------------------------ perturbation (patch 2)
# Perturbation runs only: linear outputs in float64, rounded, moved by -2..2 float32 ulps.
PERT = {"rng": None}


def linear_hook(x, w, bias=None):
    if PERT["rng"] is None:
        return _linear(x, w, bias)
    exact = _linear(x.double(), w.double(), None if bias is None else bias.double()).float()
    steps = torch.from_numpy(PERT["rng"].integers(-2, 3, size=tuple(exact.shape)).astype(np.int32))
    moved = (exact.view(torch.int32) + torch.where(exact > 0, steps, -steps)).view(torch.float32)
    return torch.where((exact != 0) & torch.isfinite(exact), moved, exact).to(x.dtype)


torch.nn.functional.linear = linear_hook


@contextlib.contextmanager
def noise(seed):
    PERT["rng"] = None if seed is None else np.random.default_rng(seed)
    try:
        yield
    finally:
        PERT["rng"] = None


# ------------------------------------------------------------------ runs
@torch.no_grad()
def run(model, tokens, plan):
    """Prefill tokens[:PROMPT_LEN] in the plan's chunks, then one-token steps over the rest.
    Returns float32 log-probs per position [len(tokens), V] and the recorded selections."""
    REC.update(on=True, rows={}, ctx=None)
    cache, start, rows = None, 0, []
    ends = plan + list(range(PROMPT_LEN + 1, len(tokens) + 1))
    for end in ends:
        REC["start"] = start
        ids = torch.tensor([tokens[start:end]])
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[0].to(torch.float64)
        rows.append((logits - torch.logsumexp(logits, -1, keepdim=True)).float())
        start = end
    REC["on"] = False
    return torch.cat(rows).numpy(), REC["rows"]


def greedy(model, prompt):
    """The library's free-running greedy decode after a one-call prefill."""
    tokens = list(prompt)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([tokens]), use_cache=True)
        cache = out.past_key_values
        for _ in range(DECODE):
            tokens.append(int(out.logits[0, -1].argmax()))
            if len(tokens) == PROMPT_LEN + DECODE:
                break
            out = model(input_ids=torch.tensor([[tokens[-1]]]), past_key_values=cache,
                        use_cache=True)
            cache = out.past_key_values
    return tokens[PROMPT_LEN:]


@torch.no_grad()
def greedy_uncached(model, prompt):
    tokens = list(prompt)
    for _ in range(DECODE):
        tokens.append(int(model(input_ids=torch.tensor([tokens])).logits[0, -1].argmax()))
    return tokens[PROMPT_LEN:]


def margins(logprobs, rows):
    """Smallest margin per kind: k-th largest minus (k+1)-th of the valid scores."""
    out = {}
    for (kind, _layer, _pos), (values, k, _chosen) in rows.items():
        if values.size > k:
            v = np.sort(values)[::-1]
            out[kind] = min(out.get(kind, math.inf), float(v[k - 1] - v[k]))
    top2 = np.sort(logprobs, axis=-1)[:, -2:]
    out["argmax"] = float((top2[:, 1] - top2[:, 0]).min())
    return out


def evaluate(config_text, layout, seed):
    model, prompt, weights = build(config_text, layout, seed)
    install_context_hooks(model)
    tokens_greedy = greedy(model, prompt)
    tokens = prompt + tokens_greedy[:-1]
    runs = {name: run(model, tokens, plan) for name, plan in PLANS.items()}
    base_lp, base_rows = runs["one_call"]

    smallest = {}
    for lp, rows in runs.values():
        for kind, m in margins(lp, rows).items():
            smallest[kind] = min(smallest.get(kind, math.inf), m)
    if min(smallest.values()) < MARGIN_FLOOR:
        kind = min(smallest, key=smallest.get)
        return None, f"{kind} margin {smallest[kind]:.3g} < floor {MARGIN_FLOOR:g}"

    perturbed = dict(runs)
    model64 = Glm5NextForConditionalGeneration(model.config).eval()
    model64.load_state_dict(model.state_dict())
    model64 = model64.double()
    # grouped_mm has no float64 kernel; the experts' own loop does.
    model64.config._experts_implementation = "eager"
    install_context_hooks(model64)
    perturbed["float64"] = run(model64, tokens, PLANS["one_call"])
    for s in NOISE_SEEDS:
        for name in ("one_call", "token_by_token"):
            with noise(s):
                perturbed[f"noise{s}_{name}"] = run(model, tokens, PLANS[name])

    change = {}
    for name, (lp, rows) in perturbed.items():
        change["argmax"] = max(change.get("argmax", 0.0), float(np.abs(lp - base_lp).max()))
        for key, (values, _k, chosen) in rows.items():
            ref_values, _, ref_chosen = base_rows[key]
            if chosen != ref_chosen:
                return None, f"{name}: {key} picks {chosen}, one call picks {ref_chosen}"
            change[key[0]] = max(change.get(key[0], 0.0), float(np.abs(values - ref_values).max()))
        if name in runs:
            g = [int(x) for x in lp[PROMPT_LEN - 1:].argmax(-1)]
            if g != tokens_greedy:
                return None, f"{name}: greedy {g} != {tokens_greedy}"
    thresholds = {k: max(MARGIN_FLOOR, MARGIN_FACTOR * v) for k, v in change.items()}
    report = {"min_margin": smallest, "max_score_change": change, "margin_threshold": thresholds}
    for kind, m in smallest.items():
        if m < thresholds[kind]:
            reason = f"{kind} margin {m:.3g} < threshold {thresholds[kind]:.3g}"
            return None, dict(report, reason=reason)

    worst = max(float(np.abs(lp - base_lp).max()) for lp, _ in perturbed.values())
    uncached = greedy_uncached(model, prompt)
    assert uncached == tokens_greedy, (uncached, tokens_greedy)
    report["max_logprob_change"] = worst
    report["decisions"] = {
        kind: sum(1 for key, (v, k, _c) in base_rows.items() if key[0] == kind and v.size > k)
        for kind in ("indexer", "moe")}
    # The indexer matters: the same weights with every pool selected change the log-probs.
    every = Glm5NextConfig(text_config=dict(TEXT, index_topk=64), vision_config=dict(VISION))
    every._attn_implementation = "sdpa"
    dense = Glm5NextForConditionalGeneration(every).float().eval()
    dense.load_state_dict(model.state_dict())
    dense_lp, _ = run(dense, tokens, PLANS["one_call"])
    report["max_logprob_change_with_every_pool_selected"] = float(np.abs(dense_lp - base_lp).max())
    report["self_consistency_max_abs_logprob_vs_one_call"] = {
        name: float(np.abs(lp - base_lp).max()) for name, (lp, _) in runs.items()}

    # The library holds exactly the formula's values: saving it again gives the same tensors.
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, safe_serialization=True)
        with safe_open(os.path.join(tmp, "model.safetensors"), "np") as f:
            assert sorted(f.keys()) == sorted(weights)
            for name, value in weights.items():
                assert np.array_equal(f.get_tensor(name), value), name
    return (prompt, tokens_greedy, base_lp), report


def packed(items, indent="        ", width=100):
    """Comma-terminated items packed into lines of at most `width` columns."""
    lines, line = [], indent
    for item in items:
        if line != indent and len(line) + len(item) + 2 > width:
            lines.append(line.rstrip())
            line = indent
        line += item + ", "
    return lines + [line.rstrip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED)
    parser.add_argument("--max-seeds", type=int, default=MAX_SEEDS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    opts = parser.parse_args()
    if transformers.__version__ != TRANSFORMERS_VERSION:
        sys.exit(f"transformers {TRANSFORMERS_VERSION} required, found {transformers.__version__}")
    torch.set_num_threads(1)
    torch.manual_seed(0)

    config_text, layout = checkpoint_layout()
    rejected = []
    for seed in range(opts.first_seed, opts.first_seed + opts.max_seeds):
        result, report = evaluate(config_text, layout, seed)
        reason = report if isinstance(report, str) else report.get("reason")
        print(f"seed {seed}: {'accepted' if result else reason}", flush=True)
        if result:
            break
        rejected.append(f"{seed}: {reason}")
    else:
        sys.exit(f"no seed in {opts.max_seeds} met the margin requirements")
    prompt, tokens_greedy, logprobs = result

    tolerance = float(f"{max(1e-5, TOLERANCE_FACTOR * report['max_logprob_change']):.2g}")
    consistency = max(report["self_consistency_max_abs_logprob_vs_one_call"].values())
    assert consistency <= tolerance / TOLERANCE_FACTOR, (consistency, tolerance)

    margin = report["min_margin"]
    lines = [
        "// Generated by tools/glm5-next-reference.py; do not edit values by hand.",
        f"// transformers {transformers.__version__} Glm5NextForConditionalGeneration (sdpa), "
        f"unmodified; PyTorch {torch.__version__} CPU FP32.",
        "//",
        "// A 5-layer model: KDA, indexed, indexed with a shared indexer, KDA, indexed;",
        "// dense MLP in the first two layers, MoE in the rest. index_kpool 4 and",
        "// index_topk 8 keep 2 of the 8 pools of the 32-token prompt plus the open tail.",
        "// Every tensor of the checkpoint layout save_pretrained writes is filled by",
        "// `weight_values` (integer hash, one float32 division) from its sorted-name index,",
        "// its element index and the weight seed, then loaded with from_pretrained. The",
        "// reference is teacher-forced on its own greedy tokens in four prefill plans; the",
        "// one-call log-probabilities are stored, and the other plans agree with them",
        "// within a tenth of the tolerance.",
        "//",
        "// A seed is kept only if every discrete choice (indexer top-k, MoE top-k, argmax)",
        f"// has a margin of at least {MARGIN_FACTOR:g} x the largest change of that score over",
        "// the other prefill plans, a float64 run, and float64 linear outputs moved by +-2",
        "// float32 ulps (two noise seeds). The tolerance is "
        f"{TOLERANCE_FACTOR:g} x the largest",
        "// log-probability change over those runs.",
        f"// Weight seed {seed}; smallest margins: indexer {margin['indexer']:.3g}, "
        f"MoE {margin['moe']:.3g}, argmax {margin['argmax']:.3g}.",
        f"// Tolerance {tolerance:g} = {TOLERANCE_FACTOR:g} x {report['max_logprob_change']:.3g}.",
        "",
        "enum GLM5NextReferenceValues {",
        "    /// config.json as `save_pretrained` writes it.",
        '    static let configuration = """',
        *["        " + line for line in config_text.rstrip("\n").splitlines()],
        '        """',
        "",
        "    /// Checkpoint tensor names and shapes as `save_pretrained` writes them, sorted.",
        "    static let checkpointTensors: [(String, [Int])] = [",
        *[f'        ("{name}", {shape}),' for name, shape in layout],
        "    ]",
        "",
        f"    static let weightSeed = {seed}",
        f"    static let tolerance: Float = {tolerance!r}",
        "    static let prompt = [",
        *packed(str(t) for t in prompt),
        "    ]",
        f"    static let greedy = {tokens_greedy}",
        "",
        "    /// One-call log-softmax at every position of `prompt + greedy.dropLast()`,",
        f"    /// [{logprobs.shape[0]}, {logprobs.shape[1]}] row-major.",
        "    static let logprobs: [Float] = [",
        *packed(f"{float(v):.9g}" for v in logprobs.reshape(-1)),
        "    ]",
        "}",
        "",
    ]
    opts.output.write_text("\n".join(lines))
    print(json.dumps(dict(report, weight_seed=seed, seeds_rejected=rejected, tolerance=tolerance,
                          torch=torch.__version__), indent=1))
    print(f"Wrote {opts.output} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
