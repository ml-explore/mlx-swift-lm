"""Writes Tests/MLXLMTests/Resources/glm5_next_tiny (config.json, model.safetensors, oracle.json)
from the model author's own implementation, `transformers.models.glm5_next` (transformers
5.22.2, torch CPU). Nothing in transformers is copied or modified.

The fixture is a 5-layer GLM-5-Next with seeded random weights: layer types linear (KDA),
indexed, indexed with a shared indexer, linear, indexed; dense feed-forward in the first two
layers and MoE in the last three; hyper-connections everywhere. The shared-indexer layer follows
an indexed layer directly: transformers rejects a shared layer after a linear one (its decoder
loop hands the next layer `None` after every linear layer). With `index_kpool` 4 and
`index_topk` 8 the indexer keeps 2 pools plus the open tail, so the 32-token prompt (8 pools)
makes its selection a strict subset.

config.json and model.safetensors are what `Glm5NextForConditionalGeneration.save_pretrained`
writes (checkpoint layout: `hc_attn_*`, per-expert projections, `q/k/v_conv1d`, top-level
forget-gate parameters, `indexed_attention` layer type, a tiny vision tower).

The oracle is the library model run in float32 on the CPU (`attn_implementation="sdpa"`) along
its own greedy tokens, in four prefill plans: one call, chunks ending at 5/13/22/32, chunks of 7,
and one token at a time; each continues with one-token decode steps fed the greedy tokens. Every
plan's log-probabilities are stored as the library produced them. Token by token is an equality
case: with exact ties removed by the margin rule below the library's decode path selects the
same pools as its one-shot path (the earlier mismatch was ReLU-clamped indexer scores tied at
exactly 0, where `topk` breaks ties differently for different pool counts).

Weight seed rule: a seed is kept only if every discrete choice (indexer top-k per query and
layer, MoE top-k per token and layer, the argmax at every position) has a margin at least
MARGIN_FACTOR times the largest change of that kind of score seen when the same tokens are run
(1) through the other prefill plans, (2) with the model cast to float64, and (3) with every
`torch.nn.functional.linear` output computed in float64, rounded to float32 and moved by random
-2..2 ulps (two noise seeds, one-call and token-by-token plans). The tolerance is
TOLERANCE_FACTOR times the largest log-probability change seen in those runs.

Monkeypatches, all observation or perturbation, none in the oracle runs' arithmetic:
  1. `torch.topk` and `torch.Tensor.topk` are wrapped to record the scores and picks of the
     indexer and router calls (values untouched).
  2. Perturbation runs only: `torch.nn.functional.linear` as described in (3) above.

Usage, from the repository root, with a python that has transformers 5.22.2 and torch:

    python -B tools/glm5_next_parity/make_tiny_fixture.py
"""

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import transformers
from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration
from transformers.models.glm5_next import modeling_glm5_next as glm

TOOLS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(TOOLS, "..", "..", "Tests", "MLXLMTests", "Resources", "glm5_next_tiny")
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
def build(seed):
    config = Glm5NextConfig(text_config=dict(TEXT), vision_config=dict(VISION))
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Glm5NextForConditionalGeneration(config).float().eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, value in sorted(model.state_dict().items()):
            noise = torch.randn(value.shape, generator=g)
            if name.endswith("norm.weight"):
                value.copy_(1 + 0.1 * noise)
            elif name.endswith("_hc.scale"):
                value.copy_(0.5 + 0.1 * noise)
            elif name.endswith("e_score_correction_bias"):
                value.copy_(0.1 * noise)
            else:
                value.copy_(0.15 * noise)
    prompt = torch.randint(0, TEXT["vocab_size"], (PROMPT_LEN,), generator=g).tolist()
    return model, prompt


# ------------------------------------------------------------------ observation (patch 1)
REC = {"on": False, "ctx": None, "layer": None, "start": 0, "rows": None}
_topk_fn, _topk_method, _linear = torch.topk, torch.Tensor.topk, torch.nn.functional.linear


def _record(scores, k, indices):
    kind = REC["ctx"]
    if kind == "moe" and scores.shape[-1] != TEXT["n_routed_experts"]:
        return  # the group pre-selection (one group)
    if scores.shape[-1] == 0:
        return  # no complete pool yet
    s =scores.detach().double().reshape(-1, scores.shape[-1]).numpy()
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
def run(model, tokens, plan, dtype=torch.float32):
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
            out = model(input_ids=torch.tensor([[tokens[-1]]]), past_key_values=cache, use_cache=True)
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


def evaluate(seed):
    model, prompt = build(seed)
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
            return None, dict(report, reason=f"{kind} margin {m:.3g} < threshold {thresholds[kind]:.3g}")

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
    return (model, prompt, tokens_greedy, runs), report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED)
    parser.add_argument("--max-seeds", type=int, default=MAX_SEEDS)
    parser.add_argument("--out", default=OUT)
    opts = parser.parse_args()
    if transformers.__version__ != TRANSFORMERS_VERSION:
        sys.exit(f"transformers {TRANSFORMERS_VERSION} required, found {transformers.__version__}")
    torch.set_num_threads(1)

    rejected = []
    for seed in range(opts.first_seed, opts.first_seed + opts.max_seeds):
        result, report = evaluate(seed)
        reason = report if isinstance(report, str) else report.get("reason")
        print(f"seed {seed}: {'accepted' if result else reason}", flush=True)
        if result:
            break
        rejected.append(f"{seed}: {reason}")
    else:
        sys.exit(f"no seed in {opts.max_seeds} met the margin requirements")
    model, prompt, tokens_greedy, runs = result

    os.makedirs(opts.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, safe_serialization=True)
        for name in ("config.json", "model.safetensors"):
            shutil.copyfile(os.path.join(tmp, name), os.path.join(opts.out, name))

    tolerance = float(f"{max(1e-5, TOLERANCE_FACTOR * report['max_logprob_change']):.2g}")
    provenance = {
        "generator": "tools/glm5_next_parity/make_tiny_fixture.py",
        "reference": f"transformers {transformers.__version__} "
                     "Glm5NextForConditionalGeneration (sdpa), unmodified",
        "torch": torch.__version__,
        "device": "cpu",
        "mode": "float32 parameters and activations",
        "weight_seed": seed,
        "seeds_rejected": rejected,
        "margin_factor": MARGIN_FACTOR,
        "margin_floor": MARGIN_FLOOR,
        "tolerance_factor": TOLERANCE_FACTOR,
        **report,
    }
    cases = [{"name": name, "prefill_ends": PLANS[name],
              "logprobs": [[float(f"{v:.9g}") for v in row] for row in runs[name][0]]}
             for name in PLANS]
    with open(os.path.join(opts.out, "oracle.json"), "w") as f:
        json.dump({"compute_dtype": "float32", "tolerance": tolerance, "prompt": prompt,
                   "greedy": tokens_greedy, "provenance": provenance, "cases": cases},
                  f, separators=(",", ":"))
    print(json.dumps(dict(provenance, tolerance=tolerance), indent=1))


if __name__ == "__main__":
    main()
