"""Writes Tests/MLXLMTests/Resources/deepseek_v4_tiny (config, weights, oracle.json).

A 4-layer DeepSeek-V4 with seeded random weights (compress ratios 0, 4, 128, 4, one hash-routed
layer, an 8-token window), quantized and laid out like the DeepSeek-V4-Flash MXFP4 MLX
conversion. The oracle is mlx-vlm's deepseek_v4 run in float32 on the CPU, teacher-forced along
its own greedy tokens; a weight seed is kept only if every discrete selection (experts, indexer
top-k, greedy argmax) has a wide margin under re-rounded BLAS results.

Needs Apple Silicon with mlx==0.32.4 and mlx-vlm==0.7.10. Seed selection depends on the
machine's BLAS rounding, so regenerating on another machine may select a different seed.
Run from the repository root:
python3 -B tools/deepseek_v4_parity/make_tiny_fixture.py
"""

import json
import math
import os
import platform
import subprocess
import tempfile

import mlx.core as mx

# Before mlx_vlm is imported: `generate_step` runs on a stream it creates on
# the default device at import time.
mx.set_default_device(mx.cpu)

import mlx.nn as nn
import mlx_vlm
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_vlm.generate.ar import generate_step  # noqa: F401
from mlx_vlm.models import cache as reference_cache
from mlx_vlm.models.deepseek_v4 import hyper_connection as reference_hc
from mlx_vlm.models.deepseek_v4 import language as reference
from mlx_vlm.models.deepseek_v4.config import ModelConfig
from mlx_vlm.models.deepseek_v4.deepseek_v4 import Model

HERE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "Tests", "MLXLMTests", "Resources", "deepseek_v4_tiny",
)
MXFP4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

CONFIG = {
    "model_type": "deepseek_v4",
    "vocab_size": 128,
    "hidden_size": 64,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "n_shared_experts": 1,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "routed_scaling_factor": 1.5,
    "q_lora_rank": 32,
    "qk_rope_head_dim": 16,
    "head_dim": 64,
    "norm_topk_prob": True,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000,
    "compress_rope_theta": 160000,
    "rope_scaling": {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    },
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "compress_ratios": [0, 4, 128, 4, 0],
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "num_hash_layers": 1,
    "swiglu_limit": 10.0,
    "sliding_window": 8,
    "o_groups": 2,
    "o_lora_rank": 32,
    # Sixteen heads: with two, a quarter of the indexer scores are relu-clamped to exactly zero
    # and top-k boundaries fall on ties.
    "index_n_heads": 16,
    "index_head_dim": 32,
    "index_topk": 4,
    "index_block": 4,
    "index_keep": 2,
    "tie_word_embeddings": False,
    "bos_token_id": 0,
    "eos_token_id": 1,
}

# (index_block, index_keep): HISA as configured, then the flat indexer.
INDEXER_MODES = {"hisa": (4, 2), "flat": (0, 0)}
# 250 tokens: two ratio-128 windows, the second completing at decode step 6.
CASES = [(250, 64), (250, 2048), (45, 16)]
DECODE = 24
FIRST_SEED = 11
MAX_SEEDS = 200
PERTURBATIONS = [("canonical", 0, 0), ("noise", 2, 1), ("noise", 2, 2), ("noise", 2, 3)]
MARGIN_FACTOR = 20.0
MARGIN_FLOOR = 1e-5
TOLERANCE_FACTOR = 10.0


def layer_quant(path):
    if ".ffn.switch_mlp." in path:
        return MXFP4
    if ".ffn.shared_experts." in path or ".attn.w" in path or ".attn.indexer.wq" in path:
        return MXFP8
    return None


def to_checkpoint(params, n_experts, o_groups):
    """Reference module paths -> the artifact's checkpoint keys."""
    out = {}
    top = {
        "model.embed_tokens.weight": "embed.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "head.weight",
        "model.hc_head.fn": "hc_head_fn",
        "model.hc_head.base": "hc_head_base",
        "model.hc_head.scale": "hc_head_scale",
    }
    for k, v in params.items():
        if k in top:
            out[top[k]] = v
            continue
        k = k.removeprefix("model.")
        for sub in ("attn", "ffn"):
            for p in ("fn", "base", "scale"):
                k = k.replace(f".{sub}_hc.{p}", f".hc_{sub}_{p}")
        k = k.replace(".ffn.gate.e_score_correction_bias", ".ffn.gate.bias")
        for new, old in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")):
            k = k.replace(f".shared_experts.{new}.", f".shared_experts.{old}.")
        if ".ffn.switch_mlp." in k:
            prefix, rest = k.split(".ffn.switch_mlp.")
            proj, suffix = rest.split(".")
            name = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[proj]
            for e in range(n_experts):
                out[f"{prefix}.ffn.experts.{e}.{name}.{suffix}"] = v[e]
            continue
        if ".attn.wo_a." in k and v.ndim == 3:
            v = v.reshape(o_groups * v.shape[1], -1)
        out[k] = v
    return out


def build(indexer_mode, weights=None, seed=FIRST_SEED):
    cfg = dict(CONFIG)
    cfg["index_block"], cfg["index_keep"] = INDEXER_MODES[indexer_mode]
    model = Model(ModelConfig.from_dict(cfg))
    lm = model.language_model
    quant = {}
    for path, module in tree_flatten(lm.leaf_modules(), is_leaf=nn.Module.is_module):
        q = layer_quant(path)
        if q is not None and hasattr(module, "to_quantized"):
            quant[path] = q
    if weights is None:
        mx.random.seed(seed)
        flat = tree_flatten(lm.parameters())
        fresh = []
        for k, v in flat:
            if k.endswith("tid2eid"):
                fresh.append((k, mx.random.randint(0, cfg["n_routed_experts"], v.shape).astype(mx.int32)))
                continue
            noise = mx.random.normal(v.shape)
            if k.endswith("norm.weight"):
                value = 1 + 0.1 * noise
            elif k.endswith("e_score_correction_bias"):
                value = 0.05 * noise
            elif k.endswith(".scale"):
                value = 0.5 + 0.1 * noise
            else:
                value = 0.15 * noise
            keep_f32 = not lm.cast_predicate(k) or k.endswith(".ape")
            fresh.append((k, value if keep_f32 else value.astype(mx.bfloat16)))
        lm.load_weights(fresh)
        nn.quantize(
            lm,
            class_predicate=lambda p, m: quant.get(p, False),
        )
        mx.eval(lm.parameters())
        return model, quant
    nn.quantize(lm, class_predicate=lambda p, m: quant.get(p, False))
    lm.load_weights(list(lm.sanitize(weights).items()))
    mx.eval(lm.parameters())
    return model, quant


def as_float32(model):
    """Every floating parameter in float32 (quantized weights and their scales stay), eval mode."""
    lm = model.language_model
    lm.update(tree_map(
        lambda p: p.astype(mx.float32) if p.dtype in (mx.bfloat16, mx.float16) else p,
        lm.parameters()))
    mx.eval(lm.parameters())
    model.eval()
    for index, layer in enumerate(lm.model.layers):
        layer._fixture_layer = index
    return model


# ------------------------------------------------------------------ recording and perturbation
# Every discrete selection is recorded with its margin. Perturbation swaps the result of every
# Python-level BLAS product (and of the attention, run as matmuls) for another valid rounding.

STATE = {"record": False, "layer": None, "events": None, "indexer_calls": None,
         "perturb": "none", "ulps": 0, "rng": None}
_argpartition = mx.argpartition
_matmul_op = mx.array.__matmul__
_matmul = mx.matmul
_addmm = mx.addmm
_sdpa = reference.scaled_dot_product_attention
_expert_select = reference._expert_select
_indexer_call = reference.Indexer.__call__
_block_call = reference.DeepseekV4Block.__call__
_hc_expand_op = reference_hc._hc_expand_op


def _record(kind, scores, k, selected):
    """`scores` (..., N) float64, the k largest are selected; masked entries are <= -1e29.
    Margin: k-th largest minus (k+1)-th of the unmasked entries, inf if nothing is rejected."""
    rows = scores.reshape(-1, scores.shape[-1])
    margins = np.empty(rows.shape[0])
    for i, row in enumerate(rows):
        valid = np.sort(row[row > -1e29])[::-1]
        margins[i] = math.inf if valid.size <= k else valid[k - 1] - valid[k]
    STATE["events"].append({
        "kind": kind, "layer": STATE["layer"], "margins": margins, "scores": rows,
        "selected": np.sort(np.asarray(selected).reshape(rows.shape[0], -1), axis=-1),
    })


def _hook_expert_select(logits, bias, top_k, scaling, normalize, func):
    inds, weights = _expert_select(logits, bias, top_k, scaling, normalize, func)
    if STATE["record"]:
        scores = reference._score_func(logits.astype(mx.float32), func) + bias
        mx.eval(scores, inds)
        _record("moe", np.array(scores).astype(np.float64), top_k, np.array(inds))
    return inds, weights


def _hook_argpartition(a, kth, axis=-1, **kw):
    out = _argpartition(a, kth, axis=axis, **kw)
    if STATE["record"] and STATE["indexer_calls"] is not None:
        mx.eval(a, out)
        scores = -np.array(a.astype(mx.float32)).astype(np.float64)
        STATE["indexer_calls"].append((scores, kth + 1, np.array(out)[..., : kth + 1]))
    return out


def _hook_indexer(self, x, q_residual, position_rope, pool_cache, offset):
    STATE["indexer_calls"] = []
    try:
        out = _indexer_call(self, x, q_residual, position_rope, pool_cache, offset)
    finally:
        calls, STATE["indexer_calls"] = STATE["indexer_calls"], None
    if STATE["record"] and out is not None:
        mx.eval(out)
        for name, (scores, k, chosen) in zip({1: ["flat"], 2: ["coarse", "fine"]}[len(calls)], calls):
            _record(name, scores, k, chosen)
            if name != "coarse":
                STATE["events"][-1]["positions"] = np.sort(np.array(out).reshape(-1, k), axis=-1)
    return out


def _hook_block(self, h, *args, **kwargs):
    STATE["layer"] = self._fixture_layer
    return _block_call(self, h, *args, **kwargs)


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _perturbed(out, exact):
    """`exact` (float64) rounded to float32, moved by a random -ulps..ulps, cast to out's dtype."""
    if STATE["perturb"] == "identity":
        return out
    r = exact.astype(np.float32)
    if STATE["ulps"]:
        steps = STATE["rng"].integers(-STATE["ulps"], STATE["ulps"] + 1, size=r.shape)
        moved = (r.view(np.int32) + np.where(r > 0, steps, -steps).astype(np.int32)).view(np.float32)
        r = np.where((r != 0) & np.isfinite(r), moved, r)
    return mx.array(r).astype(out.dtype)


def _perturbing(out):
    if STATE["perturb"] == "none":
        return False
    try:
        mx.eval(out)
    except Exception:
        return False  # inside a compile trace
    return out.dtype in (mx.float32, mx.bfloat16, mx.float16)


def _hook_matmul_op(a, b):
    out = _matmul_op(a, b)
    if isinstance(b, mx.array) and _perturbing(out):
        return _perturbed(out, np.matmul(_f64(a), _f64(b)))
    return out


def _hook_matmul(a, b, **kw):
    out = _matmul(a, b, **kw)
    return _perturbed(out, np.matmul(_f64(a), _f64(b))) if _perturbing(out) else out


def _hook_addmm(c, a, b, alpha=1.0, beta=1.0, **kw):
    out = _addmm(c, a, b, alpha=alpha, beta=beta, **kw)
    if _perturbing(out):
        return _perturbed(out, np.matmul(_f64(a), _f64(b)) * alpha + beta * _f64(c))
    return out


def _hook_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """While perturbing: the CPU attention as matmuls in the input dtype (bitwise equal to
    mx.fast's CPU path unperturbed): scores, -inf mask, sink column, precise softmax."""
    if STATE["perturb"] == "none":
        return _sdpa(queries, keys, values, cache, scale, mask, sinks)
    s = (queries * scale) @ keys.swapaxes(-1, -2)
    if mask is not None:
        assert mask.dtype == mx.bool_
        s = mx.where(mask, s, mx.array(-math.inf, dtype=s.dtype))
    sk = mx.broadcast_to(sinks.astype(s.dtype)[None, :, None, None], s.shape[:-1] + (1,))
    p = mx.softmax(mx.concatenate([sk, s], axis=-1), axis=-1, precise=True)[..., 1:]
    return p @ values


def _hook_hc_expand(x, residual, post, comb):
    """While perturbing: the compiled residual update unrolled, so its matmul is perturbed."""
    if STATE["perturb"] == "none":
        return _hc_expand_op(x, residual, post, comb)
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(x.dtype)


def install_hooks():
    mx.argpartition = _hook_argpartition
    mx.array.__matmul__ = _hook_matmul_op
    mx.matmul = _hook_matmul
    mx.addmm = _hook_addmm
    reference.scaled_dot_product_attention = _hook_sdpa
    reference._expert_select = _hook_expert_select
    reference.Indexer.__call__ = _hook_indexer
    reference.DeepseekV4Block.__call__ = _hook_block
    reference_hc._hc_expand_op = _hook_hc_expand


def run_case(model, prompt, step, forced=None, perturb=("none", 0, 0)):
    """mlx-vlm generate_step's plan, greedy (or fed `forced`). Returns per decode step the
    float32 log-probs and argmax, and every recorded selection tagged with its phase."""
    STATE["perturb"], STATE["ulps"] = perturb[0], perturb[1]
    STATE["rng"] = np.random.default_rng(perturb[2])
    lm = model.language_model
    cache = reference_cache.make_prompt_cache(lm)
    STATE["events"], STATE["record"] = [], True
    phases = []
    rest = list(prompt)
    while len(rest) > 1:
        n = min(step, len(rest) - 1)
        lm(mx.array([rest[:n]]), cache=cache)
        mx.eval([c.state for c in cache])
        phases += ["prefill"] * (len(STATE["events"]) - len(phases))
        rest = rest[n:]
    token = rest[0]
    logprobs, argmax = [], []
    for t in range(DECODE):
        logits = lm(mx.array([[token]]), cache=cache).logits[0, -1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(lp)
        logprobs.append(np.array(lp))
        argmax.append(int(np.argmax(logprobs[-1])))
        phases += [t] * (len(STATE["events"]) - len(phases))
        token = argmax[-1] if forced is None else forced[t]
    STATE["record"], STATE["perturb"], STATE["ulps"] = False, "none", 0
    events = STATE["events"]
    for event, phase in zip(events, phases):
        event["phase"] = phase
    top2 = [np.sort(lp)[-2:] for lp in logprobs]
    events += [{"kind": "argmax", "phase": t, "layer": None,
                "margins": np.array([b - a]), "scores": lp[None].astype(np.float64),
                "selected": np.array([[argmax[t]]])}
               for t, ((a, b), lp) in enumerate(zip(top2, logprobs))]
    return {"logprobs": logprobs, "argmax": argmax, "events": events}


def decisions(events, step):
    """The decode step's routed experts (score-routed layers) and indexer selections."""
    experts = [e["selected"][0].tolist() for e in events
               if e["phase"] == step and e["kind"] == "moe"]
    indexer = [e["positions"][0].tolist() for e in events
               if e["phase"] == step and e["kind"] in ("flat", "fine")]
    return experts, indexer


def evaluate(loaded, seed):
    """Run every case; reject on a margin under MARGIN_FLOOR. Then rerun each case perturbed
    and reject on any changed selection or a margin under MARGIN_FACTOR times the largest
    score change of its kind. Returns (cases, report) or (None, reason)."""
    # One prompt per length, shared by both indexer modes and both prefill chunkings.
    rng = np.random.default_rng(seed)
    prompts = {n: rng.integers(2, CONFIG["vocab_size"], size=n).tolist() for n, _ in CASES}
    runs, smallest = [], {}
    for mode in INDEXER_MODES:
        model = as_float32(build(mode, weights=dict(loaded))[0])
        for length, step in CASES:
            base = run_case(model, prompts[length], step)
            for event in base["events"]:
                finite = event["margins"][np.isfinite(event["margins"])]
                if finite.size:
                    smallest[event["kind"]] = min(smallest.get(event["kind"], math.inf), finite.min())
            if min(smallest.values()) < MARGIN_FLOOR:
                kind = min(smallest, key=smallest.get)
                return None, f"{kind} margin {smallest[kind]:.3g} < floor {MARGIN_FLOOR:.3g}"
            runs.append((mode, model, length, step, base))

    sensitivity, worst_logprob = {}, 0.0
    for mode, model, length, step, base in runs:
        prompt = prompts[length]
        same = run_case(model, prompt, step, forced=base["argmax"], perturb=("identity", 0, 0))
        assert all(np.array_equal(x, y) for x, y in zip(base["logprobs"], same["logprobs"])), \
            "the unrolled attention or hyper-connection expand differs from the reference"
        for perturb in PERTURBATIONS:
            other = run_case(model, prompt, step, forced=base["argmax"], perturb=perturb)
            for a, b in zip(base["events"], other["events"], strict=True):
                assert (a["kind"], a["phase"], a["layer"]) == (b["kind"], b["phase"], b["layer"])
                if not np.array_equal(a["selected"], b["selected"]):
                    return None, f"{mode} {length}/{step} {perturb}: {a['kind']} selection changed"
                both = (a["scores"] > -1e29) & (b["scores"] > -1e29)
                change = np.abs(a["scores"] - b["scores"])[both]
                if change.size:
                    sensitivity[a["kind"]] = max(sensitivity.get(a["kind"], 0.0), change.max())
            worst_logprob = max(worst_logprob, max(
                float(np.abs(x - y).max()) for x, y in zip(base["logprobs"], other["logprobs"])))
    thresholds = {k: max(MARGIN_FLOOR, MARGIN_FACTOR * v) for k, v in sensitivity.items()}
    for kind, margin in smallest.items():
        if margin < thresholds[kind]:
            return None, f"{kind} margin {margin:.3g} < threshold {thresholds[kind]:.3g}"

    cases = []
    for mode, model, length, step, base in runs:
        steps = [decisions(base["events"], t) for t in range(DECODE)]
        cases.append({
            "indexer": mode,
            "index_block": INDEXER_MODES[mode][0],
            "index_keep": INDEXER_MODES[mode][1],
            "prefill_step": step,
            "prompt": prompts[length],
            "greedy": base["argmax"],
            "logprobs": [[float(f"{v:.9g}") for v in lp] for lp in base["logprobs"]],
            "experts": [s[0] for s in steps],
            "indexer_selection": [s[1] for s in steps],
        })
    return cases, {
        "min_margin": {k: float(v) for k, v in sorted(smallest.items())},
        "margin_threshold": {k: float(v) for k, v in sorted(thresholds.items())},
        "max_score_change": {k: float(v) for k, v in sorted(sensitivity.items())},
        "max_logprob_change": worst_logprob,
    }


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def main():
    install_hooks()
    for seed in range(FIRST_SEED, FIRST_SEED + MAX_SEEDS):
        model, quant = build("hisa", seed=seed)
        params = dict(tree_flatten(model.language_model.parameters()))
        checkpoint = to_checkpoint(params, CONFIG["n_routed_experts"], CONFIG["o_groups"])
        with tempfile.TemporaryDirectory() as tmp:
            # The oracle comes from the written checkpoint, through the reference loader.
            mx.save_safetensors(os.path.join(tmp, "model.safetensors"), checkpoint)
            loaded = dict(mx.load(os.path.join(tmp, "model.safetensors")))
            cases, report = evaluate(loaded, seed)
        print(f"seed {seed}: {'accepted' if cases else report}", flush=True)
        if cases:
            break
    else:
        raise SystemExit(f"no seed in {MAX_SEEDS} met the margin requirements")

    mx.save_safetensors(os.path.join(HERE, "model.safetensors"), checkpoint)
    quant_cfg = dict(MXFP4)
    for path, q in quant.items():
        layer_path = path.removeprefix("model.")
        for new, old in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")):
            layer_path = layer_path.replace(f".shared_experts.{new}", f".shared_experts.{old}")
        if ".switch_mlp." not in layer_path:
            quant_cfg[layer_path] = q
    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump(dict(CONFIG, quantization=quant_cfg), f, indent=1, sort_keys=True)

    tolerance = float(f"{TOLERANCE_FACTOR * report['max_logprob_change']:.2g}")
    provenance = {
        "generator": "tools/deepseek_v4_parity/make_tiny_fixture.py",
        "mlx": mx.__version__,
        "mlx_vlm": mlx_vlm.__version__,
        "device": "cpu",
        "chip": chip(),
        "mode": "eval, float32 parameters and activations, op path (no fused kernels on cpu)",
        "weight_seed": seed,
        "prompt_seed": seed,
        "perturbation": "every Python-level BLAS product (matmul, @, addmm, the attention as "
                        "matmuls, the hyper-connection expand) replaced by the float64 product "
                        "rounded to float32 (canonical) and moved by random -2..2 ulps "
                        "(noise, seeds 1-3); teacher-forced on the greedy tokens",
        "margin_factor": MARGIN_FACTOR,
        "margin_floor": MARGIN_FLOOR,
        "tolerance_factor": TOLERANCE_FACTOR,
        **report,
    }
    with open(os.path.join(HERE, "oracle.json"), "w") as f:
        json.dump({"decode": DECODE, "compute_dtype": "float32", "tolerance": tolerance,
                   "provenance": provenance, "cases": cases}, f, separators=(",", ":"))
    print(json.dumps(dict(provenance, tolerance=tolerance), indent=1))


if __name__ == "__main__":
    main()
