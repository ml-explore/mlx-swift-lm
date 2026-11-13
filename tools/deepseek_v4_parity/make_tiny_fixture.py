"""Writes Tests/MLXLMTests/Resources/deepseek_v4_tiny (config, weights, oracle.json) from
DeepSeek's own DeepSeek-V4 reference implementation (`inference/model.py`).

The fixture is a 4-layer DeepSeek-V4 with seeded random weights (compress ratios 0, 4, 128, 4,
one hash-routed layer, an 8-token window), quantized and laid out like the DeepSeek-V4-Flash
MXFP4 MLX conversion. The oracle is DeepSeek's reference run on the dequantized weights in
float32 on the CPU, teacher-forced along its own greedy tokens; a weight seed is kept only if
every discrete selection (experts, indexer top-k, greedy argmax) has a margin at least
MARGIN_FACTOR times the largest score change measured when every product is re-rounded.

Usage, from the repository root, with a python that has torch and numpy (no MLX needed):

    python -B tools/deepseek_v4_parity/make_tiny_fixture.py --reference-dir <inference folder>

`<inference folder>` is DeepSeek-V4's `inference/` directory (model.py, kernel.py, ...). It is
imported, not copied; the sha256 of its model.py is recorded in oracle.json. Optionally, with a
python that has mlx, `python -B tools/deepseek_v4_parity/check_mxfp.py
Tests/MLXLMTests/Resources/deepseek_v4_tiny` confirms that the NumPy quantizer used here
(mxfp.py) matches `mx.quantize` bit for bit.

What runs, exactly:
- model.py is DeepSeek's file, unmodified, imported with `reference_shim/` first on sys.path, so
  its `from kernel import ...` resolves to reference_shim/kernel.py, a float32 CPU
  re-implementation of kernel.py's sparse_attn and hc_split_sinkhorn, and its
  `fast_hadamard_transform` import to reference_shim/fast_hadamard_transform.py.
- Activation quantization is identity: the in-place FP8/FP4 quantize-dequantize the reference
  applies to the window and compressed KV (non-rope dims), the indexer keys and the indexer
  queries is skipped, and so is the Hadamard rotation of the indexer queries and keys (which
  cancels in their dot products without the quantization). The FP8 quantization of Linear
  inputs never arises: the model is built with `dtype="bf16"`, `expert_dtype=None`, so every
  Linear is a plain float matmul, on the dequantized values of the fixture's MXFP4/MXFP8 weights.
- float32 on the CPU throughout (the reference runs bfloat16 on CUDA).
- Monkeypatches, all from this script:
  1. `model.rotate_activation`: the same `hadamard_transform(x, scale=x.size(-1) ** -0.5)` call
     without its bfloat16 assert.
  2. The model is built with torch's default dtype float32 (generate.py sets bfloat16) and cast
     with `.float()` before the weights load (with `dtype="bf16"`, the reference creates its
     Linear weights in bfloat16).
  3. HISA cases only (`index_block > 0`): `Indexer.forward` is replaced by a copy whose top-k
     step is mlx-vlm's two-stage block selection (pooled entries in blocks of `index_block`,
     blocks ranked by the score of their mean entry, the best `index_keep` blocks scored in
     full; only whole blocks are candidates). DeepSeek's reference has no HISA; the query, key,
     weight and score computation and the index post-processing are the reference's lines.
  4. Observation only: `Block.forward`, `Gate.forward`, `Indexer.forward` and `torch.Tensor.topk`
     are wrapped to record each selection's scores and margin; values are untouched.
  5. Margin measurement only (not the oracle run): `torch.nn.functional.linear`, `torch.einsum`,
     `Block.hc_post` and the shim's sparse_attn products are replaced by float64 products
     rounded to float32, canonically and moved by random -2..2 ulps.
- Prefill is the reference's: one call at start_pos 0 over the prompt but its last token, then
  one-token decode steps. The chunked-prefill cases share that one-shot run: the cache must not
  depend on how the prompt is split.

Weights: numpy normal draws per reference parameter, scaled as before (norm weights 1 + 0.1 n,
router bias 0.05 n, hyper-connection scales 0.5 + 0.1 n, the rest 0.15 n), rounded to bfloat16
except the float32 parameters (hyper-connections, attention sinks, compressor `ape`, router
bias), then MXFP4 (routed experts) or MXFP8 (attention projections, indexer `wq_b`, shared
expert) with group 32. Every hash-routing (`tid2eid`) row lists distinct experts. The draws
come from numpy, not MLX, so a seed does not reproduce an earlier MLX-generated fixture.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import subprocess
import sys

import numpy as np
import torch

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import mxfp  # noqa: E402

HERE = os.path.join(TOOLS, "..", "..", "Tests", "MLXLMTests", "Resources", "deepseek_v4_tiny")
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

# (index_block, index_keep): HISA as configured, then the flat (exact) indexer.
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


def quant_of(name):
    """Checkpoint quantization of a reference parameter (module path + ".weight")."""
    if not name.endswith(".weight"):
        return None
    if ".ffn.experts." in name:
        return MXFP4
    if ".ffn.shared_experts." in name or ".attn.w" in name or ".attn.indexer.wq" in name:
        return MXFP8
    return None


def keeps_float32(name):
    return ".hc_" in name or name.startswith("hc_") or name.endswith(
        ("attn_sink", ".ape", ".gate.bias"))


# ------------------------------------------------------------------ the reference
REC = {"on": False, "layer": None, "ctx": None, "events": None, "phase": None, "hisa": None}
PERT = {"fn": None}
ref = kernel = None


def _record(kind, scores, k, selected, positions=None):
    """`scores` (..., N), the k largest selected; masked entries are <= -1e29. Margin: k-th
    largest minus (k+1)-th of the unmasked entries, inf if nothing is rejected."""
    rows = np.asarray(scores, dtype=np.float64).reshape(-1, scores.shape[-1])
    margins = np.empty(rows.shape[0])
    for i, row in enumerate(rows):
        valid = np.sort(row[row > -1e29])[::-1]
        margins[i] = math.inf if valid.size <= k else valid[k - 1] - valid[k]
    event = {"kind": kind, "layer": REC["layer"], "phase": REC["phase"], "margins": margins,
             "scores": rows,
             "selected": np.sort(np.asarray(selected).reshape(rows.shape[0], -1), axis=-1)}
    if positions is not None:
        event["positions"] = np.sort(np.asarray(positions).reshape(rows.shape[0], -1), axis=-1)
    REC["events"].append(event)


def load_reference(reference_dir):
    """Import the unmodified model.py on the CPU shim and install the patches listed above."""
    global ref, kernel
    shim = os.path.join(TOOLS, "reference_shim")
    sys.path.insert(0, os.path.abspath(reference_dir))
    sys.path.insert(0, shim)
    torch.set_default_dtype(torch.float32)
    import fast_hadamard_transform
    import kernel as kernel_module
    import model as model_module
    assert os.path.samefile(os.path.dirname(model_module.__file__), reference_dir)
    assert os.path.samefile(os.path.dirname(kernel_module.__file__), shim)
    ref, kernel = model_module, kernel_module

    # 1
    ref.rotate_activation = lambda x: fast_hadamard_transform.hadamard_transform(
        x, scale=x.size(-1) ** -0.5)

    # 4: observation hooks
    topk = torch.Tensor.topk

    def topk_hook(self, k, dim=-1, largest=True, sorted=True):
        out = topk(self, k, dim=dim, largest=largest, sorted=sorted)
        if REC["on"] and REC["ctx"] in ("moe", "flat"):
            chosen = out[1].clone().numpy()
            _record(REC["ctx"], self.clone().double().numpy(), k, chosen,
                    positions=chosen if REC["ctx"] == "flat" else None)
        return out

    torch.Tensor.topk = topk_hook
    block_forward, gate_forward = ref.Block.forward, ref.Gate.forward
    indexer_forward = ref.Indexer.forward

    def block_hook(self, x, start_pos, input_ids):
        REC["layer"] = self.layer_id
        return block_forward(self, x, start_pos, input_ids)

    def gate_hook(self, x, input_ids=None):
        REC["ctx"] = "moe"
        try:
            return gate_forward(self, x, input_ids)
        finally:
            REC["ctx"] = None

    def indexer_hook(self, x, qr, start_pos, offset):
        if REC["hisa"] is not None:
            return hisa_indexer_forward(self, x, qr, start_pos, offset, *REC["hisa"], topk)
        REC["ctx"] = "flat"
        try:
            return indexer_forward(self, x, qr, start_pos, offset)
        finally:
            REC["ctx"] = None

    ref.Block.forward, ref.Gate.forward, ref.Indexer.forward = block_hook, gate_hook, indexer_hook

    # 5: perturbation hooks, inactive unless PERT["fn"] is set
    linear, einsum, hc_post = torch.nn.functional.linear, torch.einsum, ref.Block.hc_post

    def linear_hook(x, w, bias=None):
        if PERT["fn"] is None:
            return linear(x, w, bias)
        assert bias is None
        return PERT["fn"](linear(x.double(), w.double())).to(x.dtype)

    def einsum_hook(equation, *operands):
        if PERT["fn"] is None or not all(o.is_floating_point() for o in operands):
            return einsum(equation, *operands)
        return PERT["fn"](einsum(equation, *[o.double() for o in operands])).to(operands[0].dtype)

    def hc_post_hook(self, x, residual, post, comb):
        if PERT["fn"] is None:
            return hc_post(self, x, residual, post, comb)
        y = post.double().unsqueeze(-1) * x.double().unsqueeze(-2) + torch.sum(
            comb.double().unsqueeze(-1) * residual.double().unsqueeze(-2), dim=2)
        return PERT["fn"](y).type_as(x)

    torch.nn.functional.linear, torch.einsum, ref.Block.hc_post = linear_hook, einsum_hook, hc_post_hook


def hisa_indexer_forward(self, x, qr, start_pos, offset, block, keep, topk):
    """Patch 3: model.py's Indexer.forward, its top-k replaced by mlx-vlm's HISA selection."""
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
    ratio = self.compress_ratio
    rd = self.rope_head_dim
    end_pos = start_pos + seqlen
    if self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache
        self.compressor.freqs_cis = self.freqs_cis
    q = self.wq_b(qr)
    q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
    ref.apply_rotary_emb(q[..., -rd:], freqs_cis)
    q = ref.rotate_activation(q)
    ref.fp4_act_quant(q, ref.fp4_block_size, True)
    self.compressor(x, start_pos)
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
    kv = self.kv_cache[:bsz, :end_pos // ratio]
    pool = end_pos // ratio
    k = min(self.index_topk, pool)
    if not (block > 0 and pool >= block * keep and keep * block >= k):
        REC["ctx"] = "flat"
        try:
            index_score = torch.einsum("bshd,btd->bsht", q, kv)
            index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
            if start_pos == 0:
                mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
                index_score += torch.where(mask, float("-inf"), 0)
            topk_idxs = index_score.topk(k, dim=-1)[1]
        finally:
            REC["ctx"] = None
    else:
        # Pooled entries visible to each query: (t + 1) // ratio at prefill, all at decode.
        if start_pos == 0:
            visible = (torch.arange(1, seqlen + 1) // ratio)[None, :, None]
        else:
            visible = torch.full((1, seqlen, 1), pool)
        blocks = pool // block
        mean = kv[:, :blocks * block].unflatten(1, (blocks, block)).mean(2)
        coarse = (torch.einsum("bshd,btd->bsht", q, mean).relu() * weights.unsqueeze(-1)).sum(dim=2)
        coarse = torch.where((torch.arange(blocks) * block)[None, None] < visible, coarse,
                             torch.tensor(-1e30))
        kept = min(keep, blocks)
        top_blocks = topk(coarse, kept, dim=-1)[1]
        if REC["on"]:
            _record("coarse", coarse.double().numpy(), kept, top_blocks.numpy())
        positions = (top_blocks[..., None] * block + torch.arange(block)).flatten(-2)
        candidates = torch.gather(kv[:, None].expand(bsz, seqlen, pool, kv.size(-1)), 2,
                                  positions[..., None].expand(*positions.shape, kv.size(-1)))
        fine = (torch.einsum("bshd,bscd->bshc", q, candidates).relu() * weights.unsqueeze(-1)).sum(dim=2)
        fine = torch.where(positions < visible, fine, torch.tensor(-1e30))
        chosen = topk(fine, k, dim=-1)[1]
        topk_idxs = torch.gather(positions, -1, chosen)
        if REC["on"]:
            _record("fine", fine.double().numpy(), k, chosen.numpy(), positions=topk_idxs.numpy())
    if start_pos == 0:
        mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        topk_idxs = torch.where(mask, -1, topk_idxs + offset)
    else:
        topk_idxs = topk_idxs + offset
    return topk_idxs


def model_args():
    c, rs = CONFIG, CONFIG["rope_scaling"]
    assert c["num_key_value_heads"] == 1 and c["norm_topk_prob"] and c["topk_method"] == "noaux_tc"
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


def generate(seed):
    """Seeded weights: the checkpoint (name -> array, its bfloat16 names, its quantization) and
    the reference state dict with the values the checkpoint decodes to."""
    rng = np.random.default_rng([0, seed])
    shapes = [(k, tuple(v.shape)) for k, v in ref.Transformer(model_args()).named_parameters()]
    checkpoint, bf16, quant, state = {}, set(), {}, {}
    for name, shape in shapes:
        if name.endswith("tid2eid"):
            n, k = shape
            table = np.stack([rng.choice(CONFIG["n_routed_experts"], k, replace=False) for _ in range(n)])
            checkpoint[name] = table.astype(np.int32)
            state[name] = torch.from_numpy(checkpoint[name])
            continue
        noise = rng.standard_normal(shape).astype(np.float32)
        if name.endswith("norm.weight"):
            value = 1 + np.float32(0.1) * noise
        elif name.endswith(".gate.bias"):
            value = np.float32(0.05) * noise
        elif name.endswith("_scale"):
            value = np.float32(0.5) + np.float32(0.1) * noise
        else:
            value = np.float32(0.15) * noise
        if keeps_float32(name):
            checkpoint[name] = value
            state[name] = torch.from_numpy(value)
            continue
        value = mxfp.bf16_round(value)
        q = quant_of(name)
        if q is None:
            checkpoint[name] = value
            bf16.add(name)
            state[name] = torch.from_numpy(value)
            continue
        packed, scales = mxfp.quantize(value, q["bits"])
        base = name[: -len(".weight")]
        checkpoint[name], checkpoint[base + ".scales"] = packed, scales
        quant[base] = q
        state[name] = torch.from_numpy(mxfp.dequantize(packed, scales, q["bits"]))
    return checkpoint, bf16, quant, state


def build(state):
    model = ref.Transformer(model_args()).float()  # patch 2
    result = model.load_state_dict(state, strict=False)
    assert not result.missing_keys and not result.unexpected_keys, result
    return model.eval()


# ------------------------------------------------------------------ runs and margins
def _perturb_fn(kind, ulps, seed):
    rng = np.random.default_rng(seed)

    def fn(exact):
        r = exact.to(torch.float32)
        if ulps:
            steps = torch.from_numpy(rng.integers(-ulps, ulps + 1, size=tuple(r.shape)).astype(np.int32))
            moved = (r.view(torch.int32) + torch.where(r > 0, steps, -steps)).view(torch.float32)
            r = torch.where((r != 0) & torch.isfinite(r), moved, r)
        return r
    return fn


@contextlib.contextmanager
def perturbing(perturb):
    if perturb is not None:
        PERT["fn"] = kernel.PERTURB = _perturb_fn(*perturb)
    try:
        yield
    finally:
        PERT["fn"] = kernel.PERTURB = None


def run_case(state, prompt, hisa, forced=None, perturb=None):
    """One-shot prefill of prompt[:-1], then DECODE steps fed the last prompt token and then the
    greedy (or `forced`) tokens. Returns per step the float32 log-probs and argmax, and every
    recorded selection tagged with its phase."""
    model = build(state)
    REC.update(on=True, events=[], hisa=hisa, phase="prefill", layer=None)
    logprobs, argmax = [], []
    with perturbing(perturb):
        model(torch.tensor([prompt[:-1]]), 0)
        token = prompt[-1]
        for t in range(DECODE):
            REC["phase"] = t
            logits = model(torch.tensor([[token]]), len(prompt) - 1 + t)[0].to(torch.float32)
            logprobs.append((logits - torch.logsumexp(logits, -1, keepdim=True)).numpy().copy())
            argmax.append(int(np.argmax(logprobs[-1])))
            token = argmax[-1] if forced is None else forced[t]
    REC.update(on=False, hisa=None)
    events = REC["events"]
    for t, lp in enumerate(logprobs):
        a, b = np.sort(lp)[-2:]
        events.append({"kind": "argmax", "phase": t, "layer": None, "margins": np.array([b - a]),
                       "scores": lp[None].astype(np.float64), "selected": np.array([[argmax[t]]])})
    return {"logprobs": logprobs, "argmax": argmax, "events": events}


def decisions(events, step):
    """The decode step's routed experts (score-routed layers) and indexer selections."""
    experts = [e["selected"][0].tolist() for e in events if e["phase"] == step and e["kind"] == "moe"]
    indexer = [e["positions"][0].tolist() for e in events
               if e["phase"] == step and e["kind"] in ("flat", "fine")]
    return experts, indexer


def evaluate(state, seed):
    """Run every case; reject on a margin under MARGIN_FLOOR. Then rerun each case perturbed and
    reject on any changed selection or a margin under MARGIN_FACTOR times the largest score
    change of its kind. Returns (cases, report, score); cases is None on rejection and report
    the reason; score = min over kinds of margin / threshold (for the closest-seed report)."""
    rng = np.random.default_rng(seed)
    prompts = {n: rng.integers(2, CONFIG["vocab_size"], size=n).tolist() for n, _ in CASES}
    runs, smallest = {}, {}
    for mode, (block, keep) in INDEXER_MODES.items():
        for length in prompts:
            hisa = (block, keep) if block > 0 else None
            base = run_case(state, prompts[length], hisa)
            for event in base["events"]:
                finite = event["margins"][np.isfinite(event["margins"])]
                if finite.size:
                    smallest[event["kind"]] = min(smallest.get(event["kind"], math.inf), float(finite.min()))
            runs[(mode, length)] = (hisa, base)
    if min(smallest.values()) < MARGIN_FLOOR:
        kind = min(smallest, key=smallest.get)
        return None, f"{kind} margin {smallest[kind]:.3g} < floor {MARGIN_FLOOR:.3g}", 0.0

    sensitivity, worst_logprob = {}, 0.0
    for (mode, length), (hisa, base) in runs.items():
        for perturb in PERTURBATIONS:
            other = run_case(state, prompts[length], hisa, forced=base["argmax"], perturb=perturb)
            for a, b in zip(base["events"], other["events"], strict=True):
                assert (a["kind"], a["phase"], a["layer"]) == (b["kind"], b["phase"], b["layer"])
                if not np.array_equal(a["selected"], b["selected"]):
                    return None, f"{mode} {length} {perturb}: {a['kind']} selection changed", 0.0
                both = (a["scores"] > -1e29) & (b["scores"] > -1e29)
                change = np.abs(a["scores"][both] - b["scores"][both])
                if change.size:
                    sensitivity[a["kind"]] = max(sensitivity.get(a["kind"], 0.0), float(change.max()))
            worst_logprob = max(worst_logprob, max(
                float(np.abs(x - y).max()) for x, y in zip(base["logprobs"], other["logprobs"])))
    thresholds = {k: max(MARGIN_FLOOR, MARGIN_FACTOR * v) for k, v in sensitivity.items()}
    report = {
        "min_margin": dict(sorted(smallest.items())),
        "margin_threshold": dict(sorted(thresholds.items())),
        "max_score_change": dict(sorted(sensitivity.items())),
        "max_logprob_change": worst_logprob,
    }
    score = min(smallest[k] / thresholds[k] for k in smallest)
    for kind, margin in smallest.items():
        if margin < thresholds[kind]:
            return None, dict(report, reason=f"{kind} margin {margin:.3g} < threshold {thresholds[kind]:.3g}"), score

    cases = []
    for mode, (block, keep) in INDEXER_MODES.items():
        for length, step in CASES:
            hisa, base = runs[(mode, length)]
            steps = [decisions(base["events"], t) for t in range(DECODE)]
            cases.append({
                "indexer": mode,
                "index_block": block,
                "index_keep": keep,
                "prefill_step": step,
                "prompt": prompts[length],
                "greedy": base["argmax"],
                "logprobs": [[float(f"{v:.9g}") for v in lp] for lp in base["logprobs"]],
                "experts": [s[0] for s in steps],
                "indexer_selection": [s[1] for s in steps],
            })
    return cases, report, score


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference-dir", required=True,
                        help="DeepSeek-V4's inference/ folder (model.py, kernel.py)")
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED)
    parser.add_argument("--max-seeds", type=int, default=MAX_SEEDS)
    opts = parser.parse_args()
    load_reference(opts.reference_dir)

    best = None
    for seed in range(opts.first_seed, opts.first_seed + opts.max_seeds):
        checkpoint, bf16, quant, state = generate(seed)
        cases, report, score = evaluate(state, seed)
        reason = report if cases is None and isinstance(report, str) else (report or {}).get("reason")
        print(f"seed {seed}: {'accepted' if cases else reason}", flush=True)
        if cases is None and score > 0 and (best is None or score > best[1]):
            best = (seed, score, report)
        if cases:
            break
    else:
        if best:
            print(f"closest seed {best[0]} (min margin / threshold {best[1]:.3g}):\n"
                  + json.dumps(best[2], indent=1))
        raise SystemExit(f"no seed in {opts.max_seeds} met the margin requirements")

    os.makedirs(HERE, exist_ok=True)
    mxfp.save_safetensors(os.path.join(HERE, "model.safetensors"), checkpoint, bf16=bf16)
    quant_cfg = dict(MXFP4)
    for path, q in sorted(quant.items()):
        if q is not MXFP4:
            quant_cfg[path] = q
    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump(dict(CONFIG, quantization=quant_cfg), f, indent=1, sort_keys=True)

    model_py = os.path.join(opts.reference_dir, "model.py")
    tolerance = float(f"{TOLERANCE_FACTOR * report['max_logprob_change']:.2g}")
    provenance = {
        "generator": "tools/deepseek_v4_parity/make_tiny_fixture.py",
        "reference": "DeepSeek-V4 inference/model.py, unmodified, on the CPU kernel shim; "
                     "activation quantization identity; HISA cases patch mlx-vlm's block "
                     "selection into the indexer",
        "reference_model_py_sha256": hashlib.sha256(open(model_py, "rb").read()).hexdigest(),
        "torch": torch.__version__,
        "device": "cpu",
        "chip": chip(),
        "mode": "float32 parameters and activations; one-shot prefill, one-token decode",
        "weight_seed": seed,
        "prompt_seed": seed,
        "perturbation": "every product (Linear, einsum, the sparse attention's two products, the "
                        "hyper-connection residual update) replaced by the float64 product "
                        "rounded to float32 (canonical) and moved by random -2..2 ulps (noise, "
                        "seeds 1-3); teacher-forced on the greedy tokens",
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
