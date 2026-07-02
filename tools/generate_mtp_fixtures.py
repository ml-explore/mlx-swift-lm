#!/usr/bin/env python3
"""
generate_mtp_fixtures.py — Generate Python-reference fixtures for verifying
Swift port of Gemma 4 MTP speculative decoding.

Pinned to mlx-vlm commit: d49d428e9f570dc0387b9598b3b7e0ea391590d2

Capture mechanism: per-instance `__class__` swap to a locally-defined subclass
that overrides `__call__` to capture inputs/outputs into a closure-scoped list.
This is cleaner than module-level monkey-patching of `Cls.__call__` because:
  (a) the imported class is left untouched (no cross-test contamination),
  (b) capture state lives in a closure rather than a global,
  (c) restoration is a single `instance.__class__ = OriginalCls` assignment,
  (d) other instances of the same type are unaffected.

Python looks up special methods (`__call__`, `__len__`, etc.) on `type(instance)`
rather than on `instance` itself, so per-instance overrides must go through the
type. Swapping `instance.__class__` to a subclass satisfies that lookup path
without mutating the original class.

Usage:
    python generate_mtp_fixtures.py                # all suites
    python generate_mtp_fixtures.py --suite masks  # just one suite
    python generate_mtp_fixtures.py --skip-end-to-end  # skip the heaviest suite

Output:
    fixtures/
        masks/
            bidirectional_q1_kv8.safetensors
            bidirectional_swa_q1_kv8_w4.safetensors
            ...
        drafter_forward/
            case_01_q1_greedy.safetensors
            ...
        drafter_block/
            case_01_block2.safetensors
            ...
        end_to_end/
            case_01_greedy.safetensors
            ...
        FIXTURE-SCHEMA.md
        FIXTURE-MANIFEST.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import textwrap
import zlib


def _stable_hash(s: str) -> int:
    """Process-independent 32-bit hash of a string.

    Python's built-in ``hash`` randomizes per-process via PYTHONHASHSEED, so
    using it to derive a per-case RNG state defeats reproducibility across
    invocations. CRC32 is stable, fast, and well-distributed enough for
    seeding numpy RNGs.
    """
    return zlib.crc32(s.encode("utf-8"))
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PINNED_MLX_VLM_SHA = "d49d428e9f570dc0387b9598b3b7e0ea391590d2"
SEED = 42

# Dense 31B target at 8-bit, drafter at bf16. Reasoning:
# (1) Drafter value prop is cleanest on dense (MoE expert routing at batch 1
#     confounds the speedup claim — see #279 thread). Architecturally the MTP
#     code path is identical for MoE vs dense; this is purely about giving
#     reviewers a clean reference.
# (2) Target 8-bit captures the production numerical contract, not a bf16
#     "purest reference" path no one actually runs.
# (3) Drafter stays bf16: it's ~800MB, so 8-bit savings are marginal against
#     more quantization noise on a tighter parameter count.
# (4) Greedy token outputs (temperature=0) are bit-exact regardless of weight
#     precision — argmax is deterministic. Loosened tolerances apply to
#     activation/logit parity (Rungs 2 and 3) only.
#
# Note the casing asymmetry (lowercase 'b' for the target, uppercase 'B' for the
# drafter) — that's mlx-community's actual repo naming, not a typo.
TARGET_MODEL_ID_DEFAULT = "mlx-community/gemma-4-31b-it-8bit"
DRAFTER_MODEL_ID_DEFAULT = "mlx-community/gemma-4-31B-it-assistant-bf16"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_mlx_vlm_sha() -> None:
    """Verify mlx-vlm is checked out at the pinned SHA. Fail loud otherwise."""
    try:
        import mlx_vlm  # noqa: F401
    except ImportError:
        sys.exit("mlx-vlm is not installed in this environment.")

    mlx_vlm_path = Path(__import__("mlx_vlm").__file__).resolve().parent.parent
    if not (mlx_vlm_path / ".git").exists():
        # Installed via pip from a release — can't verify SHA. Warn but don't fail.
        # User is responsible for matching the release to the pinned SHA out of band.
        print(
            f"WARNING: mlx-vlm at {mlx_vlm_path} is not a git checkout. "
            f"Cannot verify pinned SHA {PINNED_MLX_VLM_SHA}. Proceed with caution.",
            file=sys.stderr,
        )
        return

    try:
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=mlx_vlm_path, text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        sys.exit(f"Failed to read mlx-vlm git HEAD: {e}")

    if actual != PINNED_MLX_VLM_SHA:
        sys.exit(
            f"mlx-vlm is at {actual} but fixtures pin to {PINNED_MLX_VLM_SHA}. "
            f"Either check out the pinned SHA or update PINNED_MLX_VLM_SHA in this script "
            f"after verifying the fixtures are still numerically correct."
        )
    print(f"✓ mlx-vlm pinned at {PINNED_MLX_VLM_SHA}")


def save_fixture(
    path: Path,
    tensors: dict[str, np.ndarray | mx.array],
    metadata: dict[str, str] | None = None,
) -> None:
    """Save a fixture as a safetensors file with optional string metadata.

    Uses ``mx.save_safetensors`` so bfloat16 mlx arrays round-trip without a
    numpy detour (numpy has no native bf16 dtype). NumPy arrays are wrapped
    in ``mx.array`` so the call site can mix dtypes freely.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    converted: dict[str, mx.array] = {}
    for k, v in tensors.items():
        if isinstance(v, mx.array):
            converted[k] = v
        else:
            converted[k] = mx.array(v)
    mx.save_safetensors(str(path), converted, metadata=metadata)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  → {path.relative_to(path.parents[2])}  ({size_mb:.2f} MB)")


@contextlib.contextmanager
def instrumented(instance: Any) -> Any:
    """Per-instance `__call__` capture.

    Yields (instance, calls) where `calls` is a list that grows as the instance
    is invoked. Each entry is a dict with 'args', 'kwargs', and 'output'.

    On exit, restores instance.__class__ to its original type.

    See module docstring for why per-instance subclassing is preferred over
    module-level monkey-patching.
    """
    original_cls = type(instance)
    calls: list[dict[str, Any]] = []

    class Instrumented(original_cls):  # type: ignore[misc, valid-type]
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            output = super().__call__(*args, **kwargs)
            calls.append({"args": args, "kwargs": kwargs, "output": output})
            return output

    # __class__ assignment requires same layout (no conflicting __slots__);
    # mlx.nn.Module subclasses are fine here.
    instance.__class__ = Instrumented
    try:
        yield instance, calls
    finally:
        instance.__class__ = original_cls


# ---------------------------------------------------------------------------
# Mask suite
# ---------------------------------------------------------------------------


def generate_mask_fixtures(out_root: Path) -> list[dict[str, Any]]:
    """Generate bidirectional and bidirectional-SWA mask fixtures.

    These mirror HF Transformers' create_bidirectional_mask and
    create_bidirectional_sliding_window_mask. We compute them directly here so
    the Swift implementations have a byte-identical reference.

    Shape convention: [queryLen, kvLen], additive (0 for attend, -inf for mask).
    SWA mask has the kv-axis flip applied (latest tokens attended, oldest masked).
    """
    print("\n=== Mask suite ===")
    manifest: list[dict[str, Any]] = []

    out_dir = out_root / "masks"

    query_lens = [1, 2, 4]
    kv_lens = [4, 8, 16]
    window_sizes = [2, 4, 8]
    dtype = np.float32  # Masks are always float; consumer casts to match input dtype

    # Bidirectional: all zeros (no causal restriction)
    for q in query_lens:
        for kv in kv_lens:
            mask = np.zeros((q, kv), dtype=dtype)
            path = out_dir / f"bidirectional_q{q}_kv{kv}.safetensors"
            save_fixture(
                path,
                {"mask": mask},
                metadata={"queryLen": str(q), "kvLen": str(kv), "kind": "bidirectional"},
            )
            manifest.append(
                {"path": str(path.relative_to(out_root)), "kind": "bidirectional", "queryLen": q, "kvLen": kv}
            )

    # Bidirectional sliding-window: -inf outside last `windowSize` kv positions
    # Then flip along the kv axis (HF's create_bidirectional_sliding_window_mask convention).
    for q in query_lens:
        for kv in kv_lens:
            for w in window_sizes:
                if w > kv:
                    continue  # window larger than kv is degenerate
                mask = np.full((q, kv), -np.inf, dtype=dtype)
                # Attend only to the last `w` positions (latest tokens)
                mask[:, kv - w :] = 0.0
                # Apply the kv-axis flip
                mask = np.flip(mask, axis=-1).copy()
                path = out_dir / f"bidirectional_swa_q{q}_kv{kv}_w{w}.safetensors"
                save_fixture(
                    path,
                    {"mask": mask},
                    metadata={
                        "queryLen": str(q),
                        "kvLen": str(kv),
                        "windowSize": str(w),
                        "kind": "bidirectional_swa",
                    },
                )
                manifest.append(
                    {
                        "path": str(path.relative_to(out_root)),
                        "kind": "bidirectional_swa",
                        "queryLen": q,
                        "kvLen": kv,
                        "windowSize": w,
                    }
                )

    print(f"Generated {len(manifest)} mask fixtures")
    return manifest


# ---------------------------------------------------------------------------
# Drafter forward suite
# ---------------------------------------------------------------------------


@dataclass
class DrafterContext:
    """Holds loaded drafter + bound target references for fixture generation."""

    drafter: Any  # Gemma4AssistantDraftModel from mlx-vlm
    target: Any  # Gemma4 from mlx-vlm
    processor: Any  # AutoProcessor returned by mlx_vlm.load; held for end_to_end
    drafter_config: Any
    target_config: Any
    backbone_hidden_size: int
    drafter_hidden_size: int
    vocab_size: int
    layer_types: list[str]
    sliding_window: int
    target_model_id: str
    drafter_model_id: str


def load_models(
    target_id: str = TARGET_MODEL_ID_DEFAULT,
    drafter_id: str = DRAFTER_MODEL_ID_DEFAULT,
) -> DrafterContext:
    """Load drafter + target from mlx-vlm and call drafter.bind(target)."""
    from mlx_vlm import load
    from mlx_vlm.speculative.drafters import load_drafter

    print(f"Loading target: {target_id}")
    target, processor = load(target_id)

    print(f"Loading drafter: {drafter_id}")
    # At mlx-vlm d49d428, load_drafter returns (model, resolved_kind)
    drafter, _resolved_kind = load_drafter(drafter_id, kind="mtp")
    drafter.bind(target)

    # Surface key shapes for fixture generation
    drafter_cfg = drafter.config
    target_cfg = target.config.text_config

    return DrafterContext(
        drafter=drafter,
        target=target,
        processor=processor,
        drafter_config=drafter_cfg,
        target_config=target_cfg,
        backbone_hidden_size=drafter_cfg.backbone_hidden_size,
        drafter_hidden_size=drafter_cfg.text_config.hidden_size,
        vocab_size=drafter_cfg.text_config.vocab_size,
        layer_types=target_cfg.layer_types,
        sliding_window=target_cfg.sliding_window,
        target_model_id=target_id,
        drafter_model_id=drafter_id,
    )


def synthetic_shared_kv(
    ctx: DrafterContext,
    kv_len: int,
    batch: int = 1,
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[mx.array, mx.array]]:
    """Build synthetic shared K/V state matching target's layer-type structure.

    Keyed by 'full_attention' and 'sliding_attention'. Values are
    (keys, values) MLXArrays of shape [B, num_kv_heads, kv_len, head_dim].
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    target_cfg = ctx.target_config

    # Per Gemma 4 (mlx-vlm d49d428) `Attention.__init__` in
    # mlx_vlm/models/gemma4/language.py: full_attention layers use
    # `global_head_dim` and (with `attention_k_eq_v=True`)
    # `num_global_key_value_heads`; sliding_attention uses `head_dim`
    # and `num_key_value_heads`. The 31B uses k_eq_v on full layers.
    sliding_kv_heads = target_cfg.num_key_value_heads
    sliding_head_dim = target_cfg.head_dim
    full_head_dim = getattr(target_cfg, "global_head_dim", None) or sliding_head_dim
    if getattr(target_cfg, "attention_k_eq_v", False) and getattr(
        target_cfg, "num_global_key_value_heads", None
    ):
        full_kv_heads = target_cfg.num_global_key_value_heads
    else:
        full_kv_heads = sliding_kv_heads

    # Match target dtype convention (bf16 in checkpoint)
    def make_kv(kv_heads: int, head_dim: int) -> tuple[mx.array, mx.array]:
        keys = rng.standard_normal((batch, kv_heads, kv_len, head_dim)).astype(np.float32)
        values = rng.standard_normal((batch, kv_heads, kv_len, head_dim)).astype(np.float32)
        return mx.array(keys).astype(mx.bfloat16), mx.array(values).astype(mx.bfloat16)

    return {
        "full_attention": make_kv(full_kv_heads, full_head_dim),
        "sliding_attention": make_kv(sliding_kv_heads, sliding_head_dim),
    }


def generate_drafter_forward_fixtures(
    ctx: DrafterContext, out_root: Path
) -> list[dict[str, Any]]:
    """Generate per-call drafter forward fixtures via the per-instance capture path.

    Each fixture captures one invocation of drafter._forward_hidden (or whatever
    the actual private method name turns out to be — verify when running) with
    synthetic inputs.
    """
    print("\n=== Drafter forward suite ===")
    manifest: list[dict[str, Any]] = []
    out_dir = out_root / "drafter_forward"

    cases = [
        {"name": "case_01_q1_kv32_baseline", "query_len": 1, "kv_len": 32},
        {"name": "case_02_q1_kv64", "query_len": 1, "kv_len": 64},
        {"name": "case_03_q1_kv128", "query_len": 1, "kv_len": 128},
        {"name": "case_04_q2_kv32_multi_query", "query_len": 2, "kv_len": 32},
        {"name": "case_05_q1_kv256_long", "query_len": 1, "kv_len": 256},
    ]

    for case in cases:
        rng = np.random.default_rng(SEED + _stable_hash(case["name"]) % 2**32)
        query_len = case["query_len"]
        kv_len = case["kv_len"]

        # Synthesize inputs
        inputs_embeds = mx.array(
            rng.standard_normal(
                (1, query_len, 2 * ctx.backbone_hidden_size)
            ).astype(np.float32)
        ).astype(mx.bfloat16)
        shared_kv = synthetic_shared_kv(ctx, kv_len=kv_len, rng=rng)
        # Position IDs are constant for the round; use kv_len as the offset.
        position_ids = mx.array(np.full((1, query_len), kv_len, dtype=np.int32))

        # Seed drafter round-state. `_forward_hidden` reads `self._kv_valid_len`
        # to build the full-attention mask via `make_drafter_masks`; if it's
        # left at the init default (0), every key position is masked to -inf
        # and the post-softmax attention is NaN. `set_shared_kv` initializes
        # that ivar (and `_position` / `_kv_offset`) to consistent values, so
        # the captured forward sees a real KV pool. The mlx-vlm round-loop
        # calls `set_shared_kv` after prefill for the same reason
        # (speculative/utils.py:594-599); the drafter_block fixture below
        # already does this. Without it the drafter_forward fixtures generated
        # at SHA d49d428 had all-NaN outputs.
        ctx.drafter.set_shared_kv(
            shared_kv_states=shared_kv,
            kv_offset=kv_len,
            position=kv_len,
        )

        # Call drafter through the instrumentation harness. We invoke
        # `__call__` (not `_forward_hidden`) so the lm-head is applied —
        # `_forward_hidden` returns (last_hidden, pre_lm_hidden); __call__
        # returns (last_hidden, logits). See gemma4_assistant.py at
        # mlx-vlm d49d428 lines 152–167.
        with instrumented(ctx.drafter) as (drafter, calls):
            last_hidden, logits = drafter(
                inputs_embeds=inputs_embeds,
                shared_kv_states=shared_kv,
                position_ids=position_ids,
            )
            mx.eval(last_hidden, logits)

        tensors: dict[str, mx.array | np.ndarray] = {
            "inputs/inputs_embeds": inputs_embeds,
            "inputs/position_ids": position_ids,
            "inputs/shared_kv/full_attention/keys": shared_kv["full_attention"][0],
            "inputs/shared_kv/full_attention/values": shared_kv["full_attention"][1],
            "inputs/shared_kv/sliding_attention/keys": shared_kv["sliding_attention"][0],
            "inputs/shared_kv/sliding_attention/values": shared_kv["sliding_attention"][1],
            "outputs/last_hidden": last_hidden,
            "outputs/logits": logits,
        }
        metadata = {
            "queryLen": str(query_len),
            "kvLen": str(kv_len),
            "vocabSize": str(ctx.vocab_size),
            "backboneHiddenSize": str(ctx.backbone_hidden_size),
            "drafterHiddenSize": str(ctx.drafter_hidden_size),
            "targetModelId": ctx.target_model_id,
            "drafterModelId": ctx.drafter_model_id,
        }

        path = out_dir / f"{case['name']}.safetensors"
        save_fixture(path, tensors, metadata=metadata)
        manifest.append({"path": str(path.relative_to(out_root)), **case, **metadata})

    print(f"Generated {len(manifest)} drafter forward fixtures")
    return manifest


# ---------------------------------------------------------------------------
# Drafter block suite
# ---------------------------------------------------------------------------


def generate_drafter_block_fixtures(
    ctx: DrafterContext, out_root: Path
) -> list[dict[str, Any]]:
    """Generate end-to-end drafter `draft_block` fixtures (greedy sampling)."""
    print("\n=== Drafter block suite ===")
    manifest: list[dict[str, Any]] = []
    out_dir = out_root / "drafter_block"

    cases = [
        {"name": "case_01_block2", "block_size": 2},
        {"name": "case_02_block4", "block_size": 4},
        {"name": "case_03_block6", "block_size": 6},
    ]

    for case in cases:
        rng = np.random.default_rng(SEED + _stable_hash(case["name"]) % 2**32)
        block_size = case["block_size"]
        kv_len = 64  # arbitrary; matches a mid-conversation state

        # Synthesize the inputs draftBlock expects after a target verify
        last_token = mx.array(rng.integers(0, ctx.vocab_size, (1,), dtype=np.int32))
        last_hidden = mx.array(
            rng.standard_normal((1, 1, ctx.backbone_hidden_size)).astype(np.float32)
        ).astype(mx.bfloat16)
        shared_kv = synthetic_shared_kv(ctx, kv_len=kv_len, rng=rng)
        position_ids = mx.array(np.full((1, 1), kv_len, dtype=np.int32))

        # At mlx-vlm d49d428: draft_block reads shared_kv/position from state
        # (set via set_shared_kv) and takes (last_bonus, hidden, cache,
        # block_size, sampler, token_dtype, greedy). For temperature=0
        # reproducibility we set greedy=True; when masked_embedding is
        # present (gemma-4 assistant has it) sampler is unused, otherwise
        # the argmax fallback is greedy as well.
        ctx.drafter.set_shared_kv(
            shared_kv_states=shared_kv,
            kv_offset=kv_len,
            position=kv_len,
        )
        # Matches the sampler shape used by mlx-vlm's own test_speculative.py
        # (no keepdims): expected per-token shape (B, 1), not (B, 1, 1).
        greedy_sampler = lambda logits: mx.argmax(logits, axis=-1)
        drafted = ctx.drafter.draft_block(
            last_bonus=last_token,
            hidden=last_hidden,
            cache=None,
            block_size=block_size,
            sampler=greedy_sampler,
            greedy=True,
        )
        mx.eval(drafted)

        tensors: dict[str, mx.array | np.ndarray] = {
            "inputs/last_token": last_token,
            "inputs/last_hidden": last_hidden,
            "inputs/position_ids": position_ids,
            "inputs/shared_kv/full_attention/keys": shared_kv["full_attention"][0],
            "inputs/shared_kv/full_attention/values": shared_kv["full_attention"][1],
            "inputs/shared_kv/sliding_attention/keys": shared_kv["sliding_attention"][0],
            "inputs/shared_kv/sliding_attention/values": shared_kv["sliding_attention"][1],
            "outputs/drafted_tokens": drafted,
        }
        metadata = {
            "blockSize": str(block_size),
            "kvLen": str(kv_len),
            "sampling": "greedy",
            "vocabSize": str(ctx.vocab_size),
            "targetModelId": ctx.target_model_id,
            "drafterModelId": ctx.drafter_model_id,
        }

        path = out_dir / f"{case['name']}.safetensors"
        save_fixture(path, tensors, metadata=metadata)
        manifest.append({"path": str(path.relative_to(out_root)), **case, **metadata})

    print(f"Generated {len(manifest)} drafter block fixtures")
    return manifest


# ---------------------------------------------------------------------------
# End-to-end suite
# ---------------------------------------------------------------------------


def generate_end_to_end_fixtures(
    ctx: DrafterContext, out_root: Path
) -> list[dict[str, Any]]:
    """Generate full target prefill + drafter round + verify fixtures.

    This is the heaviest suite. It exercises the full MTP pipeline against a
    real prompt with the real target. The Swift port should reproduce these
    end-to-end at temperature=0 (byte-identical token sequences).

    Driven through ``mlx_vlm.generate.stream_generate`` with
    ``draft_model=ctx.drafter`` and ``draft_kind="mtp"`` — this exercises the
    same code path real callers use, including the prefill → draft_block →
    verify → accept loop in ``mlx_vlm.speculative.utils._mtp_rounds`` at
    SHA d49d428.

    Acceptance metrics come from the drafter's own ``accept_lens`` and
    ``draft_lens`` lists (one entry per round) — see gemma4_assistant.py
    lines 59–60 + _record_speculative_round in speculative/utils.py.
    """
    print("\n=== End-to-end suite ===")
    manifest: list[dict[str, Any]] = []
    out_dir = out_root / "end_to_end"
    out_dir.mkdir(parents=True, exist_ok=True)

    from mlx_vlm.generate import stream_generate

    # User prompts (chat-wrapped via the tokenizer's template before the model
    # sees them). 31B-IT degenerates badly on raw text continuation; the
    # production path always wraps through the chat template, so the
    # fixtures should reflect that contract.
    user_prompts = [
        "Write one short sentence describing a quiet forest at dawn.",
        "Explain in two short sentences why the sky appears blue during the day.",
    ]
    max_new_tokens = 64

    tokenizer = (
        ctx.processor.tokenizer
        if hasattr(ctx.processor, "tokenizer")
        else ctx.processor
    )

    for i, user_prompt in enumerate(user_prompts, start=1):
        case_name = f"case_{i:02d}_greedy"

        # Apply the model's own chat template; this is what the production
        # CLI does (see mlx_vlm/generate.py: apply_chat_template path).
        if hasattr(tokenizer, "apply_chat_template") and getattr(
            tokenizer, "chat_template", None
        ):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = user_prompt

        prompt_ids_list = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tokens = np.array([prompt_ids_list], dtype=np.int32)

        # Reset acceptance counters before each prompt so they reflect only
        # this case's rounds.
        ctx.drafter.accept_lens = []
        ctx.drafter.draft_lens = []

        accepted: list[int] = []
        for response in stream_generate(
            ctx.target,
            ctx.processor,
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            draft_model=ctx.drafter,
            draft_kind="mtp",
        ):
            tok = getattr(response, "token", None)
            if tok is None:
                continue
            if hasattr(tok, "item"):
                accepted.append(int(tok.item()))
            else:
                accepted.append(int(tok))
            if len(accepted) >= max_new_tokens:
                break

        accept_lens = list(ctx.drafter.accept_lens)
        draft_lens = list(ctx.drafter.draft_lens)
        proposed = sum(draft_lens)
        accepted_count = sum(accept_lens)
        acceptance_rate = (
            float(accepted_count) / float(proposed) if proposed > 0 else 0.0
        )
        num_rounds = len(accept_lens)

        tensors: dict[str, mx.array | np.ndarray] = {
            "inputs/prompt_tokens": prompt_tokens,
            "inputs/temperature": np.array(0.0, dtype=np.float32),
            "outputs/accepted_tokens": np.array(accepted, dtype=np.int32).reshape(
                1, -1
            ),
            "outputs/acceptance_rate": np.array(acceptance_rate, dtype=np.float32),
            "outputs/num_rounds": np.array(num_rounds, dtype=np.int32),
        }
        metadata = {
            "userPrompt": user_prompt,
            "prompt": prompt,
            "targetModelId": ctx.target_model_id,
            "drafterModelId": ctx.drafter_model_id,
            "numNewTokens": str(len(accepted)),
            "maxNewTokens": str(max_new_tokens),
            "acceptLens": ",".join(str(x) for x in accept_lens),
            "draftLens": ",".join(str(x) for x in draft_lens),
            "sampling": "greedy",
            "chatTemplate": "applied" if prompt != user_prompt else "raw",
        }

        path = out_dir / f"{case_name}.safetensors"
        save_fixture(path, tensors, metadata=metadata)
        manifest.append(
            {
                "path": str(path.relative_to(out_root)),
                "userPrompt": user_prompt,
                "numNewTokens": len(accepted),
                "acceptanceRate": acceptance_rate,
                "numRounds": num_rounds,
            }
        )

    print(f"Generated {len(manifest)} end-to-end fixtures")
    return manifest


# ---------------------------------------------------------------------------
# Schema documentation
# ---------------------------------------------------------------------------

SCHEMA_DOC = textwrap.dedent(
    """\
    # MTP Speculative Decoding Fixture Schema

    These fixtures verify the Swift port of Gemma 4 MTP speculative decoding in
    `ml-explore/mlx-swift-lm` against canonical Python reference from `Blaizzy/mlx-vlm`
    pinned at SHA `{sha}`.

    All tensor files use the `safetensors` format. Keys follow a hierarchical
    naming convention with `/` as the separator. Metadata is stored as
    string-keyed string values in each file's metadata block.

    Random seed for all synthetic inputs: `{seed}`.

    ## Suites

    ### `masks/`

    Bidirectional and bidirectional-sliding-window attention masks. Used to
    verify Swift's `createBidirectionalMask(...)` and
    `createBidirectionalSlidingWindowMask(...)` in `MLXLMCommon`.

    Keys per file:
      - `mask`: shape `[queryLen, kvLen]`, dtype `float32`, additive
        (0 = attend, -inf = mask)

    Metadata:
      - `kind`: `"bidirectional"` or `"bidirectional_swa"`
      - `queryLen`, `kvLen`: as integer strings
      - `windowSize`: present only for SWA fixtures

    The SWA mask has the kv-axis flip applied (HF convention): the *last*
    `windowSize` positions in the unflipped mask are attended; the flip
    reorders so the *first* `windowSize` positions in the returned mask are
    attended. Swift implementations must apply the same flip.

    ### `drafter_forward/`

    Single invocations of the drafter's `_forward_hidden` with synthetic inputs.
    Each fixture captures one (inputs → outputs) pair sufficient to verify the
    drafter's per-call forward pass.

    Input keys:
      - `inputs/inputs_embeds`: `[1, queryLen, 2 * backboneHiddenSize]`, bf16
        (concatenation of token embedding and previous hidden state)
      - `inputs/position_ids`: `[1, queryLen]`, int32 (constant within a round)
      - `inputs/shared_kv/full_attention/keys`: `[1, num_kv_heads, kvLen, head_dim]`, bf16
      - `inputs/shared_kv/full_attention/values`: same shape
      - `inputs/shared_kv/sliding_attention/keys`: same shape
      - `inputs/shared_kv/sliding_attention/values`: same shape

    Output keys:
      - `outputs/last_hidden`: `[1, queryLen, backboneHiddenSize]`, bf16
        (post-projected back to target's hidden size for downstream concat)
      - `outputs/logits`: `[1, queryLen, vocabSize]`, fp32

    Metadata: `queryLen`, `kvLen`, `vocabSize`, `backboneHiddenSize`, `drafterHiddenSize`.

    ### `drafter_block/`

    End-to-end invocations of `draft_block` (the within-round K-1 token autoregressive
    loop). Greedy sampling (temperature=0) for reproducibility.

    Input keys:
      - `inputs/last_token`: `[1]`, int32 (the bonus token from target's last verify)
      - `inputs/last_hidden`: `[1, 1, backboneHiddenSize]`, bf16
      - `inputs/position_ids`: `[1, 1]`, int32
      - `inputs/shared_kv/...`: same shape conventions as drafter_forward

    Output keys:
      - `outputs/drafted_tokens`: `[1, blockSize - 1]`, int32

    Metadata: `blockSize`, `kvLen`, `sampling` (always `"greedy"` for these).

    ### `end_to_end/`

    Full target prefill + one or more drafter rounds + verify. Reproduces an
    entire generation slice. Greedy sampling.

    Input keys:
      - `inputs/prompt_tokens`: `[1, promptLen]`, int32
      - `inputs/temperature`: scalar fp32 (always 0.0 for these fixtures)

    Output keys:
      - `outputs/accepted_tokens`: `[1, numAcceptedTokens]`, int32
      - `outputs/acceptance_rate`: scalar fp32
      - `outputs/num_rounds`: scalar int32

    Metadata: `prompt` (the original string), `targetModelId`, `drafterModelId`,
    plus per-suite shape and parameter details.

    Note: every fixture file across every suite carries `targetModelId` and
    `drafterModelId` in its metadata block. Consumers reading a fixture can
    always check the reference pair without needing the global manifest.

    ## Versioning

    Fixtures are versioned by mlx-vlm SHA. If `PINNED_MLX_VLM_SHA` in
    `generate_mtp_fixtures.py` changes, regenerate all fixtures and bump the
    HuggingFace dataset revision. Document the change in the dataset's commit
    message: which SHA, what changed, why fixtures need regeneration.

    ## Reference models

    Default reference checkpoints (overridable via `--target-model-id` and
    `--drafter-model-id`):

      - Target: `mlx-community/gemma-4-31b-it-8bit` (dense, 8-bit weights)
      - Drafter: `mlx-community/gemma-4-31B-it-assistant-bf16` (bf16 weights)

    Dense rather than MoE: see preamble comment in `generate_mtp_fixtures.py`.

    Quantization asymmetry: the target is 8-bit because that's the production
    path real consumers run; the drafter is bf16 because at ~800MB the
    quantization savings are marginal against more numerical noise on a smaller
    parameter count. The drafter's weight precision doesn't affect K/V tensors
    in the cache (those flow through in compute dtype), so the shared K/V the
    Swift port reads from the target is independent of drafter weight quant.

    Each fixture file records `targetModelId` and `drafterModelId` in metadata
    so consumers can verify they're matched against the same reference pair.

    ## Tolerances for verification

    Weight quantization on the target slightly affects intermediate activations
    but does NOT affect greedy token outputs at temperature=0 (argmax is
    deterministic). So:

      - **Rung 4 (greedy token parity)**: bit-exact, no tolerance.
      - **Rung 3 (logit parity, 8-bit target)**: `atol=2e-3, rtol=2e-3`.
      - **Rung 2 (per-layer activations, 8-bit target)**: `atol=2e-3, rtol=2e-3`.
      - **Mask fixtures (fp32)**: `atol=1e-5, rtol=1e-5`.

    Drafter-only fixtures (where the target is touched only for bind() and
    embed_tokens) inherit the tighter bf16 tolerances of `atol=1e-3, rtol=1e-3`,
    because the drafter is bf16 and there's no 8-bit weight quant in its
    compute path.

    ## Regeneration

    From the same M5 Pro with mlx-vlm at the pinned SHA:

    ```
    python tools/generate_mtp_fixtures.py
    ```

    Then upload to HuggingFace:

    ```
    huggingface-cli upload <dataset-id> ./fixtures --repo-type=dataset
    ```
    """
)


def write_schema_doc(out_root: Path) -> None:
    path = out_root / "FIXTURE-SCHEMA.md"
    path.write_text(SCHEMA_DOC.format(sha=PINNED_MLX_VLM_SHA, seed=SEED))
    print(f"\n✓ Schema written to {path}")


def write_manifest(out_root: Path, manifests: dict[str, list[dict[str, Any]]]) -> None:
    path = out_root / "FIXTURE-MANIFEST.json"
    path.write_text(json.dumps(
        {
            "pinned_mlx_vlm_sha": PINNED_MLX_VLM_SHA,
            "seed": SEED,
            "suites": manifests,
        },
        indent=2,
    ))
    print(f"✓ Manifest written to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fixtures"),
        help="Output directory (default: ./fixtures)",
    )
    parser.add_argument(
        "--suite",
        choices=["masks", "drafter_forward", "drafter_block", "end_to_end", "all"],
        default="all",
        help="Generate just one suite, or all (default: all)",
    )
    parser.add_argument(
        "--target-model-id",
        default=TARGET_MODEL_ID_DEFAULT,
        help=f"Target model HF ID (default: {TARGET_MODEL_ID_DEFAULT})",
    )
    parser.add_argument(
        "--drafter-model-id",
        default=DRAFTER_MODEL_ID_DEFAULT,
        help=f"Drafter model HF ID (default: {DRAFTER_MODEL_ID_DEFAULT})",
    )
    parser.add_argument(
        "--skip-end-to-end",
        action="store_true",
        help="Skip the end-to-end suite (which loads the full target model)",
    )
    args = parser.parse_args()

    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out_root}")
    print(f"Suite: {args.suite}")
    assert_mlx_vlm_sha()

    manifests: dict[str, list[dict[str, Any]]] = {}

    needs_models = args.suite in ("drafter_forward", "drafter_block", "end_to_end", "all")
    ctx: DrafterContext | None = None

    if args.suite in ("masks", "all"):
        manifests["masks"] = generate_mask_fixtures(out_root)

    if needs_models:
        if args.suite == "end_to_end" or (args.suite == "all" and not args.skip_end_to_end):
            ctx = load_models(
                target_id=args.target_model_id,
                drafter_id=args.drafter_model_id,
            )
        elif args.suite in ("drafter_forward", "drafter_block") or (
            args.suite == "all" and args.skip_end_to_end
        ):
            ctx = load_models(
                target_id=args.target_model_id,
                drafter_id=args.drafter_model_id,
            )
        # Note: if running just drafter_forward or drafter_block, target is loaded
        # only because drafter.bind() needs it. The target's weights are touched
        # via embed_tokens access only. Future optimization: load target lazily,
        # only materializing embed_tokens. Skipped for now to keep the script simple.

    if args.suite in ("drafter_forward", "all"):
        assert ctx is not None
        manifests["drafter_forward"] = generate_drafter_forward_fixtures(ctx, out_root)

    if args.suite in ("drafter_block", "all"):
        assert ctx is not None
        manifests["drafter_block"] = generate_drafter_block_fixtures(ctx, out_root)

    if args.suite in ("end_to_end", "all") and not args.skip_end_to_end:
        assert ctx is not None
        manifests["end_to_end"] = generate_end_to_end_fixtures(ctx, out_root)

    write_schema_doc(out_root)
    write_manifest(out_root, manifests)

    total = sum(len(m) for m in manifests.values())
    print(f"\n✓ Generated {total} fixtures across {len(manifests)} suite(s)")


if __name__ == "__main__":
    main()
