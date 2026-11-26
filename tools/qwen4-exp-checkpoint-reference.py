#!/usr/bin/env python3
"""Export reference outputs for the mlx-community Qwen3.8-Flash-Next OptiQ-2bit checkpoint.

Runs mlx-lm's qwen4_exp model (ml-explore/mlx-lm@c4c9c762cc797fd979fc98645f5db5112753bbe9)
on the GPU over the same quantized weights the Swift test loads. The checkpoint is too large
to dequantize to FP32, so both sides run the published precision. Uses about 80 GB of memory;
run it on its own, then run Qwen4ExpCheckpointTests with MLX_QWEN4_EXP_CHECKPOINT and
MLX_QWEN4_EXP_REFERENCE_FILE set in the test process environment.

The gated delta recurrence takes the route of MLXLMCommon's `gatedDeltaUpdate` for this head
layout: `mx.fast.gated_delta_update` with float32 q, k and beta, where mlx-lm runs its own
kernel in the input dtype. The two kernels round differently, and MoE routing amplifies the
difference over 48 layers. With the same route, the Swift model matches this reference bit for
bit through layer 33 on the 19-token prompt, and drifts by bf16 rounding after that.

python tools/qwen4-exp-checkpoint-reference.py /path/to/checkpoint /tmp/qwen4-exp-reference.safetensors
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen4_exp
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.gated_delta import compute_g

REVISION = "52a21f4a18963bffa64f5a356e10459f2513a900"
PREFILL_STEP = 127
DECODE_STEPS = 16
CHAT_STEPS = 48
PROMPT = "What is the capital of France? Answer in one short sentence."


def swift_gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, **_):
    """MLXLMCommon `gatedDeltaUpdate` for layouts `MLXFast.gatedDeltaUpdate` supports."""
    if mask is not None:
        raise ValueError("The reference prompts are not padded")
    beta = mx.sigmoid(b).astype(mx.float32)
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, _, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    y, state = mx.fast.gated_delta_update(
        q.astype(mx.float32), k.astype(mx.float32), v, g, beta, state.astype(mx.float32))
    return y.astype(q.dtype), state


def greedy(model, tokens, steps):
    """Last-position logits of a prefill in `PREFILL_STEP` chunks, then `steps` greedy tokens."""
    cache = make_prompt_cache(model)
    for start in range(0, tokens.size, PREFILL_STEP):
        logits = model(tokens[None, start : start + PREFILL_STEP], cache=cache)
        mx.eval(logits)
    logits = logits[:, -1].astype(mx.float32)
    prefill_logits = logits
    generated = []
    for _ in range(steps):
        token = mx.argmax(logits, axis=-1)
        generated.append(token.item())
        logits = model(token[None], cache=cache)[:, -1].astype(mx.float32)
        mx.eval(logits)
    return prefill_logits, mx.array(generated, dtype=mx.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.checkpoint.resolve().name != REVISION:
        raise ValueError(f"Expected the snapshot directory of revision {REVISION}")
    qwen4_exp.gated_delta_update = swift_gated_delta_update

    start = time.monotonic()
    model, tokenizer = load(str(args.checkpoint))
    mx.eval(model.parameters())
    print(f"load: {time.monotonic() - start:.1f}s, active={mx.get_active_memory() / 2**30:.1f} GiB")
    vocab = json.loads((args.checkpoint / "config.json").read_text())["text_config"]["vocab_size"]

    outputs = {}
    for length in [19, 257]:
        tokens = mx.array([(i * 7 + 3) % vocab for i in range(length)], dtype=mx.int32)
        start = time.monotonic()
        logits, generated = greedy(model, tokens, DECODE_STEPS)
        outputs[f"tokens_{length}"] = tokens
        outputs[f"logits_{length}"] = logits
        outputs[f"greedy_{length}"] = generated
        print(f"{length} tokens: {time.monotonic() - start:.1f}s, top1={mx.argmax(logits).item()}")

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, enable_thinking=False)
    tokens = mx.array(chat, dtype=mx.int32)
    logits, generated = greedy(model, tokens, CHAT_STEPS)
    outputs["tokens_chat"] = tokens
    outputs["logits_chat"] = logits
    outputs["greedy_chat"] = generated
    print("chat:", repr(tokenizer.decode(generated.tolist())))
    print(f"peak={mx.get_peak_memory() / 2**30:.1f} GiB")

    mx.save_safetensors(str(args.output), outputs, metadata={
        "checkpoint_revision": REVISION,
        "mlx_lm_revision": "c4c9c762cc797fd979fc98645f5db5112753bbe9",
        "gated_delta": "mx.fast.gated_delta_update, float32 q/k/beta",
        "prompt": PROMPT,
    })


if __name__ == "__main__":
    main()
