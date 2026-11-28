# Copyright © 2026 Apple Inc.

"""Write reference outputs of a DeepSeek-V4.1 checkpoint with mlx-lm.

The Swift checkpoint test repeats these steps and compares its results:
an uncached forward over the prompt, a cached prefill in chunks, then greedy
decode steps. Run it from the repository root, with an mlx-lm checkout:

    uv run --python 3.12 --with-editable <mlx-lm checkout> \\
        python tools/deepseek-v4-reference/deepseek-v41-checkpoint-reference.py \\
        MODEL OUTPUT.safetensors
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load

TEXT = (
    "The river bends twice before it reaches the old mill. Past the second bend, "
    "the water slows and widens into a shallow pool where herons wait for fish. "
    "In spring the miller's children count the herons, and in autumn they count "
    "the leaves that drift over the weir. "
)


def log_softmax(logits):
    logits = logits.astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--prompt-tokens", type=int, default=1100)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--tail", type=int, default=64)
    args = parser.parse_args()

    start = time.time()
    model, tokenizer = load(args.model)
    print(f"loaded in {time.time() - start:.1f}s")

    ids = tokenizer.encode(TEXT * (args.prompt_tokens // 40 + 1))
    tokens = mx.array(ids[: args.prompt_tokens], dtype=mx.int32)
    assert tokens.size == args.prompt_tokens

    start = time.time()
    full = log_softmax(model(tokens[None])[0])
    mx.eval(full)
    print(f"uncached forward in {time.time() - start:.1f}s")

    cache = model.make_cache()
    start = time.time()
    chunks = []
    for begin in range(0, tokens.size, args.chunk):
        chunks.append(log_softmax(model(tokens[None, begin : begin + args.chunk], cache=cache)[0]))
        mx.eval(chunks[-1])
    cached = mx.concatenate(chunks)
    prefill = cached[-1]
    print(f"cached prefill in {time.time() - start:.1f}s")

    # The gap between mlx-lm's own uncached and cached paths: the noise floor
    # the Swift comparison is held to.
    baseline_agreement = (mx.argmax(full, -1) == mx.argmax(cached, -1)).mean().item()
    baseline_kl = (mx.exp(full) * (full - cached)).sum(-1).mean().item()
    print(f"mlx-lm uncached vs cached: argmax agreement {baseline_agreement:.4f}, KL {baseline_kl:.3e}")

    generated, decode = [], []
    step = prefill
    for _ in range(args.decode):
        token = mx.argmax(step).astype(mx.int32)
        generated.append(token)
        step = log_softmax(model(token.reshape(1, 1), cache=cache)[0, -1])
        mx.eval(step)
        decode.append(step)

    outputs = {
        "tokens": tokens,
        "full_argmax": mx.argmax(full, axis=-1).astype(mx.int32),
        "full_tail": full[-args.tail :],
        "prefill_last": prefill,
        "generated": mx.stack(generated),
        "decode": mx.stack(decode),
    }
    metadata = {
        "chunk": str(args.chunk),
        "model": args.model,
        "baseline_argmax_agreement": str(baseline_agreement),
        "baseline_kl": str(baseline_kl),
    }
    mx.save_safetensors(args.output, outputs, metadata=metadata)
    print("generated:", [t.item() for t in generated])
    print("text:", repr(tokenizer.decode([t.item() for t in generated])))


if __name__ == "__main__":
    main()
