#!/usr/bin/env python3
"""Print deterministic LFM3 vision reference values for LFM3VisionTests.

Run with numpy installed. This uses explicit spatial convolution and attention
so the Swift tests compare against an independent implementation without files
of serialized weights. Reference architecture is LiquidAI/LFM3-VL-2B,
revision 71d8e745d97c224984e0f20412f532a5ce9dfd3e.
"""

import numpy as np


def weights():
    shapes = {"patchifier.weight": (16, 24), "embedding_norm.weight": (16,)}
    for i in range(2):
        prefix = f"layers.{i}."
        shapes[prefix + "operator_norm.weight"] = (16,)
        shapes[prefix + "ffn_norm.weight"] = (16,)
        for name, shape in {"w1": (24, 16), "w2": (16, 24), "w3": (24, 16)}.items():
            shapes[prefix + f"feed_forward.{name}.weight"] = shape
    for name, shape in {
        "in_proj": (48, 16), "out_proj": (16, 16), "spatial_conv": (16, 3, 3, 1)
    }.items():
        shapes[f"layers.0.conv.{name}.weight"] = shape
    for name, shape in {
        "q_proj": (16, 16), "k_proj": (8, 16), "v_proj": (8, 16),
        "out_proj": (16, 16), "q_layernorm": (8,), "k_layernorm": (8,)
    }.items():
        shapes[f"layers.1.self_attn.{name}.weight"] = shape
    return {
        name: (np.sin(np.arange(np.prod(shape), dtype=np.float32) + i + 1) * 0.1).reshape(shape)
        for i, (name, shape) in enumerate(sorted(shapes.items()))
    }


def reference():
    w = weights()

    def linear(x, name):
        return x @ w[name + ".weight"].T

    def norm(x, name):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5) * w[name + ".weight"]

    def mlp(x, prefix):
        gate = linear(x, prefix + "w1")
        return linear(gate / (1 + np.exp(-gate)) * linear(x, prefix + "w3"), prefix + "w2")

    # Patch sequence uses 2x2 merge blocks on a 2x4 rectangular grid.
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    pixels = np.sin(np.arange(192, dtype=np.float32)).reshape(8, 24)
    x = norm(linear(pixels, "patchifier"), "embedding_norm")
    p = "layers.0."
    b, c, values = np.split(linear(norm(x, p + "operator_norm"), p + "conv.in_proj"), 3, axis=-1)
    grid = np.zeros((2, 4, 16), dtype=np.float32)
    for index, (row, col) in enumerate(positions):
        grid[row, col] = (b * values)[index]
    filtered = np.zeros_like(grid)
    kernel = w[p + "conv.spatial_conv.weight"]
    for row in range(2):
        for col in range(4):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if 0 <= row + dy < 2 and 0 <= col + dx < 4:
                        filtered[row, col] += grid[row + dy, col + dx] * kernel[:, dy + 1, dx + 1, 0]
    x += linear(c * np.stack([filtered[r, c] for r, c in positions]), p + "conv.out_proj")
    x += mlp(norm(x, p + "ffn_norm"), p + "feed_forward.")
    p = "layers.1."
    a = p + "self_attn."
    h = norm(x, p + "operator_norm")
    q = norm(linear(h, a + "q_proj").reshape(8, 2, 8), a + "q_layernorm")
    k = norm(linear(h, a + "k_proj").reshape(8, 1, 8), a + "k_layernorm")
    v = linear(h, a + "v_proj").reshape(8, 1, 8)
    freq = 1000000.0 ** (-np.arange(0, 4, 2, dtype=np.float32) / 4)
    axes = np.asarray(positions, dtype=np.float32)[:, :, None] * freq
    angles = axes.reshape(8, 4)
    angles = np.concatenate([angles, angles], axis=-1)[:, None, :]

    def rotate(tensor):
        rotated = np.concatenate([-tensor[..., 4:], tensor[..., :4]], axis=-1)
        return tensor * np.cos(angles) + rotated * np.sin(angles)

    q, k = rotate(q).transpose(1, 0, 2), rotate(k).transpose(1, 0, 2)
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(np.float32(8))
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    attended = (probs @ v.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(8, 16)
    x += linear(attended, a + "out_proj")
    x += mlp(norm(x, p + "ffn_norm"), p + "feed_forward.")
    return x


if __name__ == "__main__":
    for row in reference():
        print("            " + ", ".join(f"{v:.9g}" for v in row) + ",")
