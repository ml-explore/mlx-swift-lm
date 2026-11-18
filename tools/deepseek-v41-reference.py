# Copyright © 2026 Apple Inc.

"""Write Tests/MLXLMTests/DeepseekV41ReferenceValues.swift from mlx-lm's deepseek_v41.

Run from the repository root, with an mlx-lm checkout:

    uv run --with mlx --with-editable <mlx-lm checkout> python tools/deepseek-v41-reference.py

The tiny model's weights come from `deterministic_weight`, which DeepseekV41Tests rebuilds
in Swift, so no weights are stored. Everything runs on the CPU.
"""

import argparse
import math
import subprocess
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import deepseek_v41

VOCAB = 128

# The tiny config of mlx-lm's tests/test_deepseek_v41.py, with YaRN scaling added so the
# compressed layers use it.
CONFIG = dict(
    vocab_size=VOCAB,
    hidden_size=64,
    num_hidden_layers=6,
    num_attention_heads=4,
    head_dim=32,
    q_lora_rank=32,
    qk_rope_head_dim=8,
    o_groups=2,
    o_lora_rank=16,
    moe_intermediate_size=32,
    n_routed_experts=8,
    num_experts_per_tok=2,
    sliding_window=4,
    compress_ratios=(0, 2, 2, 2, 1, 1),
    kv_source_layer_ids=(1, 4),
    index_source_layer_ids=(1, 2, 4, 5),
    index_n_heads=4,
    index_head_dim=32,
    index_topk=3,
    candidate_source_layer_id=4,
    candidate_topk_blocks=2,
    candidate_block_size=2,
    rope_scaling={
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 16,
        "beta_fast": 32,
        "beta_slow": 1,
    },
    max_position_embeddings=64,
)

BATCH, LENGTH = 2, 16

# Raw DeepSeek names that cover every rule of `_rename`.
RAW_NAMES = [
    "embed.weight",
    "head.weight",
    "norm.weight",
    "layers.0.attn_norm.weight",
    "layers.0.ffn_norm.weight",
    "layers.0.hc_attn_fn",
    "layers.0.hc_ffn_base",
    "layers.0.attn.attn_sink",
    "layers.0.attn.wq_a.weight",
    "layers.0.attn.wq_b.weight",
    "layers.0.attn.q_norm.weight",
    "layers.0.attn.wkv.weight",
    "layers.0.attn.kv_norm.weight",
    "layers.0.attn.wo_a.weight",
    "layers.0.attn.wo_b.weight",
    "layers.1.attn.compressor.wkv.weight",
    "layers.1.attn.compressor.wgate.weight",
    "layers.1.attn.compressor.norm.weight",
    "layers.1.attn.indexer.wq_b.weight",
    "layers.1.attn.indexer.wk.weight",
    "layers.1.attn.indexer.k_norm.weight",
    "layers.1.attn.indexer.weights_proj.weight",
    "layers.2.ffn.gate.weight",
    "layers.2.ffn.gate.bias",
    "layers.2.ffn.shared_experts.w1.weight",
    "layers.2.ffn.shared_experts.w2.weight",
    "layers.2.ffn.shared_experts.w3.weight",
]


def phase(name: str) -> float:
    h = 0
    for c in name.encode():
        h = (h * 31 + c) % 1_000_003
    return (h % 997) / 997 * 2 * math.pi


def deterministic_weight(name: str, shape: tuple[int, ...]) -> mx.array:
    x = mx.sin(mx.arange(math.prod(shape), dtype=mx.float32) * 0.7713 + phase(name))
    x = x.reshape(shape)
    if name.endswith("norm.weight") or name.endswith(".scale"):
        return 1 + 0.1 * x
    if len(shape) >= 2:
        return x / math.sqrt(shape[-1])
    return 0.5 * x


def tokens() -> mx.array:
    rows = [[(i * 7 + 3 + 11 * b) % VOCAB for i in range(LENGTH)] for b in range(BATCH)]
    return mx.array(rows, dtype=mx.int32)


def fake_quant_input() -> mx.array:
    return 3 * mx.sin(mx.arange(2 * 64, dtype=mx.float32) * 1.37).reshape(2, 64)


def floats(x: mx.array) -> str:
    values = x.astype(mx.float32).reshape(-1).tolist()
    return " ".join(f"{v:.9g}" for v in values)


def swift_string(name: str, value: str) -> str:
    lines = []
    words = value.split(" ")
    for i in range(0, len(words), 8):
        lines.append("        " + " ".join(words[i : i + 8]))
    body = "\n".join(lines)
    return f'    static let {name} = """\n{body}\n        """\n'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Tests/MLXLMTests/DeepseekV41ReferenceValues.swift"),
    )
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)

    model = deepseek_v41.Model(deepseek_v41.ModelArgs(**CONFIG))
    params = tree_flatten(model.parameters())
    model.update(
        tree_unflatten([(k, deterministic_weight(k, v.shape)) for k, v in params])
    )
    mx.eval(model.parameters())

    x = tokens()
    full = model(x)
    cache = model.make_cache()
    steps = [model(x[:, s:e], cache=cache) for s, e in ((0, 7), (7, 15), (15, 16))]
    cached = mx.concatenate(steps, axis=1)
    assert mx.allclose(cached, full, rtol=1e-4, atol=1e-4).item(), "mlx-lm cache mismatch"

    q = fake_quant_input()
    fp8 = deepseek_v41._fake_quant_fp8(q, 32)
    fp4 = deepseek_v41._fake_quant_fp4(q, block_size=32)
    fp4_e4m3 = deepseek_v41._fake_quant_fp4(q, block_size=16, e4m3_scale=True)

    weight = (mx.arange(64 * 64, dtype=mx.uint32) % 251).astype(mx.uint8).reshape(64, 64)
    scale = mx.array([[0.5, 2.0], [4.0, 0.125]], dtype=mx.float32)
    packed, ue8m0 = deepseek_v41._repack(weight, scale)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(deepseek_v41.__file__).parent,
        capture_output=True,
        text=True,
    ).stdout.strip()

    shapes = ",\n".join(
        f'        "{k}": {list(v.shape)}' for k, v in sorted(params, key=lambda p: p[0])
    )
    renames = ",\n".join(
        f'        "{k}": "{deepseek_v41._rename(k)}"' for k in RAW_NAMES
    )
    token_rows = ",\n".join(f"        {row}" for row in x.tolist())
    out = [
        "// Copyright © 2026 Apple Inc.\n",
        "\n",
        "// Generated by tools/deepseek-v41-reference.py from mlx-lm\n",
        f"// {commit}. Do not edit.\n",
        "\n",
        "enum DeepseekV41Reference {\n",
        f'    static let mlxLMCommit = "{commit}"\n',
        "\n",
        "    static let parameterShapes: [String: [Int]] = [\n",
        shapes,
        ",\n    ]\n\n",
        "    static let renames: [String: String] = [\n",
        renames,
        ",\n    ]\n\n",
        "    static let tokens: [[Int32]] = [\n",
        token_rows,
        ",\n    ]\n\n",
        f"    /// `[{BATCH}, {LENGTH}, {VOCAB}]` logits of one full forward.\n",
        swift_string("logits", floats(full)),
        "\n",
        swift_string("fakeQuantFP8", floats(fp8)),
        "\n",
        swift_string("fakeQuantFP4", floats(fp4)),
        "\n",
        swift_string("fakeQuantFP4E4M3", floats(fp4_e4m3)),
        "\n",
        f"    static let repackedWords: [UInt32] = {packed.reshape(-1)[:8].tolist()}\n",
        f"    static let repackedShape: [Int] = {list(packed.shape)}\n",
        f"    static let repackedScales: [UInt8] = {ue8m0[::31].reshape(-1).tolist()}\n",
        f"    static let repackedScalesShape: [Int] = {list(ue8m0.shape)}\n",
        "}\n",
    ]
    args.output.write_text("".join(out))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
