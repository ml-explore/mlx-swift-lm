"""Writes Tests/MLXLMTests/RicursiveReferenceValues.swift from the PyTorch reference.

A tiny float32 model: 5 layers (four sliding, one full), an 8-token window and a
prompt longer than it, so the cached path crosses eviction. Every weight comes from a
closed-form formula over the parameter names in sorted order (see `make_weights`), so
RicursiveTests builds the same weights exactly in Swift. The reference forward is
`ricursive_torch.py`. Requires torch and safetensors. Run from the repository root:
python3 -B tools/ricursive_parity/ricursive-reference.py
"""

import json
import math
from pathlib import Path

import torch

from ricursive_torch import RicursiveTorch

OUTPUT = Path(__file__).resolve().parents[2] / "Tests/MLXLMTests/RicursiveReferenceValues.swift"
CONFIG = {
    "model_type": "ricursive",
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "sliding_window": 8,
    "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
    "use_gated_attention": True,
    "gate_full_precision": True,
    "use_valres": True,
    "ffn_clamp_threshold": 2.0,
    "embed_scale": math.sqrt(32),
    "hidden_act": "silu",
    "tie_word_embeddings": False,
}
PROMPT = [(i * 37 + 11) % 64 for i in range(24)]
DECODE = 12
PREFILL_STEPS = [1, 3, 8, 23]


def parameter_shapes():
    c, shapes = CONFIG, {}
    heads, kv, head = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    hidden, inner = c["hidden_size"], c["intermediate_size"]
    shapes["model.embed_tokens.weight"] = (c["vocab_size"], hidden)
    shapes["lm_head.weight"] = (c["vocab_size"], hidden)
    shapes["model.norm.weight"] = (hidden,)
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}"
        a = p + ".self_attn"
        shapes[a + ".q_proj.weight"] = (heads * head, hidden)
        shapes[a + ".k_proj.weight"] = (kv * head, hidden)
        shapes[a + ".v_proj.weight"] = (kv * head, hidden)
        shapes[a + ".o_proj.weight"] = (hidden, heads * head)
        shapes[a + ".g_proj.weight"] = (heads * head, hidden)
        shapes[a + ".q_norm.weight"] = (head,)
        shapes[a + ".k_norm.weight"] = (head,)
        shapes[a + ".value_res_lambda"] = (1,)
        shapes[p + ".mlp.gate_proj.weight"] = (inner, hidden)
        shapes[p + ".mlp.up_proj.weight"] = (inner, hidden)
        shapes[p + ".mlp.down_proj.weight"] = (hidden, inner)
        for name in ("input", "post_attention", "pre_feedforward", "post_feedforward"):
            shapes[f"{p}.{name}_layernorm.weight"] = (hidden,)
    return shapes


def make_weights():
    """Parameter `i` in sorted name order, element `j` in row-major order:
    `((j * j * 7 + j * 13 + i * 31) % 97 - 48) / 256`, plus 1 for norm weights;
    `value_res_lambda` divides by 32 instead. The divisors are powers of two, so the
    float32 values are exact on both sides."""
    weights = {}
    for i, (name, shape) in enumerate(sorted(parameter_shapes().items())):
        j = torch.arange(math.prod(shape), dtype=torch.int64)
        values = ((j * j * 7 + j * 13 + i * 31) % 97 - 48).float()
        if name.endswith("value_res_lambda"):
            values = values / 32
        else:
            values = values / 256
            if name.endswith("norm.weight"):
                values = values + 1
        weights[name] = values.reshape(shape)
    return weights


def floats(values, indent="        "):
    return [
        indent + ", ".join(f"{value:.9e}" for value in values[offset : offset + 5]) + ","
        for offset in range(0, len(values), 5)
    ]


def main():
    torch.set_num_threads(4)
    model = RicursiveTorch(CONFIG, make_weights(), torch.device("cpu"))

    full = model.forward(torch.tensor([PROMPT]))[0]

    cases = []
    for step in PREFILL_STEPS:
        greedy, logits = [], []

        def record(_, token, row):
            greedy.append(token)
            logits.append(row)

        model.prefill_and_generate(PROMPT, DECODE, step, record)
        cases.append((greedy, torch.stack(logits)))

    # Every chunking must give the same decode: the cached path is exact. The Swift test
    # compares each of its chunkings with the first case's logits.
    chunking = 0.0
    for greedy, logits in cases[1:]:
        assert greedy == cases[0][0], "chunking changed the greedy tokens"
        chunking = max(chunking, (logits - cases[0][1]).abs().max().item())
    assert chunking < 1e-4, f"chunking changed the logits by {chunking}"

    # The cached last-prompt logits must equal the uncached forward's.
    cached = (cases[0][1][0] - full[-1]).abs().max().item()
    assert cached < 1e-4, f"cached and uncached logits differ by {cached}"

    # Greedy decisions must not hinge on a near tie.
    top = cases[0][1].topk(2, dim=-1).values
    margin = (top[:, 0] - top[:, 1]).min().item()
    assert margin > 1e-2, f"a greedy decision has margin {margin}"

    greedy, logits = cases[0]
    lines = [
        "// Generated by tools/ricursive_parity/ricursive-reference.py; do not edit values by hand.",
        "// Reference: tools/ricursive_parity/ricursive_torch.py (a transcription of",
        "// modeling_ricursive.py in ricursiveAI/Ripple-65B-Preview);",
        f"// PyTorch {torch.__version__.split('+')[0]} CPU FP32.",
        "",
        "enum RicursiveReferenceValues {",
        '    static let configuration = """',
        *["        " + line for line in json.dumps(CONFIG, indent=2).splitlines()],
        '        """',
        "",
        "    static let prompt = [",
        "        " + ", ".join(map(str, PROMPT)) + ",",
        "    ]",
        f"    static let prefillSteps = {PREFILL_STEPS}",
        f"    static let greedy = {greedy}",
        "",
        f"    /// Uncached logits for every prompt position, `[{len(PROMPT)}, vocab]` row-major.",
        "    static let fullLogits: [Float] = [",
        *floats(full.flatten().tolist()),
        "    ]",
        "",
        f"    /// The logits each greedy token was chosen from, `[{DECODE}, vocab]` row-major.",
        "    static let decodeLogits: [Float] = [",
        *floats(logits.flatten().tolist()),
        "    ]",
        "}",
        "",
    ]
    OUTPUT.write_text("\n".join(lines))
    print("greedy", greedy)
    print(f"chunking max |dlogit| {chunking:.3g}, cached vs uncached {cached:.3g}")
    print(f"smallest greedy margin {margin:.4g}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
