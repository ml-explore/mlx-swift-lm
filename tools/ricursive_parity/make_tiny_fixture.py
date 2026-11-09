"""Writes Tests/MLXLMTests/Resources/ricursive_tiny (config, weights, oracle.json).

Random float32 weights, 5 layers (four sliding, one full), an 8-token window and a
prompt longer than it, so the cached path crosses eviction. Run with -B from the
repository root: python3 -B tools/ricursive_parity/make_tiny_fixture.py
"""

import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from ricursive_torch import RicursiveTorch

OUT = Path(__file__).parents[2] / "Tests/MLXLMTests/Resources/ricursive_tiny"
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
PROMPT_LENGTH = 24
DECODE = 12
PREFILL_STEPS = [1, 3, 8, 23]


def make_weights(generator):
    c, w = CONFIG, {}
    heads, kv, head = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    hidden, inner = c["hidden_size"], c["intermediate_size"]

    def normal(*shape, std=0.15):
        return std * torch.randn(*shape, generator=generator)

    def norm(size):
        return 1 + normal(size, std=0.1)

    w["model.embed_tokens.weight"] = normal(c["vocab_size"], hidden)
    w["lm_head.weight"] = normal(c["vocab_size"], hidden)
    w["model.norm.weight"] = norm(hidden)
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}"
        a = p + ".self_attn"
        w[a + ".q_proj.weight"] = normal(heads * head, hidden)
        w[a + ".k_proj.weight"] = normal(kv * head, hidden)
        w[a + ".v_proj.weight"] = normal(kv * head, hidden)
        w[a + ".o_proj.weight"] = normal(hidden, heads * head)
        w[a + ".g_proj.weight"] = normal(heads * head, hidden)
        w[a + ".q_norm.weight"] = norm(head)
        w[a + ".k_norm.weight"] = norm(head)
        w[a + ".value_res_lambda"] = normal(1, std=1.0)
        w[p + ".mlp.gate_proj.weight"] = normal(inner, hidden)
        w[p + ".mlp.up_proj.weight"] = normal(inner, hidden)
        w[p + ".mlp.down_proj.weight"] = normal(hidden, inner)
        for name in ("input", "post_attention", "pre_feedforward", "post_feedforward"):
            w[f"{p}.{name}_layernorm.weight"] = norm(hidden)
    return w


def main():
    generator = torch.Generator().manual_seed(2270)
    weights = make_weights(generator)
    prompt = torch.randint(0, CONFIG["vocab_size"], (PROMPT_LENGTH,), generator=generator).tolist()
    model = RicursiveTorch(CONFIG, weights, torch.device("cpu"))

    full = model.forward(torch.tensor([prompt]))[0]

    cases = []
    for step in PREFILL_STEPS:
        greedy, logits = [], []

        def record(_, token, row):
            greedy.append(token)
            logits.append(row.tolist())

        model.prefill_and_generate(prompt, DECODE, step, record)
        cases.append({"prefill_step": step, "greedy": greedy, "logits": logits})

    # Every chunking must give the same decode: the cached path is exact.
    for case in cases[1:]:
        assert case["greedy"] == cases[0]["greedy"], "chunking changed the greedy tokens"
        difference = (torch.tensor(case["logits"]) - torch.tensor(cases[0]["logits"])).abs().max()
        assert difference < 1e-4, f"chunking changed the logits by {difference}"

    # The cached last-prompt logits must equal the uncached forward's.
    assert (torch.tensor(cases[0]["logits"][0]) - full[-1]).abs().max() < 1e-4

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    save_file(weights, OUT / "model.safetensors")
    oracle = {"prompt": prompt, "full_logits": full.tolist(), "cases": cases}
    (OUT / "oracle.json").write_text(json.dumps(oracle) + "\n")
    print("greedy", cases[0]["greedy"])


if __name__ == "__main__":
    main()
