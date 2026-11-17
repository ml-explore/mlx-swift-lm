"""Writes Tests/MLXLMTests/RicursiveReferenceValues.swift from the model author's code.

The reference is `modeling_ricursive.py` (with `configuration_ricursive.py` beside it)
from the ricursiveAI/Ripple-65B-Preview Hugging Face repository, imported unmodified
from the directory given by `--reference` and run in float32 on the CPU. Requires torch
and transformers. Run from the repository root:

    python3 -B tools/ricursive_parity/ricursive-reference.py \\
        --reference <dir>/modeling_ricursive.py --revision <repository commit>

A tiny model: 5 layers (four sliding, one full), an 8-token window and a prompt longer
than it, so the cached path crosses eviction. Every weight comes from a closed-form
formula over the parameter names in sorted order (see `formula_weights`), so
RicursiveTests builds the same weights exactly in Swift. The sorted parameter names and
shapes are written out too, so a renamed Swift parameter fails the tests.
"""

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import torch
import transformers
from transformers.cache_utils import DynamicCache

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
SEEDS = range(32)


def formula_weights(shapes, seed):
    """Parameter `i` in sorted name order, element `j` in row-major order, takes the
    integer `v = (j * j * 7 + j * (13 + 2 * i) + i * 31 + seed) % 97 - 48` in [-48, 48],
    then:
    `(2 * v + 65) / 64` if `v >= 0`, else `(2 * v - 63) / 64`, for `value_res_lambda`
    (so `|lambda| > 1` and the blend is far from one half), `1 + v / 128` for norm
    weights, `v / 64` for the MLP gate and up projections (so the clamp at 2 binds), and
    `v / 256` for everything else. The divisors are powers of two, so the float32
    values are exact on both sides. `main` picks the seed from `SEEDS` whose greedy
    decode has the largest top-1/top-2 margin."""
    weights = {}
    for i, name in enumerate(sorted(shapes)):
        j = torch.arange(math.prod(shapes[name]), dtype=torch.int64)
        v = (j * j * 7 + j * (13 + 2 * i) + i * 31 + seed) % 97 - 48
        if name.endswith("value_res_lambda"):
            values = torch.where(v >= 0, 2 * v + 65, 2 * v - 63).float() / 64
        elif name.endswith("norm.weight"):
            values = v.float() / 128 + 1
        elif name.endswith(("gate_proj.weight", "up_proj.weight")):
            values = v.float() / 64
        else:
            values = v.float() / 256
        weights[name] = values.reshape(shapes[name])
    return weights


def load_reference(path):
    """Import the author's module from its directory, as the package it lives in."""
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent.parent))
    module = importlib.import_module(f"{path.parent.name}.{path.stem}")
    return module


class Reference:
    def __init__(self, module):
        config = module.RicursiveConfig(**{k: v for k, v in CONFIG.items() if k != "model_type"})
        config._attn_implementation = "sdpa"
        self.config = config
        self.model = module.RicursiveForCausalLM(config).float().eval()
        self.shapes = {n: tuple(p.shape) for n, p in self.model.named_parameters()}

    @torch.no_grad()
    def load(self, weights):
        parameters = dict(self.model.named_parameters())
        assert sorted(parameters) == sorted(weights), "weight names differ"
        for name, value in weights.items():
            parameters[name].copy_(value)

    @torch.no_grad()
    def full_logits(self):
        return self.model(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0].float()

    @torch.no_grad()
    def decode(self, step):
        """Prefill all but the last prompt token in chunks of `step`, then decode greedily."""
        cache = DynamicCache(config=self.config)
        offset, last = 0, len(PROMPT) - 1
        while offset < last:
            count = min(step, last - offset)
            self.model(
                input_ids=torch.tensor([PROMPT[offset : offset + count]]),
                past_key_values=cache, use_cache=True,
                cache_position=torch.arange(offset, offset + count))
            offset += count
        token, greedy, logits = PROMPT[last], [], []
        for _ in range(DECODE):
            row = self.model(
                input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True,
                cache_position=torch.arange(offset, offset + 1)).logits[0, -1].float()
            offset += 1
            token = int(row.argmax())
            greedy.append(token)
            logits.append(row)
        return greedy, torch.stack(logits)


def floats(values, indent="        "):
    return [
        indent + ", ".join(f"{value:.9e}" for value in values[offset : offset + 5]) + ","
        for offset in range(0, len(values), 5)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="path to the author's modeling_ricursive.py")
    parser.add_argument("--revision", required=True, help="the model repository commit it comes from")
    args = parser.parse_args()
    digest = hashlib.sha256(Path(args.reference).read_bytes()).hexdigest()

    torch.set_num_threads(4)
    reference = Reference(load_reference(args.reference))

    # Take the seed whose greedy decode is furthest from a near tie.
    def margin_of(logits):
        top = logits.topk(2, dim=-1).values
        return (top[:, 0] - top[:, 1]).min().item()

    margins = {}
    for seed in SEEDS:
        reference.load(formula_weights(reference.shapes, seed))
        margins[seed] = margin_of(reference.decode(PREFILL_STEPS[0])[1])
    seed = max(SEEDS, key=margins.get)
    reference.load(formula_weights(reference.shapes, seed))

    full = reference.full_logits()
    cases = [reference.decode(step) for step in PREFILL_STEPS]

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
    margin = margin_of(cases[0][1])
    assert margin > 1e-2, f"a greedy decision has margin {margin}"

    names = sorted(reference.shapes)
    greedy, logits = cases[0]
    lines = [
        "// Generated by tools/ricursive_parity/ricursive-reference.py; do not edit values by hand.",
        f"// ricursiveAI/Ripple-65B-Preview {args.revision}, modeling_ricursive.py",
        f"// sha256 {digest};",
        f"// PyTorch {torch.__version__.split('+')[0]} CPU FP32, transformers {transformers.__version__}.",
        "",
        "enum RicursiveReferenceValues {",
        '    static let configuration = """',
        *["        " + line for line in json.dumps(CONFIG, indent=2).splitlines()],
        '        """',
        "",
        "    /// The reference model's parameter names, sorted, and their shapes.",
        "    static let parameterNames = [",
        *[f'        "{name}",' for name in names],
        "    ]",
        "    static let parameterShapes = [",
        *[f"        {list(reference.shapes[name])}," for name in names],
        "    ]",
        "",
        f"    static let seed = {seed}",
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
    print(f"seed {seed}, smallest greedy margin {margin:.4g}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
