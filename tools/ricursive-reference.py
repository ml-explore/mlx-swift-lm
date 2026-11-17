#!/usr/bin/env python3
"""Generate Swift FP32 reference values from ricursiveAI's original implementation.

Requires torch, transformers==5.22.2 and huggingface_hub.
Run from the repository root: python tools/ricursive-reference.py
Only modeling_ricursive.py and configuration_ricursive.py are downloaded, at a pinned
revision with checked sha256 digests, and run on the CPU without model weights.
Weights come from `formula_weights` in sorted parameter-name order, as in RicursiveTests.
"""

import hashlib
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers.cache_utils import DynamicCache

REPOSITORY = "ricursiveAI/Ripple-65B-Preview"
REVISION = "b172323ffd63e62de9ce1ba3830dfffe493c7671"
SHA256 = {
    "modeling_ricursive.py": "77161004d1a8d2493327e055c71e87de6ee560ed2ca255d6598bba266998ed9a",
    "configuration_ricursive.py": "002cb3915be9685eb962d617623bec14d45b5d6b5d46e689fac98a3b38cb6cf6",
}
OUTPUT = Path(__file__).resolve().parents[1] / "Tests/MLXLMTests/RicursiveReferenceValues.swift"
CONFIG = {
    "model_type": "ricursive", "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
    "num_hidden_layers": 5, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 16, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "sliding_window": 8,
    "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
    "use_gated_attention": True, "gate_full_precision": True, "use_valres": True,
    "ffn_clamp_threshold": 2.0, "embed_scale": math.sqrt(32), "hidden_act": "silu",
    "tie_word_embeddings": False,
}
PROMPT = [(i * 37 + 11) % 64 for i in range(24)]
DECODE = 12
PREFILL_STEPS = [1, 3, 8, 23]
# Chosen among 0..31 for the largest top-1/top-2 margin of the greedy decode.
SEED = 4


def formula_weights(shapes, seed):
    """Integers per parameter `i` (sorted names) and element `j`, then power-of-two
    divisors, so the float32 values are exact in Python and Swift. The MLP gate and up
    projections are larger so the clamp binds, and `|value_res_lambda| > 1`."""
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


def load_reference(directory):
    """Download the pinned files, check their digests, and import them as a package from
    `directory` alone."""
    package = Path(directory) / "ricursive_reference"
    snapshot_download(REPOSITORY, revision=REVISION, allow_patterns=list(SHA256), local_dir=package)
    for name, digest in SHA256.items():
        actual = hashlib.sha256((package / name).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"{name}: sha256 {actual}, expected {digest}")
    (package / "__init__.py").write_text("")
    spec = importlib.util.spec_from_file_location(
        package.name, package / "__init__.py", submodule_search_locations=[str(package)])
    sys.modules[package.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[package.name])
    return importlib.import_module(f"{package.name}.modeling_ricursive")


def decode(model, config, step):
    """Prefill all but the last prompt token in chunks of `step`, then decode greedily."""
    cache = DynamicCache(config=config)
    offset, last = 0, len(PROMPT) - 1
    while offset < last:
        count = min(step, last - offset)
        model(input_ids=torch.tensor([PROMPT[offset : offset + count]]), past_key_values=cache,
              use_cache=True, cache_position=torch.arange(offset, offset + count))
        offset += count
    token, greedy, logits = PROMPT[last], [], []
    for _ in range(DECODE):
        row = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(offset, offset + 1)).logits[0, -1].float()
        offset += 1
        token = int(row.argmax())
        greedy.append(token)
        logits.append(row)
    return greedy, torch.stack(logits)


def floats(values):
    return ["        " + ", ".join(f"{value:.9e}" for value in values[offset : offset + 5]) + ","
            for offset in range(0, len(values), 5)]


def main():
    torch.set_num_threads(4)
    with TemporaryDirectory(prefix="ricursive-reference-") as directory:
        module = load_reference(directory)
        config = module.RicursiveConfig(**{k: v for k, v in CONFIG.items() if k != "model_type"})
        config._attn_implementation = "sdpa"
        model = module.RicursiveForCausalLM(config).float().eval()
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    with torch.no_grad():
        parameters = dict(model.named_parameters())
        for name, value in formula_weights(shapes, SEED).items():
            parameters[name].copy_(value)
        full = model(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0].float()
        cases = [decode(model, config, step) for step in PREFILL_STEPS]

    # Every chunking must give the same decode; RicursiveTests compares each with the first.
    greedy, logits = cases[0]
    chunking = max((other - logits).abs().max().item() for _, other in cases[1:])
    assert all(other == greedy for other, _ in cases[1:]), "chunking changed the greedy tokens"
    assert chunking < 1e-4, f"chunking changed the logits by {chunking}"
    cached = (logits[0] - full[-1]).abs().max().item()
    assert cached < 1e-4, f"cached and uncached logits differ by {cached}"
    top = logits.topk(2, dim=-1).values
    margin = (top[:, 0] - top[:, 1]).min().item()
    assert margin > 1e-2, f"a greedy decision has margin {margin}"

    names = sorted(shapes)
    lines = [
        "// Generated by tools/ricursive-reference.py; do not edit values by hand.",
        f"// {REPOSITORY} {REVISION}; sha256 checked:",
        *[f"// {name} {digest}" for name, digest in SHA256.items()],
        f"// PyTorch {torch.__version__.split('+')[0]} CPU FP32, transformers {transformers.__version__}.",
        "", "enum RicursiveReferenceValues {", '    static let configuration = """',
        *["        " + line for line in json.dumps(CONFIG, indent=2).splitlines()],
        '        """', "",
        "    /// The reference model's parameter names, sorted, and their shapes.",
        "    static let parameterNames = [", *[f'        "{name}",' for name in names], "    ]",
        "    static let parameterShapes = [", *[f"        {list(shapes[name])}," for name in names],
        "    ]", "",
        f"    static let seed = {SEED}",
        "    static let prompt = [", "        " + ", ".join(map(str, PROMPT)) + ",", "    ]",
        f"    static let prefillSteps = {PREFILL_STEPS}",
        f"    static let greedy = {greedy}", "",
        f"    /// Uncached logits for every prompt position, `[{len(PROMPT)}, vocab]` row-major.",
        "    static let fullLogits: [Float] = [", *floats(full.flatten().tolist()), "    ]", "",
        f"    /// The logits each greedy token was chosen from, `[{DECODE}, vocab]` row-major.",
        "    static let decodeLogits: [Float] = [", *floats(logits.flatten().tolist()), "    ]",
        "}", "",
    ]
    OUTPUT.write_text("\n".join(lines))
    print(f"greedy {greedy}; chunking max |dlogit| {chunking:.3g}, cached vs uncached {cached:.3g}")
    print(f"smallest greedy margin {margin:.4g}; wrote {OUTPUT}")


if __name__ == "__main__":
    main()
