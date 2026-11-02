#!/usr/bin/env python3
"""Export FP32 reference outputs for the official LFM3-1.7B MLX 4-bit checkpoint.

Requires torch==2.15.0, transformers==5.22.1, mlx, numpy, huggingface_hub
and safetensors. Download checkpoint revision d0ca73dcfa70ce18733754a492cdc1fb0123262d
separately. This script runs LiquidAI's pinned Python code on the CPU and uses
about 8 GB of memory to dequantize the same weights used by the Swift test.

python scripts/lfm3-checkpoint-reference.py /path/to/checkpoint /tmp/lfm3-reference.safetensors
Then run LFM3CheckpointTests with MLX_LFM3_CHECKPOINT, MLX_LFM3_REFERENCE_FILE
and MLX_ENABLE_TF32=0 set in the test process environment.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

REVISION = "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with (args.checkpoint / "model.safetensors").open("rb") as weights_file:
        digest = hashlib.file_digest(weights_file, "sha256").hexdigest()
    if digest != "1be5bf672013cd91aa62148f77c5eea8d2784a95845d5fe826e08c7b84823815":
        raise ValueError("Expected the official 4-bit checkpoint at the documented revision")
    torch.set_num_threads(4)
    checkpoint_config = json.loads((args.checkpoint / "config.json").read_text())
    quantization = checkpoint_config["quantization"]
    with TemporaryDirectory(prefix="lfm3-checkpoint-reference-") as source:
        snapshot_download("LiquidAI/LFM3-1.7B", revision=REVISION, local_dir=source,
                          allow_patterns=["config.json", "*.py"])
        config = AutoConfig.from_pretrained(source, trust_remote_code=True)
        for key in ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "head_dim", "vocab_size", "conv_L_cache", "conv_bias",
                    "layer_types", "sliding_window", "rope_parameters", "norm_eps",
                    "use_sandwich_norm", "tie_word_embeddings"]:
            if getattr(config, key) != checkpoint_config[key]:
                raise ValueError(f"Checkpoint differs from the reference architecture: {key}")
        config._attn_implementation = "eager"
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.model.rotary_emb = type(model.model.rotary_emb)(config, device="cpu")
    arrays = mx.load(str(args.checkpoint / "model.safetensors"))
    state = {}
    for name in model.state_dict():
        if name == "lm_head.weight" and config.tie_word_embeddings:
            state[name] = state["model.embed_tokens.weight"]
            continue
        key = name if name in arrays else "language_model." + name
        value = arrays[key]
        prefix = key.removesuffix(".weight")
        if prefix + ".scales" in arrays:
            value = mx.dequantize(
                value, arrays[prefix + ".scales"].astype(mx.float32),
                arrays[prefix + ".biases"].astype(mx.float32),
                group_size=quantization["group_size"], bits=quantization["bits"])
        if name.endswith("conv.conv.weight") and value.shape[1] > 1:
            value = value.transpose(0, 2, 1)
        state[name] = torch.from_numpy(np.array(value.astype(mx.float32)))
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    del arrays, state
    gc.collect()
    mx.clear_cache()
    outputs = {}
    for length in [19, 257, 1024]:
        tokens = torch.tensor([(i * 7 + 3) % config.vocab_size for i in range(length)]).reshape(1, length)
        start = time.monotonic()
        with torch.no_grad():
            hidden = model.model(tokens, use_cache=False).last_hidden_state
            logits = model.lm_head(hidden[:, -1:])
        outputs[f"tokens_{length}"] = tokens.to(torch.int32)
        outputs[f"hidden_{length}"] = hidden[:, -1:].contiguous()
        outputs[f"logits_{length}"] = logits.contiguous()
        print(f"{length} tokens: {time.monotonic() - start:.3f}s, top1={logits.argmax(-1).item()}")
    save_file(outputs, str(args.output), metadata={
        "source_revision": REVISION, "precision": "float32 dequantized",
        "checkpoint_revision": "d0ca73dcfa70ce18733754a492cdc1fb0123262d",
    })


if __name__ == "__main__":
    main()
