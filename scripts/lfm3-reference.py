#!/usr/bin/env python3
"""Regenerate the small FP32 LFM3 oracle using LiquidAI's original implementation.

Requires torch, transformers==5.22.1, huggingface_hub and safetensors.
Run from the repository root: python scripts/lfm3-reference.py
The downloaded Python code is pinned and runs on the CPU without model weights.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

REVISION = "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
OUTPUT = Path("Tests/MLXLMTests/Resources")


def main():
    torch.set_num_threads(4)
    torch.manual_seed(7)
    with TemporaryDirectory(prefix="lfm3-reference-") as source:
        snapshot_download(
            "LiquidAI/LFM3-1.7B", revision=REVISION,
            allow_patterns=["config.json", "*.py"], local_dir=source,
        )
        config = AutoConfig.from_pretrained(source, trust_remote_code=True)
        config.hidden_size = 8
        config.intermediate_size = 16
        config.num_hidden_layers = 3
        config.num_attention_heads = 2
        config.num_key_value_heads = 1
        config.head_dim = 8
        config.vocab_size = 16
        config.layer_types = ["conv", "full_attention", "sliding_attention"]
        config.sliding_window = 4
        config.conv_bias = True
        config.pad_token_id = 0
        config.bos_token_id = 1
        config.eos_token_id = 2
        config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).float().eval()
    with torch.no_grad():
        for i, parameter in enumerate(model.parameters()):
            values = (torch.arange(parameter.numel()).reshape(parameter.shape) + i * 3) % 17
            parameter.copy_((values.float() - 8) / 32)
            if parameter.ndim == 1 and parameter.numel() == 8:
                parameter.add_(1)
    weights = {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}
    outputs = {}
    handles = []
    for name, module in model.named_modules():
        if name in ["model.embed_tokens", "model.embedding_norm"] or (
            name.startswith("model.layers.") and name.count(".") == 2
        ) or name.endswith((".conv", ".self_attn")) and name.count(".") == 3:
            def hook(_module, _args, result, name=name):
                tensor = result[0] if isinstance(result, tuple) else result
                outputs[name] = tensor.detach().contiguous().clone()
            handles.append(module.register_forward_hook(hook))
    tokens = torch.tensor([(i * 7 + 3) % 16 for i in range(38)]).reshape(2, 19)
    with torch.no_grad():
        outputs["logits"] = model(tokens, use_cache=False).logits
    for handle in handles:
        handle.remove()
    outputs["tokens"] = tokens.to(torch.int32)
    with torch.no_grad():
        cache = None
        logits = []
        for start, end in [(0, 2), (2, 7), (7, 18), (18, 19)]:
            out = model(tokens[:, start:end], past_key_values=cache, use_cache=True)
            logits.append(out.logits)
            cache = out.past_key_values
        outputs["cached_logits"] = torch.cat(logits, dim=1)
    OUTPUT.mkdir(exist_ok=True)
    save_file(weights, OUTPUT / "lfm3-reference-weights.safetensors")
    save_file({k: v.contiguous() for k, v in outputs.items()}, OUTPUT / "lfm3-reference-outputs.safetensors",
              metadata={"source_revision": REVISION, "precision": "float32", "transformers": "5.22.1"})
    config_dict = config.to_dict()
    for key in ["_name_or_path", "auto_map"]:
        config_dict.pop(key, None)
    (OUTPUT / "lfm3-reference-config.json").write_text(json.dumps(config_dict, indent=2) + "\n")
    error = (outputs["cached_logits"] - outputs["logits"]).abs().max().item()
    print(f"Reference cached/uncached maximum absolute error: {error:.9g}")
    print("Outputs:", ", ".join(outputs))


if __name__ == "__main__":
    main()
