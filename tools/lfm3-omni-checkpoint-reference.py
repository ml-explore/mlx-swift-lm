#!/usr/bin/env python3
"""Generate FP32 references from the official, locally downloaded Omni checkpoint.

Requires torch, transformers and safetensors. The checkpoint's Python code is
executed: use a trusted snapshot, such as LiquidAI/LFM3-Omni-2B at
21d75123f86c1f7ffb4bef1ea202710c4e7ebe67.

python tools/lfm3-omni-checkpoint-reference.py /path/to/checkpoint /tmp/omni-reference --component codec
"""

import argparse
import pathlib
import sys
import types

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--component", choices=["codec", "audio", "depth"], required=True)
    args = parser.parse_args()
    package = types.ModuleType("omni_reference")
    package.__path__ = [str(args.checkpoint.resolve())]
    sys.modules[package.__name__] = package

    def load_component(module, prefix):
        with safe_open(args.checkpoint / "model.safetensors", framework="pt") as weights:
            state = {key.removeprefix(prefix): weights.get_tensor(key).float()
                     for key in weights.keys() if key.startswith(prefix)}
        module.load_state_dict(state, strict=True)
        return module.float().eval()

    with torch.no_grad():
        if args.component == "codec":
            from omni_reference.configuration_liquid_codec import LiquidCodecConfig
            from omni_reference.modeling_liquid_codec import LiquidCodecModel
            directory = args.checkpoint / "audio_tokenizer"
            config = LiquidCodecConfig.from_pretrained(directory)
            model = LiquidCodecModel.from_pretrained(
                directory, config=config, dtype=torch.float32, attn_implementation="eager").eval()
            t = torch.arange(15360, dtype=torch.float32) / 24000
            samples = (0.1 * torch.sin(2 * torch.pi * 220 * t)
                       + 0.03 * torch.cos(2 * torch.pi * 701 * t))[None]
            codes = model.encode(samples).audio_codes
            values = {"samples": samples, "codes": codes, "audio": model.decode(codes).audio_values}
        else:
            from omni_reference.configuration_lfm3_omni import Lfm3OmniConfig
            from omni_reference.modeling_lfm3_omni import (
                Lfm3OmniEncoder, Lfm3OmniProjector, Lfm3OmniDepthDecoder)
            config = Lfm3OmniConfig.from_pretrained(args.checkpoint, attn_implementation="eager")
            config.audio_encoder_config.text_config._attn_implementation = "eager"
            config.depth_config._attn_implementation = "eager"
            if args.component == "audio":
                model = load_component(Lfm3OmniEncoder(config.audio_encoder_config), "model.audio_encoder.")
                projector = load_component(Lfm3OmniProjector(config), "model.audio_projector.")
                # Both padding boundaries and the encoder's sliding window are exercised.
                features = torch.sin(torch.arange(2 * 128 * 337, dtype=torch.float32) * 0.013).reshape(2, 128, 337)
                lengths = torch.tensor([337, 17])
                hidden, counts = model(features, lengths)
                rows = torch.cat([hidden[i, :count] for i, count in enumerate(counts)])
                values = {"features": features, "lengths": lengths, "encoded": rows, "projected": projector(rows)}
            else:
                model = load_component(Lfm3OmniDepthDecoder(config), "model.depth_decoder.")
                hidden = torch.sin(torch.arange(2 * config.text_config.hidden_size, dtype=torch.float32) * 0.017).reshape(2, -1)
                codes = (torch.arange(2 * config.num_codebooks) * 97 % config.codebook_size).reshape(2, -1)
                masked = torch.arange(config.num_codebooks)[None].expand(2, -1) % 3 != 0
                values = {"hidden": hidden, "codes": codes, "masked": masked, "logits": model(hidden, codes, masked)}
        save_file({name: value.contiguous() for name, value in values.items()}, args.output)
        print({name: list(value.shape) for name, value in values.items()})


if __name__ == "__main__":
    main()
