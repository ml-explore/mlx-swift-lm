#!/usr/bin/env python3
"""Export LFM3-VL image inputs and last-token logits for the opt-in Swift test.

Requires mlx-vlm. Run with CHECKPOINT_DIRECTORY OUTPUT.safetensors and
MLX_ENABLE_TF32=0. Floating parameters are promoted to float32 for parity.
Set MLX_LFM3_VL_CHECKPOINT and MLX_LFM3_VL_REFERENCE in the Swift test process.
The integration test also checks generation at the checkpoint's native dtype.
"""
import sys
import mlx.core as mx
from PIL import Image
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load(sys.argv[1], trust_remote_code=True)
model.set_dtype(mx.float32)
prompt = apply_chat_template(
    processor, model.config,
    "What is the color of this image? Answer with one word.",
    num_images=1, enable_thinking=False,
)
inputs = processor(text=[prompt], images=[Image.new("RGB", (256, 256), (255, 0, 0))], return_tensors="mlx")
output = model(**inputs)
mx.eval(output.logits)
mx.save_safetensors(sys.argv[2], {
    "tokens": inputs["input_ids"], "pixels": inputs["pixel_values"],
    "grid": inputs["image_grid_thw"], "logits": output.logits[:, -1:, :].astype(mx.float32),
})
