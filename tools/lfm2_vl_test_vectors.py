#!/usr/bin/env python3
"""Prints the expected values of the LFM2-VL unit tests, computed with Hugging Face transformers.

The tests are ImagePixelsTests.swift, LFM2VLImageProcessingTests.swift, LFM2VLProcessorTests.swift,
LFM2VLTests.swift and Siglip2NaFlexTests.swift.
Each section names the test it feeds. No model weights are needed.

    pip install transformers torch torchvision pillow
    python tools/lfm2_vl_test_vectors.py

The checked-in values come from transformers 5.19.0, torch 2.14.1, torchvision 0.29.1 and
Pillow 12.3.0 on macOS arm64.
"""

import base64
import hashlib
import io
import re
import struct
import zlib
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvF
from transformers.image_utils import load_image
from transformers.models.lfm2_vl.image_processing_lfm2_vl import Lfm2VlImageProcessor
from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlMultiModalProjector
from transformers.models.lfm2_vl.processing_lfm2_vl import Lfm2VlProcessor
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings


def section(title):
    print(f"\n// MARK: {title}")


def pattern(w, h):
    """The test pattern: (x*31 + y*17 + c*59 + ((x*y) % 7)*23) % 256."""
    y, x, c = np.meshgrid(np.arange(h), np.arange(w), np.arange(3), indexing="ij")
    return ((x * 31 + y * 17 + c * 59 + ((x * y) % 7) * 23) % 256).astype(np.uint8)


def chw(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).contiguous()


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).astype("<f4").tobytes()).hexdigest()


def process(images, **kwargs):
    p = Lfm2VlImageProcessor(**kwargs)
    return p, p(images=[[chw(i) for i in images]], return_row_col_info=True)


def token_run(p, out, i):
    """`Lfm2VlProcessor._build_image_tokens` for image i, as (image tokens, run length)."""
    me = SimpleNamespace(
        image_processor=p, image_token="<image>", image_start_token="<|image_start|>",
        image_end_token="<|image_end|>", image_thumbnail_token="<|img_thumbnail|>")
    for name in ["_build_image_tokens", "_get_image_num_tokens", "_compute_tokens_per_tile",
                 "_compute_tokens_for_image"]:
        setattr(me, name, getattr(Lfm2VlProcessor, name).__get__(me))
    run = Lfm2VlProcessor.replace_image_token(me, out, i, use_image_special_tokens=True)
    return run.count("<image>"), len(re.findall(r"<image>|<\|[^|]+\|>", run))


def layout(p, w, h, lo, hi):
    """(tiled, columns, rows, thumbnail width, thumbnail height) from the processor's helpers."""
    tw, th = p.smart_resize(h, w, p.downsample_factor, p.min_image_tokens, p.max_image_tokens,
                            p.encoder_patch_size)
    large = p._is_image_too_large(h, w, p.max_image_tokens, p.encoder_patch_size,
                                  p.downsample_factor, p.max_pixels_tolerance)
    tiled = large and not lo == hi == 1
    cols, rows = p._get_grid_layout(h, w, lo, hi, p.tile_size)[:2] if tiled else (1, 1)
    return f"(({w}, {h}), {str(tiled).lower()}, {cols}, {rows}, {tw}, {th}),"


section("LFM2VLImageLayoutTests.defaultCases")
p = Lfm2VlImageProcessor()
for w, h in [(640, 480), (64, 48), (300, 1400), (1, 1), (2000, 1200), (3000, 260), (1000, 1000),
             (1100, 1100), (48, 1040), (1250, 1000), (2000, 1000), (1920, 1080), (8000, 20),
             (48, 80), (1536, 1024), (400, 600), (1024, 768), (1182, 886), (886, 1182),
             (686, 1526), (2560, 1440), (960, 540)]:
    print(layout(p, w, h, 2, 10))

section("LFM2VLImageLayoutTests.layoutFollowsOtherSettings (128/32 tokens, tolerance 1.5)")
p = Lfm2VlImageProcessor(max_image_tokens=128, min_image_tokens=32, max_pixels_tolerance=1.5)
for w, h in [(640, 480), (1000, 1000), (700, 300), (300, 700), (3000, 260), (512, 512), (1100, 1100)]:
    print(layout(p, w, h, 2, 10))

section("LFM2VLImageLayoutTests.targetRatiosSortByTileCountThenColumns: _target_ratios(min, max)")
for lo, hi in [(2, 10), (1, 4), (6, 6), (1, 1)]:
    grids = sorted(Lfm2VlImageProcessor()._target_ratios(lo, hi), key=lambda r: (r[0] * r[1], r))
    ratios = " ".join(f"{w}x{h}" for w, h in grids)
    print(f'(({lo}, {hi}), "{ratios}"),')

section("ImagePixelsResizeTests.torchCases: tvF.resize(uint8, antialias=True)")
modes = {2: (InterpolationMode.BILINEAR, "bilinear"), 3: (InterpolationMode.BICUBIC, "bicubic")}
for w, h, ow, oh, m in [(37, 53, 96, 64, 3), (37, 53, 15, 20, 3), (64, 48, 23, 17, 3),
                        (50, 40, 50, 13, 3), (50, 40, 7, 40, 3), (3, 500, 2, 37, 3),
                        (3, 500, 3, 41, 3), (300, 2, 31, 3, 3), (1, 1, 5, 3, 3), (37, 53, 96, 64, 2),
                        (37, 53, 15, 20, 2), (3, 500, 2, 37, 2), (1024, 768, 1536, 1024, 3),
                        (768, 1024, 416, 576, 2)]:
    out = tvF.resize(chw(pattern(w, h)), [oh, ow], interpolation=modes[m][0], antialias=True)
    b = out.permute(1, 2, 0).reshape(-1).numpy().astype(np.int64)
    print(f"({w}, {h}, {ow}, {oh}, .{modes[m][1]}, {int(b.sum()):_}, {b[:6].tolist()}, "
          f"{b[-3:].tolist()}),")

section("ImagePixelsResizeTests.orientationMatchesExifTranspose: ImageOps.exif_transpose")
source = Image.fromarray(np.arange(18, dtype=np.uint8).reshape(2, 3, 3), "RGB")
for orientation in range(1, 9):
    exif = Image.Exif()
    exif[0x0112] = orientation
    buffer = io.BytesIO()
    source.save(buffer, "PNG", exif=exif)
    turned = load_image(base64.b64encode(buffer.getvalue()).decode())
    print(f"{orientation}: ({turned.width}, {turned.height}, {np.asarray(turned).reshape(-1).tolist()}),")


def png(width, height, depth, color_type, row, *chunks):
    """A one-row PNG from raw scanline bytes, with extra (type, data) chunks before IDAT."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    header = struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0)
    body = b"".join(chunk(kind, data) for kind, data in chunks)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + body
            + chunk(b"IDAT", zlib.compress(b"\0" + row)) + chunk(b"IEND", b""))


def fixture(name, data):
    pixels = np.asarray(load_image(base64.b64encode(data).decode())).reshape(-1)
    print(f'{name}: "{base64.b64encode(data).decode()}"')
    print(f"{name} load_image bytes: {pixels.tolist()}")


section("ImagePixelsDecodingTests: encoded fixtures and their load_image bytes")
exif = Image.Exif()
exif[0x0112] = 6
buffer = io.BytesIO()
source.save(buffer, "PNG", exif=exif)
fixture("RGB PNG, EXIF orientation 6", buffer.getvalue())
fixture("RGBA PNG", png(2, 1, 8, 6, bytes([10, 20, 30, 0, 200, 100, 50, 128])))
fixture("gray PNG", png(3, 1, 8, 0, bytes([0, 77, 255])))
fixture("2-bit palette PNG with tRNS",
        png(4, 1, 2, 3, bytes([0b00011011]),
            (b"PLTE", bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 8, 7])),
            (b"tRNS", bytes([0, 255, 128, 0]))))
fixture("16-bit gray and alpha PNG", png(2, 1, 16, 4, struct.pack(">4H", 1000, 0, 0xC8FF, 0xFFFF)))
blocks = np.zeros((8, 12, 3), dtype=np.uint8)
for i, colour in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
                            (255, 0, 255)]):
    blocks[(i // 3) * 4:(i // 3) * 4 + 4, (i % 3) * 4:(i % 3) * 4 + 4] = colour
buffer = io.BytesIO()
Image.fromarray(blocks, "RGB").save(buffer, "JPEG", quality=90, subsampling=2, exif=exif)
data = buffer.getvalue()
pixels = np.asarray(load_image(base64.b64encode(data).decode()))
print(f'4:2:0 JPEG, EXIF orientation 6: "{base64.b64encode(data).decode()}"')
print(f"its load_image size {pixels.shape[1]}x{pixels.shape[0]}, bytes (base64): "
      f'"{base64.b64encode(pixels.tobytes()).decode()}"')

section("LFM2VLImageSettingsTests")
print("Lfm2VlImageProcessor().resample:", int(Lfm2VlImageProcessor().resample))
print("max_num_patches with tile 1024, splitting on and off:",
      Lfm2VlImageProcessor(tile_size=1024).max_num_patches,
      Lfm2VlImageProcessor(do_image_splitting=False, tile_size=1024).max_num_patches)


def normalization_bits(mean, std):
    values = torch.tensor([0, 1, 77, 127, 128, 200, 254, 255], dtype=torch.uint8)
    values = values.reshape(1, 1, 8).expand(3, 1, 8).contiguous()
    p = Lfm2VlImageProcessor(image_mean=mean, image_std=std)
    r = p.rescale_and_normalize(values[None], True, 1 / 255, True, tuple(mean), tuple(std))[0]
    return [", ".join(f"0x{int(v):08x}" for v in r[c, 0].numpy().view(np.uint32)) for c in range(3)]


print("normalization, mean = std = 0.5:", normalization_bits([0.5] * 3, [0.5] * 3)[0])

section("LFM2VLImageProcessingTests.normalizationMatchesTransformers: the 0.5 row above, and CLIP")
for c, row in enumerate(normalization_bits([0.48145466, 0.4578275, 0.40821073],
                                           [0.26862954, 0.26130258, 0.27577711])):
    print(f"normalization, CLIP mean and std, channel {c}: {row}")

section("LFM2VLImageProcessingTests.assetCases: ((w, h), (cols, rows), thumbnail, tokens, run length)")
for w, h in [(640, 480), (1024, 768), (1182, 886), (886, 1182), (400, 600), (686, 1526),
             (1600, 1200), (720, 1600), (3024, 4032), (1000, 1000), (64, 48), (1, 1), (3000, 260)]:
    p, out = process([np.zeros((h, w, 3), np.uint8)])
    tokens, length = token_run(p, out, 0)
    print(f"(({w}, {h}), ({out['image_cols'][0]}, {out['image_rows'][0]}), "
          f"{out['spatial_shapes'][-1].tolist()}, {tokens}, {length}),")

section("LFM2VLImageProcessingTests.tilingKeysWorkAsInTransformers: "
        "(settings, (w, h), spatial shapes, tokens, run length)")
variants = {
    '"do_image_splitting": false': dict(do_image_splitting=False),
    '"use_thumbnail": false': dict(use_thumbnail=False),
    '"max_tiles": 4': dict(max_tiles=4),
    '"min_tiles": 1, "max_tiles": 4': dict(min_tiles=1, max_tiles=4),
    '"min_tiles": 6, "max_tiles": 6': dict(min_tiles=6, max_tiles=6),
}
for key, kwargs in variants.items():
    for w, h in [(1024, 768), (3000, 260), (2000, 2000), (724, 724), (64, 48)]:
        p, out = process([np.zeros((h, w, 3), np.uint8)], **kwargs)
        tokens, length = token_run(p, out, 0)
        print(f"(#\"{{{key}}}\"#, ({w}, {h}), {out['spatial_shapes'].tolist()}, {tokens}, {length}),")
_, out = process([pattern(1000, 1000)], use_thumbnail=False)
print("pixel_values sha256, use_thumbnail false, 1000x1000:", sha(out["pixel_values"].numpy()))
_, out = process([pattern(1024, 768)], do_image_splitting=False)
print("pixel_values sha256, do_image_splitting false, 1024x768:", sha(out["pixel_values"].numpy()))

section("LFM2VLImageProcessingTests.pixelsMatchTransformersBitForBit: sha256 of pixel_values")
for r, name in [(3, "bicubic"), (2, "bilinear")]:
    for w, h in [(64, 48), (400, 600), (1024, 768)]:
        _, out = process([pattern(w, h)], resample=r)
        print(f'(.{name}, ({w}, {h}), "{sha(out["pixel_values"].numpy())}"),')

section("LFM2VLProcessorPrepareTests.imagesExpandInPromptOrder")
_, out = process([pattern(64, 48), pattern(1000, 1000), pattern(400, 600)])
print("shape", list(out["pixel_values"].shape), "sha256", sha(out["pixel_values"].numpy()))

section("LFM2VLProcessorPrepareTests, the resize tests: do_image_splitting=False for one call")


def call(p, images, **kwargs):
    """Spatial shapes, image tokens per image, and the shape and sha256 of pixel_values."""
    out = p(images=[[chw(i) for i in images]], return_row_col_info=True, **kwargs)
    tokens = [token_run(p, out, i)[0] for i in range(len(images))]
    return (f"shapes {out['spatial_shapes'].tolist()} tokens {tokens} "
            f"pixel_values {list(out['pixel_values'].shape)} sha256 {sha(out['pixel_values'].numpy())}")


p = Lfm2VlImageProcessor()
print("1024x768, default:", call(p, [pattern(1024, 768)]))
print("1024x768, splitting off:", call(p, [pattern(1024, 768)], do_image_splitting=False))
print("64x48, 1000x1000 and 400x600, splitting off:",
      call(p, [pattern(64, 48), pattern(1000, 1000), pattern(400, 600)], do_image_splitting=False))
p = Lfm2VlImageProcessor(max_image_tokens=128, resample=3)
print("1024x768, 128 tokens, bicubic, splitting off:",
      call(p, [pattern(1024, 768)], do_image_splitting=False))

section("Siglip2NaFlexTests.bilinearWeights: F.interpolate weights, (in, out, row) -> nonzero taps")
for n_in, n_out, row in [(16, 14, 0), (16, 14, 1), (16, 2, 0), (16, 32, 0), (16, 32, 1),
                         (16, 32, 31), (16, 26, 1)]:
    eye = torch.eye(n_in, dtype=torch.float32).reshape(n_in, 1, 1, n_in)
    weights = F.interpolate(eye, size=(1, n_out), mode="bilinear", align_corners=False,
                            antialias=True).reshape(n_in, n_out).T[row]
    taps = {j: float(v) for j, v in enumerate(weights.tolist()) if v != 0}
    print(f"({n_in}, {n_out}, {row}): {taps}")

section("Siglip2NaFlexTests.resizedPositionEmbeddings: resize_positional_embeddings")
table = (torch.arange(16 * 16 * 4, dtype=torch.float32) * 37 % 101 / 8).reshape(16, 16, 4)
shapes = torch.tensor([[14, 20], [2, 40], [32, 32], [16, 16]])
out = Siglip2VisionEmbeddings.resize_positional_embeddings(table, shapes, 1024)
for row, position in [(0, 0), (0, 21), (0, 279), (1, 0), (1, 57), (2, 33)]:
    print(f"({row}, {position}, {out[row, position].tolist()}),")
print("sums:", float(out[0, :280].double().sum()), float(out[1, :80].double().sum()),
      float(out[2].double().sum()))

section("LFM2VLTests.pixelUnshuffle: pixel_unshuffle of the zero-padded 1x3x5x1 input")
x = F.pad(torch.arange(15, dtype=torch.float32).reshape(1, 3, 5, 1), (0, 0, 0, 1, 0, 1))
y = Lfm2VlMultiModalProjector.pixel_unshuffle(SimpleNamespace(factor=2), x)
print("shape", list(y.shape), "[0, 0, 0]", y[0, 0, 0].tolist(), "[0, 1, 2]", y[0, 1, 2].tolist())
