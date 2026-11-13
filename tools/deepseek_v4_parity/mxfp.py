"""NumPy MXFP4 / MXFP8 quantization in MLX's layout, bfloat16 rounding, and a minimal safetensors
writer, so the fixture can be written without MLX.

Layout (as `mx.quantize(w, group_size=32, bits=4|8, mode="mxfp4"|"mxfp8")`): `weight` uint32
[out, in * bits / 32] with element 0 in the lowest bits (FP4 e2m1 nibbles / FP8 e4m3 bytes),
`scales` uint8 [out, in / 32] holding the e8m0 exponent (scale = 2 ** (s - 127)).
`check_mxfp.py` (next to this file) compares these functions with MLX bit for bit.
"""

import json
import struct

import numpy as np

GROUP = 32
_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
_E4M3 = np.zeros(256)
for _c in range(256):
    _s, _e, _m = _c >> 7, (_c >> 3) & 15, _c & 7
    _v = (_m / 8) * 2.0 ** -6 if _e == 0 else (1 + _m / 8) * 2.0 ** (_e - 7)
    _E4M3[_c] = -_v if _s else _v
_E4M3[0x7F] = _E4M3[0xFF] = np.nan
_E4M3_POS = _E4M3[:0x7F]  # codes 0..126, ascending magnitudes 0 .. 448
_E2M1_POS = _E2M1[:8]


def bf16_round(x):
    """float32 -> float32 values rounded to bfloat16 (nearest even)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def _nearest_code(a, grid):
    """Index of the nearest grid magnitude to each |value| (grid ascending), ties to the even
    code; values past the end saturate to the last code."""
    hi = np.clip(np.searchsorted(grid, a, side="left"), 1, len(grid) - 1)
    lo = hi - 1
    dlo, dhi = a - grid[lo], grid[hi] - a
    code = np.where(dhi < dlo, hi, lo)
    tie = dhi == dlo
    code = np.where(tie, np.where(lo % 2 == 0, lo, hi), code)
    return np.where(a >= grid[-1], len(grid) - 1, code)


def quantize(w, bits):
    """float32 [out, in] (bfloat16 values) -> (uint32 weight, uint8 scales)."""
    w = np.asarray(w, dtype=np.float32)
    out, inp = w.shape
    g = w.reshape(out, inp // GROUP, GROUP).astype(np.float64)
    amax = np.abs(g).max(-1)
    # The shared exponent is log2(amax / largest element) rounded to the nearest integer, as MLX
    # does; elements past the largest value saturate.
    top = np.float32(6.0 if bits == 4 else 448.0)
    ratio = (amax.astype(np.float32) / top).astype(np.float64)
    with np.errstate(divide="ignore"):
        e = np.where(ratio > 0, np.round(np.log2(np.where(ratio > 0, ratio, 1))), -127)
    e = np.clip(e, -127, 127)
    scaled = g / np.exp2(e)[..., None]
    grid = _E2M1_POS if bits == 4 else _E4M3_POS
    code = _nearest_code(np.abs(scaled), grid)
    code = code | np.signbit(scaled) * (8 if bits == 4 else 128)
    code = code.reshape(out, inp).astype(np.uint32)
    per = 32 // bits
    code = code.reshape(out, inp // per, per)
    packed = np.zeros((out, inp // per), dtype=np.uint32)
    for i in range(per):
        packed |= code[..., i] << np.uint32(i * bits)
    return packed, (e + 127).astype(np.uint8)


def dequantize(weight, scales, bits):
    u = np.asarray(weight, dtype=np.uint32)
    per = 32 // bits
    mask = (1 << bits) - 1
    codes = np.stack([(u >> np.uint32(i * bits)) & mask for i in range(per)], -1).reshape(u.shape[0], -1)
    vals = (_E2M1 if bits == 4 else _E4M3)[codes]
    s = np.exp2(np.asarray(scales).astype(np.float64) - 127)
    return (vals * np.repeat(s, GROUP, axis=1)).astype(np.float32)


_DTYPES = {np.dtype(np.float32): "F32", np.dtype(np.uint32): "U32", np.dtype(np.uint8): "U8",
           np.dtype(np.int32): "I32"}


def save_safetensors(path, tensors, bf16=()):
    """Write `tensors` (name -> numpy array) in key order; names in `bf16` are float32 arrays of
    bfloat16 values stored as BF16."""
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name])
        if name in bf16:
            data = (bf16_round(a).view(np.uint32) >> 16).astype("<u2").tobytes()
            dtype = "BF16"
        else:
            dtype = _DTYPES[a.dtype]
            data = a.astype(a.dtype.newbyteorder("<")).tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
