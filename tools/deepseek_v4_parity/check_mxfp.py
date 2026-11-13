"""Checks mxfp.py against MLX bit for bit: quantize / dequantize of random bfloat16 matrices (and
of the fixture's own weights when given), and that the written safetensors load in MLX.
Needs a python with mlx and numpy:
python3 -B tools/deepseek_v4_parity/check_mxfp.py [Tests/MLXLMTests/Resources/deepseek_v4_tiny]
"""
import os
import sys
import tempfile

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mxfp  # noqa: E402

rng = np.random.default_rng(0)
mats = [rng.standard_normal((64, 128)).astype(np.float32) * s for s in (0.15, 1e-3, 30.0, 1e-30)]
mats.append(np.zeros((32, 64), dtype=np.float32))
edge = rng.standard_normal((32, 64)).astype(np.float32)
edge[0, :32] = [6, 5, 3.5, 2.5, 1.75, 1.25, 0.75, 0.25] * 4
edge[1, :32] = [448, 464, 416, 15, 17, 19, 0.001953125, -0.0029296875] * 4
mats.append(edge)
fixture = sys.argv[1] if len(sys.argv) > 1 else None
if fixture:
    for k, v in mx.load(os.path.join(fixture, "model.safetensors")).items():
        if k.endswith(".scales"):
            continue
        if v.dtype == mx.bfloat16 and v.ndim == 2 and v.shape[1] % 32 == 0:
            mats.append(np.array(v.astype(mx.float32)))
bad = 0
for i, m in enumerate(mats):
    m = mxfp.bf16_round(m)
    for bits, mode in ((4, "mxfp4"), (8, "mxfp8")):
        w, s = mx.quantize(mx.array(m).astype(mx.bfloat16), group_size=32, bits=bits, mode=mode)[:2]
        pw, ps = mxfp.quantize(m, bits)
        same = np.array_equal(np.array(w), pw) and np.array_equal(np.array(s), ps)
        deq = np.array(mx.dequantize(w, s, group_size=32, bits=bits, mode=mode).astype(mx.float32))
        same_d = np.array_equal(deq, mxfp.dequantize(pw, ps, bits))
        if not (same and same_d):
            bad += 1
            print(f"matrix {i} {mode}: quantize equal {same}, dequantize equal {same_d}")
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "t.safetensors")
    t = {"a": mxfp.bf16_round(mats[0]), "b": np.arange(6, dtype=np.uint32), "c": np.ones(3, np.uint8),
         "d": np.arange(4, dtype=np.int32), "e": mats[1]}
    mxfp.save_safetensors(path, t, bf16={"a"})
    back = mx.load(path)
    ok = back["a"].dtype == mx.bfloat16 and np.array_equal(np.array(back["a"].astype(mx.float32)), t["a"])
    ok &= all(np.array_equal(np.array(back[k]), t[k]) for k in "bcde")
    ok &= np.array_equal(np.array(mx.array(mats[0]).astype(mx.bfloat16).astype(mx.float32)), t["a"])
    if not ok:
        bad += 1
        print("safetensors round trip or bfloat16 rounding differs from MLX")
print(f"{len(mats)} matrices x 2 modes: {'all bit-exact' if not bad else f'{bad} mismatches'} (mlx {mx.__version__})")
sys.exit(1 if bad else 0)
