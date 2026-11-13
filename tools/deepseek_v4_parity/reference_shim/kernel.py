# CPU re-implementation, in float32 PyTorch, of the kernels DeepSeek-V4's reference
# `inference/model.py` imports from its `inference/kernel.py` (tilelang, CUDA), so that the
# unmodified model.py runs on a CPU. Written from DeepSeek's kernel.py (MIT License,
# Copyright (c) 2023 DeepSeek). Activation quantization is left out (identity): act_quant and
# fp4_act_quant in place return their input unchanged, and the FP8/FP4 GEMMs, which the fixture
# never reaches (its Linear weights are float), are not provided.
#
# Differences from kernel.py besides the identity quantization: float32 throughout (the kernels
# keep bfloat16 inputs and outputs, and sparse_attn rounds its probabilities to bfloat16 before
# the value product), and PERTURB, a hook the fixture generator uses to replace the products
# inside sparse_attn by re-rounded float64 products when it measures selection margins.

import torch

PERTURB = None


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block FP8 quantize-dequantize in kernel.py; identity here."""
    if not inplace:
        raise NotImplementedError("FP8 weights are not supported by the CPU shim")
    return x


def fp4_act_quant(x, block_size=32, inplace=False):
    """Block FP4 quantize-dequantize in kernel.py; identity here."""
    if not inplace:
        raise NotImplementedError("FP4 weights are not supported by the CPU shim")
    return x


def fp8_gemm(*args, **kwargs):
    raise NotImplementedError("FP8 weights are not supported by the CPU shim")


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError("FP4 weights are not supported by the CPU shim")


def _product(equation, a, b):
    if PERTURB is None:
        return torch.einsum(equation, a, b)
    return PERTURB(torch.einsum(equation, a.double(), b.double()))


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.py's sparse_attn: per query, attention over the kv rows named by topk_idxs (-1 =
    unused) plus a per-head sink logit that joins the softmax denominator only:
    o = sum_j p_j kv_j / (sum_j p_j + exp(sink - m)), p_j = exp(s_j - m), s_j = (q . kv_j) * scale,
    m the largest s_j."""
    b, m, h, d = q.shape
    idx = topk_idxs.to(torch.int64)
    valid = idx != -1
    gathered = torch.gather(
        kv.float()[:, None].expand(b, m, kv.size(1), d), 2,
        idx.clamp_min(0)[..., None].expand(b, m, idx.size(-1), d)) * valid[..., None]
    s = _product("bmhd,bmkd->bmhk", q.float(), gathered) * softmax_scale
    s = torch.where(valid[:, :, None, :], s, torch.tensor(float("-inf")))
    top = s.amax(dim=-1, keepdim=True)
    p = torch.exp(s - top)
    denom = p.sum(-1) + torch.exp(attn_sink.float()[None, None, :] - top[..., 0])
    return (_product("bmhk,bmkd->bmhd", p, gathered) / denom[..., None]).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """kernel.py's hc_split_sinkhorn: pre = sigmoid + eps, post = 2 sigmoid, comb = row softmax
    + eps, then column / row / column ... normalizations (sinkhorn_iters column steps)."""
    hc = hc_mult
    mixes, hc_scale, hc_base = mixes.float(), hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
