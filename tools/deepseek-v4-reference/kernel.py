# FP32 CPU stand-in for DeepSeek-V4's inference/kernel.py (tilelang, CUDA), written from it
# (MIT License, Copyright (c) 2023 DeepSeek). Activation quantization is identity.

import torch


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    if not inplace:
        raise NotImplementedError("FP8 weights are not supported by the CPU shim")
    return x


def fp4_act_quant(x, block_size=32, inplace=False):
    if not inplace:
        raise NotImplementedError("FP4 weights are not supported by the CPU shim")
    return x


def fp8_gemm(*args, **kwargs):
    raise NotImplementedError("FP8 weights are not supported by the CPU shim")


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError("FP4 weights are not supported by the CPU shim")


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Attention over the kv rows in topk_idxs (-1 = unused); the per-head sink logit joins
    only the softmax denominator."""
    b, m, h, d = q.shape
    idx = topk_idxs.to(torch.int64)
    valid = idx != -1
    gathered = torch.gather(
        kv.float()[:, None].expand(b, m, kv.size(1), d), 2,
        idx.clamp_min(0)[..., None].expand(b, m, idx.size(-1), d)) * valid[..., None]
    s = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered) * softmax_scale
    s = torch.where(valid[:, :, None, :], s, torch.tensor(float("-inf")))
    top = s.amax(dim=-1, keepdim=True)
    p = torch.exp(s - top)
    denom = p.sum(-1) + torch.exp(attn_sink.float()[None, None, :] - top[..., 0])
    return (torch.einsum("bmhk,bmkd->bmhd", p, gathered) / denom[..., None]).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """pre = sigmoid + eps, post = 2 sigmoid, comb = row softmax + eps, then alternating
    column and row normalizations (sinkhorn_iters column steps)."""
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
