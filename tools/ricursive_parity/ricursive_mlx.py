"""Python port of the Ricursive model for mlx-lm's generation loop.

mlx-lm has no ricursive model, so the "mlx-lm" baseline and greedy oracle in the PR use
this file: the same forward as the Swift `RicursiveModel`, run through mlx-lm's
`generate_step`. Load it with `mlx_lm.utils.load_model(..., model_config={"model_file":
None}, get_model_classes=lambda c: (Model, ModelArgs))`, so no model-repository code runs.

RMSNorm defaults to the reference form (float32 variance, then `weight` in the input
precision). Set RICURSIVE_MLX_FAST_NORM=1 to use the fused `nn.RMSNorm`, as the Swift port
does; the throughput comparison uses that setting.
"""

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache

SLIDING = "sliding_attention"


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int = 100352
    hidden_size: int = 3072
    intermediate_size: int = 24576
    num_hidden_layers: int = 50
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    sliding_window: int = 1024
    layer_types: Optional[List[str]] = None
    use_gated_attention: bool = True
    gate_full_precision: bool = True
    use_valres: bool = False
    ffn_clamp_threshold: Optional[float] = None
    embed_scale: Optional[float] = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 5 == 4 else SLIDING for i in range(self.num_hidden_layers)
            ]


if os.environ.get("RICURSIVE_MLX_FAST_NORM") == "1":

    class RMSNorm(nn.RMSNorm):
        def __init__(self, dims, eps):
            super().__init__(dims, eps=eps)

else:

    class RMSNorm(nn.Module):
        def __init__(self, dims, eps):
            super().__init__()
            self.weight = mx.ones((dims,))
            self.eps = eps

        def __call__(self, x):
            value = x.astype(mx.float32)
            normed = value * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.eps)
            return self.weight * normed.astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, args, layer):
        super().__init__()
        self.args = args
        self.sliding = args.layer_types[layer] == SLIDING
        heads, kv, head = args.num_attention_heads, args.num_key_value_heads, args.head_dim
        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, heads * head, bias=False)
        self.k_proj = nn.Linear(dim, kv * head, bias=False)
        self.v_proj = nn.Linear(dim, kv * head, bias=False)
        self.o_proj = nn.Linear(heads * head, dim, bias=False)
        if args.use_gated_attention:
            self.g_proj = nn.Linear(dim, heads * head, bias=False)
        self.q_norm = RMSNorm(head, args.rms_norm_eps)
        self.k_norm = RMSNorm(head, args.rms_norm_eps)
        if args.use_valres:
            self.value_res_lambda = mx.zeros((1,))
        self._inverse = 1.0 / (
            args.rope_theta ** (mx.arange(0, head, 2, dtype=mx.float32) / head)
        )

    def _rope(self, x, offset):
        positions = mx.arange(offset, offset + x.shape[2]).astype(mx.float32)
        frequencies = positions[:, None] * self._inverse[None, :]
        angles = mx.concatenate([frequencies, frequencies], axis=-1)
        cosine = mx.cos(angles).astype(x.dtype).astype(mx.float32)
        sine = mx.sin(angles).astype(x.dtype).astype(mx.float32)
        source = x.astype(mx.float32)
        half = source.shape[-1] // 2
        rotated = mx.concatenate([-source[..., half:], source[..., :half]], axis=-1)
        return (source * cosine + rotated * sine).astype(x.dtype)

    def __call__(self, x, mask, cache, value_residual):
        a = self.args
        batch, length, _ = x.shape
        value_input = x
        if value_residual is not None:
            blend = mx.sigmoid(self.value_res_lambda[0]).astype(x.dtype)
            start = value_residual.astype(x.dtype)
            value_input = start + blend * (x - start)
        q = self.q_norm(self.q_proj(x).reshape(batch, length, a.num_attention_heads, a.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(batch, length, a.num_key_value_heads, a.head_dim))
        v = self.v_proj(value_input).reshape(batch, length, a.num_key_value_heads, a.head_dim)
        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        if self.sliding:
            offset = cache.offset if cache is not None else 0
            q, k = self._rope(q, offset), self._rope(k, offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        y = scaled_dot_product_attention(q, k, v, cache=cache, scale=a.head_dim**-0.5, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if a.use_gated_attention:
            gate = self.g_proj(x)
            if a.gate_full_precision:
                gate = gate.astype(mx.float32)
            y = y * mx.sigmoid(gate).astype(y.dtype)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.limit = args.ffn_clamp_threshold or None
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.limit:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        return self.down_proj(nn.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args, layer):
        super().__init__()
        self.self_attn = Attention(args, layer)
        self.mlp = MLP(args)
        dim, eps = args.hidden_size, args.rms_norm_eps
        self.input_layernorm = RMSNorm(dim, eps)
        self.post_attention_layernorm = RMSNorm(dim, eps)
        self.pre_feedforward_layernorm = RMSNorm(dim, eps)
        self.post_feedforward_layernorm = RMSNorm(dim, eps)

    def __call__(self, x, mask, cache, value_residual):
        h = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), mask, cache, value_residual)
        )
        return h + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))


class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Block(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        if self.args.embed_scale is not None:
            h = h * mx.array(self.args.embed_scale).astype(h.dtype)
        value_residual = h if self.args.use_valres else None
        cache = cache or [None] * len(self.layers)
        masks = {}
        for kind, c in zip(self.args.layer_types, cache):
            if kind not in masks:
                window = self.args.sliding_window if kind == SLIDING else None
                masks[kind] = create_attention_mask(h, c, window_size=window)
        for layer, kind, c in zip(self.layers, self.args.layer_types, cache):
            h = layer(h, masks[kind], c, value_residual)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Backbone(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, **kwargs):
        return self.lm_head(self.model(inputs, cache))

    def sanitize(self, weights):
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            RotatingKVCache(max_size=self.args.sliding_window) if kind == SLIDING else KVCache()
            for kind in self.args.layer_types
        ]
