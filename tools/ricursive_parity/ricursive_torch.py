"""Minimal PyTorch Ricursive forward, for making parity oracles.

A transcription of the forward path in `modeling_ricursive.py` from
ricursiveAI/Ripple-65B-Preview (Apache-2.0), written here so that the oracle does
not have to execute Python from a downloaded model repository. It has no dependency
on transformers. Padding, labels and generation helpers are left out. The cache
keeps the sliding layers' last `window - 1` keys, which is the eviction a rotating
cache performs and does not change the attended keys.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def _rms(x, weight, eps):
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps)).to(dtype)


def _rope(x, cos, sin):
    dtype = x.dtype
    x = x.float()
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos.float() + rotated * sin.float()).to(dtype)


class LayerCache:
    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, keys, values, keep):
        """Append keys and return them, retaining at most `keep` older rows."""
        if self.keys is not None:
            if keep is not None and self.keys.shape[2] > keep:
                self.keys = self.keys[:, :, -keep:]
                self.values = self.values[:, :, -keep:]
            keys = torch.cat((self.keys, keys), dim=2)
            values = torch.cat((self.values, values), dim=2)
        self.keys, self.values = keys, values
        return keys, values


class RicursiveTorch:
    def __init__(self, config, weights, device):
        self.c = config
        self.w = weights
        self.device = device
        self.sliding = [t == "sliding_attention" for t in config["layer_types"]]
        self.eps = config["rms_norm_eps"]
        head_dim = config["head_dim"]
        self.inverse_frequencies = 1.0 / (
            config["rope_theta"]
            ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )

    def new_cache(self):
        return [LayerCache() for _ in self.sliding]

    def _linear(self, name, x):
        return x @ self.w[name + ".weight"].t()

    def _attention(self, i, x, value_residual, cache, offset):
        c, p = self.c, f"model.layers.{i}.self_attn"
        batch, length, _ = x.shape
        heads, kv_heads, head_dim = (
            c["num_attention_heads"],
            c["num_key_value_heads"],
            c["head_dim"],
        )

        q = self._linear(p + ".q_proj", x)
        k = self._linear(p + ".k_proj", x)
        value_states = x
        if value_residual is not None:
            blend = torch.sigmoid(self.w[p + ".value_res_lambda"][0]).to(x.dtype)
            value_states = torch.lerp(value_residual.to(x.dtype), x, blend)
        v = self._linear(p + ".v_proj", value_states)

        q = _rms(q.view(batch, length, heads, head_dim), self.w[p + ".q_norm.weight"], self.eps)
        k = _rms(k.view(batch, length, kv_heads, head_dim), self.w[p + ".k_norm.weight"], self.eps)
        v = v.view(batch, length, kv_heads, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.sliding[i]:
            positions = torch.arange(offset, offset + length, device=x.device).float()
            frequencies = positions[:, None] * self.inverse_frequencies[None, :]
            angles = torch.cat((frequencies, frequencies), dim=-1)
            cos = angles.cos().to(x.dtype)[None, None]
            sin = angles.sin().to(x.dtype)[None, None]
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)

        window = c["sliding_window"]
        if cache is not None:
            k, v = cache.update(k, v, window - 1 if self.sliding[i] else None)

        groups = heads // kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

        # Absolute positions: the keys end at the newest query.
        last = offset + length
        key_positions = torch.arange(last - k.shape[2], last, device=x.device)
        query_positions = torch.arange(offset, last, device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        if self.sliding[i]:
            allowed = allowed & (key_positions[None, :] >= query_positions[:, None] - (window - 1))
        mask = None if allowed.all() else allowed[None, None]

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        output = output.transpose(1, 2).contiguous().view(batch, length, -1)

        if c["use_gated_attention"]:
            gate = self._linear(p + ".g_proj", x)
            if c["gate_full_precision"]:
                gate = gate.float()
            output = output * torch.sigmoid(gate).to(output.dtype)
        return self._linear(p + ".o_proj", output)

    def _mlp(self, i, x):
        p, limit = f"model.layers.{i}.mlp", self.c.get("ffn_clamp_threshold")
        gate = self._linear(p + ".gate_proj", x)
        up = self._linear(p + ".up_proj", x)
        if limit:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        return self._linear(p + ".down_proj", F.silu(gate) * up)

    @torch.no_grad()
    def forward(self, ids, cache=None, offset=0, head_rows=None):
        """Logits for the last `head_rows` positions (None for all, 0 for none)."""
        scale = self.c.get("embed_scale")
        h = F.embedding(ids, self.w["model.embed_tokens.weight"])
        if scale is not None:
            h = h * scale
        value_residual = h if self.c["use_valres"] else None

        for i in range(self.c["num_hidden_layers"]):
            p = f"model.layers.{i}"
            normed = _rms(h, self.w[p + ".input_layernorm.weight"], self.eps)
            layer_cache = cache[i] if cache is not None else None
            attended = self._attention(i, normed, value_residual, layer_cache, offset)
            h = h + _rms(attended, self.w[p + ".post_attention_layernorm.weight"], self.eps)
            normed = _rms(h, self.w[p + ".pre_feedforward_layernorm.weight"], self.eps)
            h = h + _rms(self._mlp(i, normed), self.w[p + ".post_feedforward_layernorm.weight"], self.eps)

        if head_rows == 0:
            return None
        h = _rms(h, self.w["model.norm.weight"], self.eps)
        if head_rows is not None:
            h = h[:, -head_rows:, :]
        return self._linear("lm_head", h)

    def prefill_and_generate(self, prompt, steps, prefill_step, on_step):
        """Prefill all but the last token in chunks, then decode greedily.

        `on_step(index, token, logits)` receives each generated token and the
        `[vocab]` float32 logits it was chosen from.
        """
        cache = self.new_cache()
        offset = 0
        last = len(prompt) - 1
        while offset < last:
            count = min(prefill_step, last - offset)
            ids = torch.tensor([prompt[offset : offset + count]], device=self.device)
            self.forward(ids, cache, offset, head_rows=0)
            offset += count
        token = prompt[last]
        for index in range(steps):
            ids = torch.tensor([[token]], device=self.device)
            logits = self.forward(ids, cache, offset, head_rows=1)[0, -1].float()
            offset += 1
            token = int(logits.argmax())
            on_step(index, token, logits)


def load_checkpoint(directory, dtype, device):
    """Read `config.json` and every safetensors shard of a BF16 or float32 checkpoint."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors files in {directory}")
    weights = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key).to(dtype).to(device)
    return config, weights
