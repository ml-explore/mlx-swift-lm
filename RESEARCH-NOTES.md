# RESEARCH-NOTES — Gemma 4 MTP drafter Swift port (PLAN §1 deliverable)

> Produced for `PLAN-Gemma4Assistant-PR_v3.1.md` §1. Each answer cites
> file:line in the relevant source. Citations use:
>
> - **mlx-vlm @ d49d428** (`/Users/nellymoon/Documents/Code/mlx-vlm/`)
> - **mlx-swift-lm @ lmoutput-state** (`/Users/nellymoon/Documents/Code/mlx-swift-lm/`, branch `gemma4-assistant` off `upstream/lmoutput-state`)

---

## A. Drafter `hidden_size` vs target `hidden_size`

**Independent.** The drafter has its own `text_config.hidden_size` (1024 in both 31B and 26B-A4B-assistant checkpoints — see question N). The target's `hidden_size` is recorded separately as `Gemma4AssistantConfig.backbone_hidden_size`.

Citations:
- `mlx-vlm/mlx_vlm/speculative/drafters/gemma4_assistant/config.py:18` — `backbone_hidden_size: int = 1536` (top-level config field, separate from `text_config.hidden_size`)
- `mlx-vlm/.../gemma4_assistant.py:40-42` — `pre_projection = Linear(2 * config.backbone_hidden_size, text_cfg.hidden_size, bias=False)` (proves the two are different axes of the projection)
- `mlx-vlm/.../gemma4_assistant.py:43-45` — `post_projection = Linear(text_cfg.hidden_size, config.backbone_hidden_size, bias=False)` (round-trips back)

---

## B. Drafter `embed_tokens`: own table or borrowed from target?

**Both — and which one matters depends on the call site.** The drafter *owns* its own `model.embed_tokens` (an `nn.Embedding`) that gets loaded from the checkpoint — but it is **not** used as an input lookup at runtime. Its only runtime role is to serve as the tied `lm_head` (transposed weight matrix) when `tie_word_embeddings=True`. At runtime the drafter borrows the **target's** `embed_tokens` for input lookup, cached via `bind()`.

Citations:
- `mlx-vlm/.../gemma4_assistant.py:19-21` — drafter's own embed_tokens declared in `_DraftInner.__init__`
- `mlx-vlm/.../gemma4_assistant.py:67-93` — `bind(target_model)` walks the target's nesting (`target_model.embed_tokens` direct, `.model.embed_tokens` one level, `.language_model.model.embed_tokens` two levels) and caches `_input_embed = inner.embed_tokens`
- `mlx-vlm/.../gemma4_assistant.py:269` — `tok_embed = self._input_embed(tok) * self._input_embed_scale` (input lookup uses **target's** embedding)
- `mlx-vlm/.../gemma4_assistant.py:69-76` — `_lm_head_fn` resolution: if `masked_embedding` → use it; elif `tie_word_embeddings` → `self.model.embed_tokens.as_linear` (drafter's own); else → `self.lm_head` (drafter's own separate head)

Implication for Swift: `Gemma4AssistantDraftInner.embedTokens` must be wired as an `Embedding` so weight loading succeeds, but never receive an input ID at runtime — only its `weight` is consumed (via `asLinear`).

---

## C. Input projection exact shape

**Confirmed:** `Linear(2 * backbone_hidden_size → drafter_hidden_size)`, no bias.

Citations:
- `mlx-vlm/.../gemma4_assistant.py:40-42` — `self.pre_projection = nn.Linear(2 * config.backbone_hidden_size, text_cfg.hidden_size, bias=False)`
- Weight-key cross-check (question N): 31B drafter `pre_projection.weight` is `[1024, 10752]` = `[drafter_hidden_size, 2 * backbone_hidden_size]` (10752 = 2×5376); 26B-A4B drafter `pre_projection.weight` is `[1024, 5632]` (5632 = 2×2816). Matches.

---

## D. Within-round hidden state propagation

The output that feeds back into step N+1's concat is the **post-projection** output (`last_hidden`, in backbone-hidden-size space), NOT the embedding lookup output and NOT the pre-norm hidden.

Citations:
- `mlx-vlm/.../gemma4_assistant.py:213-215` — `_forward_hidden` returns `(last_hidden, h)`: `last_hidden = post_projection(norm(h))`; `h` is the pre-projection-space norm output used for lm_head
- `mlx-vlm/.../gemma4_assistant.py:268-282` — `draft_block` loop:
  ```
  for _ in range(block_size - 1):
      tok_embed = self._input_embed(tok) * self._input_embed_scale  # [B, 1, backbone_hidden_size]
      inputs_embeds = mx.concatenate([tok_embed, h_prev], axis=-1)  # [B, 1, 2*backbone_hidden_size]
      h_prev, logits = self(inputs_embeds, shared_kv, position_ids)  # h_prev is last_hidden
      tok = sampler(logits)
  ```
- The shapes must agree on `axis=-1`: `tok_embed` is `backbone_hidden_size`, so `h_prev` *must* also be `backbone_hidden_size`. That's why `post_projection` exists.

---

## E. Layer composition: Q-only with cross-attention, `kv_shared_only=True`

**Confirmed.** All 4 drafter layers are constructed via `DecoderLayer(text_config, layer_idx=i, kv_shared_only=True)`. Per-layer weight inventory (question N) shows only `q_proj`, `q_norm`, `o_proj` for `self_attn` — no `k_proj`, `v_proj`, `k_norm`, `v_norm`.

Citations:
- `mlx-vlm/.../gemma4_assistant.py:22-25` — `DecoderLayer(text_config, layer_idx=i, kv_shared_only=True) for i in range(text_config.num_hidden_layers)`
- `mlx-vlm/.../gemma4_assistant.py:201-211` — runtime: `kv = shared_kv_states[layer.layer_type]; layer(h, mask=mask, shared_kv=kv, offset=offset, ...)`
- Weight inventories (question N): both 31B and 26B-A4B drafters have exactly 12 self_attn weight tensors (3 per layer × 4 layers = `q_proj`, `q_norm`, `o_proj`); zero `k_*` or `v_*` keys.

Implication for Swift: `Gemma4TextAttention.init(config:, layerIdx:, kvSharedOnly: true)` must skip constructing `kProj`, `vProj`, `kNorm`, `vNorm`. PLAN §4's Optional `@ModuleInfo` migration matches this.

---

## F. Q head count source

The drafter's Q head count is determined by its **own** `text_config` (number of heads × head_dim must match `q_proj` output shape from the checkpoint). Independent of target.

Citations:
- `mlx-vlm/.../gemma4_assistant.py:22-25` — `DecoderLayer(text_config, layer_idx=i, ...)` — text_config is the drafter's, not the target's
- Weight inventory (question N): 31B drafter layers 0/1/2 have `q_proj.weight = [8192, 1024]`, `q_norm = [256]` (head_dim=256, num_heads=32). Layer 3 has `q_proj.weight = [16384, 1024]`, `q_norm = [512]` (head_dim=512, num_heads=32 for the wide layer — corresponds to "global" / full_attention).
- 26B-A4B drafter layers 0/1/2: `q_proj.weight = [4096, 1024]`, `q_norm = [256]` (head_dim=256, num_heads=16). Layer 3: `q_proj.weight = [8192, 1024]`, `q_norm = [512]` (head_dim=512, num_heads=16).

Implication for Swift: the drafter's `text_config` is loaded from the assistant checkpoint's `config.json`. Heads/head_dim are taken from there; do **not** read from target's config.

---

## G. lm_head resolution + `tie_word_embeddings`

Three branches in `bind()` set `_lm_head_fn`:
1. If `masked_embedding is not None` → `_lm_head_fn = lambda h: masked(h, embed_w)` (where `embed_w = self.model.embed_tokens.weight`)
2. Elif `tie_word_embeddings=True` → `_lm_head_fn = self.model.embed_tokens.as_linear`
3. Else → `_lm_head_fn = self.lm_head` (a separately-declared Linear)

Both 31B and 26B-A4B-assistant checkpoints have **`tie_word_embeddings=True`** (verified: no `lm_head.weight` key in the safetensors — see question N). The default in `Gemma4AssistantConfig` is also `True` (`config.py:22`).

Citations:
- `mlx-vlm/.../gemma4_assistant.py:46-49` — separate `lm_head` Linear only declared when `not config.tie_word_embeddings`
- `mlx-vlm/.../gemma4_assistant.py:69-76` — `bind()` resolution of `_lm_head_fn`
- `mlx-vlm/.../gemma4_assistant.py:285-294` — `sanitize` drops `lm_head.weight` if `tie_word_embeddings=True`
- `mlx-vlm/.../config.py:22` — `tie_word_embeddings: bool = True` default

Implication for Swift: `Gemma4AssistantDraftModel.lmHead: Linear?` is Optional, constructed only when `!config.tieWordEmbeddings`. In Swift, `Embedding.asLinear(h)` is the tied path. `sanitize` removes the orphaned `lm_head.weight` key on tied checkpoints.

---

## H. Shared K/V indexing

**Confirmed: dict keyed by layer_type string ("full_attention" / "sliding_attention").** The drafter looks up its layer's `layer_type` to retrieve the corresponding `(K, V)` pair.

Per-layer-type, with only two distinct types in Gemma 4, there are **exactly two** entries in the dict.

Citations:
- `mlx-vlm/.../gemma4_assistant.py:201-211` — `for layer in self.model.layers: kv = shared_kv_states[layer.layer_type]`
- `mlx-vlm/.../masks.py:157-179` — `make_drafter_masks` iterates `for layer_type, kv in shared_kv_states.items()`; branches on `layer_type == "sliding_attention"` vs full
- `mlx-vlm/.../gemma4_assistant.py:96-99` — `bind()` records `self.config.target_layer_types = list(tcfg.layer_types)` (the full list of types per target layer; consumers infer `{full, sliding}` from this)

Implication for Swift: `sharedKV: [String: (MLXArray, MLXArray)]` with keys `"full_attention"` and `"sliding_attention"`. The MTP iterator (Phase B) extracts these from target's `LMOutput.state[mtpSharedKVStatesKey]`.

---

## I. Attention masking

**`bidirectional_full_mask` and `bidirectional_swa_mask`** are defined in mlx-vlm itself, not imported from HF transformers.

### `bidirectional_full_mask` (`masks.py:48-79`)

In the unbatched, no-padding case → returns `None` (SDPA handles it). With per-row valid lengths (batched MTP), returns an additive bias `[B, 1, 1, kv_len]` with `0.0` for positions `< kv_valid_len` and `-inf` past.

### `bidirectional_swa_mask` (`masks.py:82-142`)

For each query position `q ∈ [query_offset, query_offset + query_len)`, allow attention to KV positions `k` where `|q - k| < window` (bidirectional!). Returns `None` if the entire KV fits inside everyone's window. Otherwise returns additive bias of shape `[1, 1, query_len, kv_len]` (unbatched) or `[B, 1, query_len, kv_len]` (batched).

**The "SWA kv-axis flip" claim from PLAN v3.1 doesn't apply here** — there is no explicit `.flip(dims=(-1,))` in mlx-vlm's masks. The cache rotation (rotating KV cache) is handled by `mlx_lm.models.cache.dynamic_roll` (imported at `masks.py:14`) when normalizing batched layouts, but the mask itself is computed by query-key distance comparison, not flip. The PLAN's "flip" terminology comes from HF transformers' `masking_utils.py`; mlx-vlm reaches the same effect by computing `dist = q_idx - k_idx; inside = (dist > -window) & (dist < window)` and then constructing the additive bias.

Citations:
- `mlx-vlm/.../masks.py:1-9` — docstring summarizing intent
- `mlx-vlm/.../masks.py:48-79` — `bidirectional_full_mask`
- `mlx-vlm/.../masks.py:82-142` — `bidirectional_swa_mask`
- `mlx-vlm/.../masks.py:145-179` — `make_drafter_masks` dispatcher
- `mlx-vlm/.../masks.py:117-123` — explicit dist-based formula: `dist = q_idx - k_idx; inside = (dist > -window) & (dist < window)`
- `mlx-vlm/.../masks.py:14` — `from mlx_lm.models.cache import dynamic_roll` (used only for normalize_batched_shared_kv_states)

Implication for Swift: `Libraries/MLXLMCommon/BidirectionalMasks.swift` should provide both `createBidirectionalMask` (a thin wrapper that returns nil when no masking needed, or a zero `[queryLen, kvLen]` array for the always-attend case) and `createBidirectionalSlidingWindowMask` (additive bias from distance check). **No literal `.flip` step needed** — compute the additive bias from distances directly. Phase A's mask tests verify byte-identity against `fixtures/masks/`, which were generated by this Python code.

---

## J. `MaskedEmbedder` (centroid-routed sparse softmax)

**Active only when `use_ordered_embeddings=True`.** In the two checkpoints we care about (31B-assistant-bf16 and 26B-A4B-assistant-bf16), this flag is **`False`** (default from `config.py:19`), verified by weight inventory (question N): zero `masked_embedding.*` keys in either checkpoint.

What it computes (when active):
- Centroid scores: `centroids(hidden_states)` → `[B, L, num_centroids]`
- Top-K: `argpartition(...)[..., -top_k:]` → `[B, L, top_k]`
- Lookup `token_ordering` (reshaped to `[num_centroids, vocab_size_per_centroid]`) to get canonical token IDs for selected clusters
- Gather corresponding embeddings, score them densely, scatter back to full-vocab with a sentinel value for un-selected tokens

Citations:
- `mlx-vlm/.../masked_embedder.py:21-72` — class body
- `mlx-vlm/.../masked_embedder.py:39-63` — `__call__` (dense scatter)
- `mlx-vlm/.../masked_embedder.py:65-72` — `argmax` (cheaper greedy path)
- `mlx-vlm/.../config.py:19-21` — `use_ordered_embeddings: bool = False`, `num_centroids: int = 2048`, `centroid_intermediate_top_k: int = 32`
- Weight inventory (question N): no `masked_embedding.*` keys in 31B-assistant-bf16 or 26B-A4B-assistant-bf16

Implication for Swift: implement `Gemma4AssistantMaskedEmbedder` as an Optional module on `Gemma4AssistantDraftModel`, constructed only when `useOrderedEmbeddings == true`. Since neither test checkpoint exercises this path, Phase A correctness for `MaskedEmbedder` rests on **shape tests against synthetic data**, not fixture parity. Real-world MaskedEmbedder verification would need a different checkpoint (e.g., the E2B/E4B drafters referenced in `masked_embedder.py:1`).

---

## K. Round-loop sequencing (`_speculative_walk` equivalent)

In mlx-vlm the equivalent of `_speculative_walk` for MTP is `_mtp_rounds` (`speculative/utils.py:535-684`). Sequencing:

1. **`bind` called once** at the top: `draft_model.reset(model)` (`utils.py:568`), which calls `bind(target_model)` internally (`gemma4_assistant.py:108-109`)
2. **First `set_shared_kv` after prefill**: `utils.py:594-599` — sets initial `shared_kv_states`, `kv_offset`, `position`, `kv_valid_len`
3. **Per-round `draft_block`**: `utils.py:614-622` — passes `(b, hidden, None, bs, sampler, token_dtype, ...)` where `b` is the bonus token, `hidden` is the target's last hidden state (sliced from prefill or previous verify)
4. **Verify step**: `_mtp_verify_target` at `utils.py:629-634` — one target forward pass on `[bonus] + draft_tokens`
5. **Acceptance walk**: `_mtp_acceptance_walk` at `utils.py:635-641` — compares drafted vs verified tokens, returns `(accepted_count, new_tokens)`
6. **Cache rewind on rejection**: `utils.py:666-669` — `lm.rollback_speculative_cache(prompt_cache, ...)` (target only; drafter has no cache, per `gemma4_assistant.py:104-106` `make_cache() returns []`)
7. **Per-round `set_shared_kv` update**: `utils.py:675-680` — re-bind drafter's shared K/V with the new offset/position after acceptance + rollback

Citations: as numbered above.

Implication for Swift `MTPSpeculativeTokenIterator` (Phase B):
- Call `drafter.bind(target:)` in `init`
- Per round: read `LMOutput.state[mtpSharedKVStatesKey]` and `[mtpLastHiddenStatesKey]` from the prior main forward, then call `drafter.draftBlock(lastToken:, lastHidden:, sharedKV:, positionIds:, blockSize:, sampler:)`
- After verify, trim main cache by `(draftedCount - accepted)` tokens
- Drafter has **no cache** to trim — simpler than `SpeculativeTokenIterator` in this regard

---

## L. Position IDs constant per round

**Confirmed constant within a drafting round.** The drafter constructs `position_ids` once from the bind-time `self._position` ivar (set by `set_shared_kv`) and re-uses it for every step in `draft_block`'s inner loop.

Citations:
- `mlx-vlm/.../gemma4_assistant.py:255-258` — `position_ids = mx.array([[self._position]])` (computed once, before the loop)
- `mlx-vlm/.../gemma4_assistant.py:268-282` — `for _ in range(block_size - 1): ...` uses `position_ids` repeatedly without mutating it
- `mlx-vlm/speculative/utils.py:594-599` — initial `position = _mtp_draft_position(kv_offset)` = target's current cache offset

Implication for Swift: `draftBlock(positionIds:)` takes the constant position as a method argument; it doesn't change across inner iterations.

---

## M. Initial bonus token after prefill

The bonus token (`b` in `_mtp_rounds`) is **sampled by the caller from target's last prefill logits** and passed in. `_mtp_rounds` accepts it as `first_bonus` (`utils.py:540` parameter; consumed as `b = first_bonus` at `utils.py:601`).

Citations:
- `mlx-vlm/speculative/utils.py:540` (parameter declaration in function signature)
- `mlx-vlm/speculative/utils.py:601-602` — `b = first_bonus; emitted = 1  # caller already yielded the first bonus`
- Upstream sampling site: `_mtp_rounds` is called from `generate_step` (further up `utils.py`); the caller has already done `sampler(prefill_logits[:, -1, :])`

Implication for Swift: `MTPSpeculativeTokenIterator` consumes the first bonus token from the prefill `LMOutput.logits` itself (just like `SpeculativeTokenIterator` does), then starts the round loop.

---

## N. Drafter weight keys (exhaustive)

Captured via `tools/inspect_drafter_layout.py` against locally cached HF snapshots. Both checkpoints have **48 tensors** with the **same key layout** (only shapes differ).

### 31B-assistant-bf16 (snapshot `28e92270…`)

```
model.embed_tokens.weight                         [262144, 1024]  BF16
model.layers.0.input_layernorm.weight             [1024]          BF16
model.layers.0.layer_scalar                       [1]             BF16
model.layers.0.mlp.down_proj.weight               [1024, 8192]    BF16
model.layers.0.mlp.gate_proj.weight               [8192, 1024]    BF16
model.layers.0.mlp.up_proj.weight                 [8192, 1024]    BF16
model.layers.0.post_attention_layernorm.weight    [1024]          BF16
model.layers.0.post_feedforward_layernorm.weight  [1024]          BF16
model.layers.0.pre_feedforward_layernorm.weight   [1024]          BF16
model.layers.0.self_attn.o_proj.weight            [1024, 8192]    BF16
model.layers.0.self_attn.q_norm.weight            [256]           BF16
model.layers.0.self_attn.q_proj.weight            [8192, 1024]    BF16
model.layers.1.input_layernorm.weight             [1024]          BF16
model.layers.1.layer_scalar                       [1]             BF16
model.layers.1.mlp.down_proj.weight               [1024, 8192]    BF16
model.layers.1.mlp.gate_proj.weight               [8192, 1024]    BF16
model.layers.1.mlp.up_proj.weight                 [8192, 1024]    BF16
model.layers.1.post_attention_layernorm.weight    [1024]          BF16
model.layers.1.post_feedforward_layernorm.weight  [1024]          BF16
model.layers.1.pre_feedforward_layernorm.weight   [1024]          BF16
model.layers.1.self_attn.o_proj.weight            [1024, 8192]    BF16
model.layers.1.self_attn.q_norm.weight            [256]           BF16
model.layers.1.self_attn.q_proj.weight            [8192, 1024]    BF16
model.layers.2.input_layernorm.weight             [1024]          BF16
model.layers.2.layer_scalar                       [1]             BF16
model.layers.2.mlp.down_proj.weight               [1024, 8192]    BF16
model.layers.2.mlp.gate_proj.weight               [8192, 1024]    BF16
model.layers.2.mlp.up_proj.weight                 [8192, 1024]    BF16
model.layers.2.post_attention_layernorm.weight    [1024]          BF16
model.layers.2.post_feedforward_layernorm.weight  [1024]          BF16
model.layers.2.pre_feedforward_layernorm.weight   [1024]          BF16
model.layers.2.self_attn.o_proj.weight            [1024, 8192]    BF16
model.layers.2.self_attn.q_norm.weight            [256]           BF16
model.layers.2.self_attn.q_proj.weight            [8192, 1024]    BF16
model.layers.3.input_layernorm.weight             [1024]          BF16
model.layers.3.layer_scalar                       [1]             BF16
model.layers.3.mlp.down_proj.weight               [1024, 8192]    BF16
model.layers.3.mlp.gate_proj.weight               [8192, 1024]    BF16
model.layers.3.mlp.up_proj.weight                 [8192, 1024]    BF16
model.layers.3.post_attention_layernorm.weight    [1024]          BF16
model.layers.3.post_feedforward_layernorm.weight  [1024]          BF16
model.layers.3.pre_feedforward_layernorm.weight   [1024]          BF16
model.layers.3.self_attn.o_proj.weight            [1024, 16384]   BF16
model.layers.3.self_attn.q_norm.weight            [512]           BF16
model.layers.3.self_attn.q_proj.weight            [16384, 1024]   BF16
model.norm.weight                                 [1024]          BF16
post_projection.weight                            [5376, 1024]    BF16
pre_projection.weight                             [1024, 10752]   BF16
```

### 26B-A4B-assistant-bf16 (snapshot `cda74908…`)

Same key set; differing shapes (smaller backbone):

```
post_projection.weight                            [2816, 1024]    BF16
pre_projection.weight                             [1024, 5632]    BF16
# (layers 0/1/2: q_proj.weight [4096, 1024], q_norm [256], o_proj [1024, 4096])
# (layer 3:      q_proj.weight [8192, 1024], q_norm [512], o_proj [1024, 8192])
```

(Full listing reproduced verbatim by re-running `python tools/inspect_drafter_layout.py mlx-community/gemma-4-26B-A4B-it-assistant-bf16`.)

### Cross-checkpoint observations

- **No `lm_head.weight`** in either checkpoint → confirms `tie_word_embeddings=True` is the operative case.
- **No `masked_embedding.*` keys** → confirms `use_ordered_embeddings=False` for these two checkpoints (question J).
- **Per-layer attention has only `q_proj`, `q_norm`, `o_proj`** — no `k_proj`, `v_proj`, `k_norm`, `v_norm` → confirms all 4 layers are `kv_shared_only=True` (question E).
- **Layer 3 alone has wider Q dims** in both checkpoints (head_dim 512 instead of 256). Cross-references mlx-vlm's `text_config.layer_types`: layer 3 is the global (full_attention) layer with the wider head; layers 0/1/2 are sliding_attention with narrower heads.
- **Top-level (outside `model.`)**: `pre_projection.weight` and `post_projection.weight` only. The drafter `Module` graph in Swift should declare these at the top level (alongside `model: Gemma4AssistantDraftInner`), matching the Python module hierarchy.

### Swift `@ModuleInfo(key:)` mapping

| Weight key | Swift attribute path |
|---|---|
| `model.embed_tokens.weight` | `model.embedTokens.weight` |
| `model.layers.N.input_layernorm.weight` | `model.layers[N].inputLayernorm.weight` |
| `model.layers.N.layer_scalar` | `model.layers[N].layerScalar` |
| `model.layers.N.mlp.{gate,up,down}_proj.weight` | `model.layers[N].mlp.{gate,up,down}Proj.weight` |
| `model.layers.N.post_attention_layernorm.weight` | `model.layers[N].postAttentionLayernorm.weight` |
| `model.layers.N.post_feedforward_layernorm.weight` | `model.layers[N].postFeedforwardLayernorm.weight` |
| `model.layers.N.pre_feedforward_layernorm.weight` | `model.layers[N].preFeedforwardLayernorm.weight` |
| `model.layers.N.self_attn.{q,o}_proj.weight` | `model.layers[N].selfAttn.{q,o}Proj.weight` |
| `model.layers.N.self_attn.q_norm.weight` | `model.layers[N].selfAttn.qNorm.weight` |
| `model.norm.weight` | `model.norm.weight` |
| `pre_projection.weight` | `preProjection.weight` |
| `post_projection.weight` | `postProjection.weight` |

These mostly match `Gemma4TextDecoderLayer` and `Gemma4TextAttention` already in `Libraries/MLXVLM/Models/Gemma4.swift` — confirms PLAN §4 / §6's reuse-don't-reinvent strategy.

---

## O. `ModelTypeRegistry<T>.register(_:creator:)` visibility on `lmoutput-state`

The method exists and is **public**, but is named **`registerModelType(_:creator:)`**, not `register(_:creator:)`.

Citation:
- `Libraries/MLXLMCommon/Registries/ModelTypeRegistry.swift:20-24`:
  ```swift
  public func registerModelType(
      _ type: String, creator: @escaping (Data) throws -> T
  )
  ```

Implication: **no `ModelTypeRegistry<T>` extension needed** (PLAN risk R10 is moot). PLAN §8's `Gemma4AssistantRegistration` should call `MTPDrafterTypeRegistry.shared.registerModelType("gemma4_assistant", creator: ...)`.

---

## P. Internal-scope access on `Gemma4TextLanguageModel`

The class itself is **`private`** as of `lmoutput-state` (line 1057 of `Gemma4.swift`):

```
1057:private final class Gemma4TextLanguageModel: Module, KVCacheDimensionProvider {
```

Inside the class, `embedTokens`/`embedScale`/`config.layerTypes` themselves are accessible (member properties of an internal-target class), but you can't reach them from outside the class declaration if the class is `private`. The `Gemma4TextConfiguration.layerTypes` field is itself `public` (`Gemma4.swift` `public struct Gemma4TextConfiguration`), but the **path through the language-model instance** is gated by the language-model class being `private`.

**Conclusion:** PLAN §4 visibility list must include `Gemma4TextLanguageModel` (in addition to the seven symbols PLAN names: `Gemma4TextAttention`, `Gemma4TextDecoderLayer`, `Gemma4SharedKVState`, `Gemma4RMSNormZeroShift`, `Gemma4RMSNormNoScale`, `gemma4AdjustAttentionMask`, `gemma4DefaultTextRopeParameters`). Without it, `Gemma4AssistantDraftModel.bind(target:)`'s `extractGemma4Text` helper cannot reach `target.languageModel.embedTokens`.

Also need access to `Gemma4TextBackbone` (line 862, also `private`) since the target's nesting may be `Gemma4.languageModel.model.embedTokens` (mirrors mlx-vlm's three-level `.language_model.model.embed_tokens` path at `gemma4_assistant.py:85-90`). **Add `Gemma4TextBackbone` to the visibility list as well.**

Citations:
- `Libraries/MLXVLM/Models/Gemma4.swift:862` — `private final class Gemma4TextBackbone`
- `Libraries/MLXVLM/Models/Gemma4.swift:1057` — `private final class Gemma4TextLanguageModel`
- Existing `public struct Gemma4TextConfiguration` (around line 234) — already public; `layerTypes` field accessible once you can reach a `config` instance.

---

## Verification gate (PLAN §1 transient gate)

- [x] `RESEARCH-NOTES.md` exists at repo root
- [x] Every answer (A–P) has file:line citation
- [ ] Spot-check 3 answers manually by re-opening cited files (to be done after this file is reviewed)

## Open follow-ups (not blocking Phase A)

- **(I, mask flip)** PLAN v3.1 §2/§5 describes a "SWA kv-axis flip" that doesn't exist literally in mlx-vlm @ d49d428. Swift `BidirectionalMasks.swift` should follow mlx-vlm's distance-comparison formulation, not the "flip" framing. Update PLAN §5 doc-comment language to match.
- **(P, visibility)** PLAN §4 should be amended in implementation to also expose `Gemma4TextBackbone` and `Gemma4TextLanguageModel`.
- **(J, MaskedEmbedder)** Phase A verification for MaskedEmbedder can't ride on the current fixture set (both test checkpoints have `use_ordered_embeddings=False`). Add a synthetic shape test in `Gemma4AssistantDraftModelTests`.
- **(§10, drafter_forward fixtures — fixed)** Original NaN issue: all five `tools/fixtures/drafter_forward/*.safetensors` files had **all-NaN `outputs/last_hidden` and `outputs/logits`** despite finite inputs. **Root cause:** `_forward_hidden` reads `self._kv_valid_len` (an ivar) to build the full-attention mask via `make_drafter_masks` (`gemma4_assistant.py:193`); when left at the init default `0`, `bidirectional_full_mask` masks every key position to `-inf` (`masks.py:67-72`), and post-softmax attention becomes NaN. The block-suite captures already called `set_shared_kv` (which initializes `_kv_valid_len`), which is why only `drafter_forward` was affected. **Fix:** `tools/generate_mtp_fixtures.py` `generate_drafter_forward_fixtures` now calls `ctx.drafter.set_shared_kv(shared_kv_states=…, kv_offset=kv_len, position=kv_len)` before invoking `drafter(...)`. Regenerated fixtures (still at SHA `d49d428`) have finite outputs with reasonable magnitudes (last_hidden max ≈ 27–40, logits max ≈ 21–32 across the five cases). Swift Rung 2/3 byte-identity test (`allClose(..., rtol: 1e-3, atol: 1e-3)`) now passes against the regenerated fixtures.
