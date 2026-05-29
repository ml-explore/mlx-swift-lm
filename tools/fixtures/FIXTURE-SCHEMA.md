# MTP Speculative Decoding Fixture Schema

These fixtures verify the Swift port of Gemma 4 MTP speculative decoding in
`ml-explore/mlx-swift-lm` against canonical Python reference from `Blaizzy/mlx-vlm`
pinned at SHA `d49d428e9f570dc0387b9598b3b7e0ea391590d2`.

All tensor files use the `safetensors` format. Keys follow a hierarchical
naming convention with `/` as the separator. Metadata is stored as
string-keyed string values in each file's metadata block.

Random seed for all synthetic inputs: `42`.

## Suites

### `masks/`

Bidirectional and bidirectional-sliding-window attention masks. Used to
verify Swift's `createBidirectionalMask(...)` and
`createBidirectionalSlidingWindowMask(...)` in `MLXLMCommon`.

Keys per file:
  - `mask`: shape `[queryLen, kvLen]`, dtype `float32`, additive
    (0 = attend, -inf = mask)

Metadata:
  - `kind`: `"bidirectional"` or `"bidirectional_swa"`
  - `queryLen`, `kvLen`: as integer strings
  - `windowSize`: present only for SWA fixtures

The SWA mask has the kv-axis flip applied (HF convention): the *last*
`windowSize` positions in the unflipped mask are attended; the flip
reorders so the *first* `windowSize` positions in the returned mask are
attended. Swift implementations must apply the same flip.

### `drafter_forward/`

Single invocations of the drafter's `_forward_hidden` with synthetic inputs.
Each fixture captures one (inputs → outputs) pair sufficient to verify the
drafter's per-call forward pass.

Input keys:
  - `inputs/inputs_embeds`: `[1, queryLen, 2 * backboneHiddenSize]`, bf16
    (concatenation of token embedding and previous hidden state)
  - `inputs/position_ids`: `[1, queryLen]`, int32 (constant within a round)
  - `inputs/shared_kv/full_attention/keys`: `[1, num_kv_heads, kvLen, head_dim]`, bf16
  - `inputs/shared_kv/full_attention/values`: same shape
  - `inputs/shared_kv/sliding_attention/keys`: same shape
  - `inputs/shared_kv/sliding_attention/values`: same shape

Output keys:
  - `outputs/last_hidden`: `[1, queryLen, backboneHiddenSize]`, bf16
    (post-projected back to target's hidden size for downstream concat)
  - `outputs/logits`: `[1, queryLen, vocabSize]`, fp32

Metadata: `queryLen`, `kvLen`, `vocabSize`, `backboneHiddenSize`, `drafterHiddenSize`.

### `drafter_block/`

End-to-end invocations of `draft_block` (the within-round K-1 token autoregressive
loop). Greedy sampling (temperature=0) for reproducibility.

Input keys:
  - `inputs/last_token`: `[1]`, int32 (the bonus token from target's last verify)
  - `inputs/last_hidden`: `[1, 1, backboneHiddenSize]`, bf16
  - `inputs/position_ids`: `[1, 1]`, int32
  - `inputs/shared_kv/...`: same shape conventions as drafter_forward

Output keys:
  - `outputs/drafted_tokens`: `[1, blockSize - 1]`, int32

Metadata: `blockSize`, `kvLen`, `sampling` (always `"greedy"` for these).

### `end_to_end/`

Full target prefill + one or more drafter rounds + verify. Reproduces an
entire generation slice. Greedy sampling.

Input keys:
  - `inputs/prompt_tokens`: `[1, promptLen]`, int32
  - `inputs/temperature`: scalar fp32 (always 0.0 for these fixtures)

Output keys:
  - `outputs/accepted_tokens`: `[1, numAcceptedTokens]`, int32
  - `outputs/acceptance_rate`: scalar fp32
  - `outputs/num_rounds`: scalar int32

Metadata: `prompt` (the original string), `targetModelId`, `drafterModelId`,
plus per-suite shape and parameter details.

Note: every fixture file across every suite carries `targetModelId` and
`drafterModelId` in its metadata block. Consumers reading a fixture can
always check the reference pair without needing the global manifest.

## Versioning

Fixtures are versioned by mlx-vlm SHA. If `PINNED_MLX_VLM_SHA` in
`generate_mtp_fixtures.py` changes, regenerate all fixtures and bump the
HuggingFace dataset revision. Document the change in the dataset's commit
message: which SHA, what changed, why fixtures need regeneration.

## Reference models

Default reference checkpoints (overridable via `--target-model-id` and
`--drafter-model-id`):

  - Target: `mlx-community/gemma-4-31b-it-8bit` (dense, 8-bit weights)
  - Drafter: `mlx-community/gemma-4-31B-it-assistant-bf16` (bf16 weights)

Dense rather than MoE: see preamble comment in `generate_mtp_fixtures.py`.

Quantization asymmetry: the target is 8-bit because that's the production
path real consumers run; the drafter is bf16 because at ~800MB the
quantization savings are marginal against more numerical noise on a smaller
parameter count. The drafter's weight precision doesn't affect K/V tensors
in the cache (those flow through in compute dtype), so the shared K/V the
Swift port reads from the target is independent of drafter weight quant.

Each fixture file records `targetModelId` and `drafterModelId` in metadata
so consumers can verify they're matched against the same reference pair.

## Tolerances for verification

Weight quantization on the target slightly affects intermediate activations
but does NOT affect greedy token outputs at temperature=0 (argmax is
deterministic). So:

  - **Rung 4 (greedy token parity)**: bit-exact, no tolerance.
  - **Rung 3 (logit parity, 8-bit target)**: `atol=2e-3, rtol=2e-3`.
  - **Rung 2 (per-layer activations, 8-bit target)**: `atol=2e-3, rtol=2e-3`.
  - **Mask fixtures (fp32)**: `atol=1e-5, rtol=1e-5`.

Drafter-only fixtures (where the target is touched only for bind() and
embed_tokens) inherit the tighter bf16 tolerances of `atol=1e-3, rtol=1e-3`,
because the drafter is bf16 and there's no 8-bit weight quant in its
compute path.

## Regeneration

From the same M5 Pro with mlx-vlm at the pinned SHA:

```
python tools/generate_mtp_fixtures.py
```

Then upload to HuggingFace:

```
huggingface-cli upload <dataset-id> ./fixtures --repo-type=dataset
```
