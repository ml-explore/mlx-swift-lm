# KV Cache Quantization

Reduce the memory footprint of long-context generation by compressing the
key/value cache.

## Overview

At long context lengths the KV cache, not the weights, dominates memory. Use
``KVCacheConfiguration`` to select capacity, compression, and compatibility as
one validated value:

```swift
let parameters = GenerateParameters(
    kvCache: KVCacheConfiguration(
        strategy: .turboQuant(.balanced)))
```

The strategy is opaque so new cache implementations can be added without
turning a public enum into an exhaustive client-side switch. The current
strategies are:

- **Affine quantization**: ``KVCacheConfiguration/Strategy/affine(_:)``
  quantizes both K and V with MLX's affine scheme.
- **TurboQuant**: ``KVCacheConfiguration/Strategy/turboQuant(_:)`` rotates
  vectors with a Walsh-Hadamard transform and quantizes them against a
  Lloyd-Max codebook.
  TurboQuant schemes are asymmetric. Keys and values can use different
  precision because attention quality is far more sensitive to key error
  (softmax amplifies it) than to value error (linear averaging smooths it).

The older `maxKVSize`, `kvBits`, `kvGroupSize`, `quantizedKVStart`, and
`kvScheme` fields remain available as a compatibility adapter. Do not combine
them with `kvCache`; unknown legacy scheme strings are rejected when generation
starts.

The standalone legacy ``maybeQuantizeKVCache(cache:kvBits:kvGroupSize:quantizedKVStart:kvScheme:)``
hook leaves custom schemes unchanged; generation APIs reject them.

Prefill is unaffected: the cache stores raw fp16 during prompt processing,
compresses on the first decode step, and encodes each new token incrementally
afterwards. Compressed caches round-trip through prompt-cache save/restore.

## Capacity and compatibility

A capacity creates rotating caches for model cache factories that support the
generic bounded-cache path. Rotating caches are not compressible, so combining a
capacity with compression can leave no eligible layers. Select the failure
semantics explicitly. Typed configuration defaults to
`.requireAtLeastOneLayer`, preventing a compression request from silently
becoming an all-fp16 no-op:

- `.allowPartial` compresses eligible global layers and retains rotating layers
  as fp16.
- `.requireAtLeastOneLayer` rejects an all-rotating no-op configuration.
- `.requireAllLayers` rejects any uncompressed attention layer.

Compatibility is checked again after prefill, so unsupported realized head
shapes fail according to the selected policy.

Applications that need a hard total-context cap and compression should omit
cache capacity, use `.requireAtLeastOneLayer`, and enforce the total token budget
before inference. A bounded compressed ring cache is not currently implemented.

``ChatSession/kvCacheRuntimeReport()`` reports the requested configuration and
each realized layer's state, resolved strategy, and skip reason. Its aggregate
counts let an application display compressed, pending, and skipped layer counts
without assuming that a request took effect.

A realized `ChatSession` cache is bound to its configuration. Call
``ChatSession/clear()`` before changing its capacity or strategy.

## Scheme reference

Scheme names read `turbo<K-bits>v<V-bits>`; `0` means keys stay fp16.

| scheme | keys | values | KV compression | character |
|---|---|---|---|---|
| `affine8` | 8-bit affine | 8-bit affine | 1.88x | near-lossless on most models, full decode speed |
| `affine4` | 4-bit affine | 4-bit affine | 3.56x | collapses on some families; validate first |
| `turbo0v4` | fp16 | 4-bit turbo | 1.58x | safest start; beats affine8 quality on most models tested |
| `turbo0v3` | fp16 | 3-bit turbo | 1.66x | light value compression |
| `turbo0v2` | fp16 | 2-bit turbo | 1.58x† | aggressive value compression |
| `turbo8v4` | 8-bit affine | 4-bit turbo | 2.51x | conservative asymmetric |
| `turbo8v3` | 8-bit affine | 3-bit turbo | 2.75x | recommended default |
| `turbo8v2` | 8-bit affine | 2-bit turbo | 2.32x† | memory-bound long context |
| `turbo4`, `turbo3`, `turbo2` | turbo | turbo | up to 3.4x† | maximum compression; key sensitivity varies strongly by family |

† boundary-layer protection auto-engages for fragile schemes (turbo-quantized
keys or 2-bit values): the first and last two attention layers use 8-bit
affine instead of the turbo scheme.

## Choosing a scheme

Start asymmetric and light, verify output quality on your model, then
compress further:

1. `turbo0v4`: keys untouched. If this is not faithful, the model is
   unusually quantization-sensitive; stop here.
2. `turbo8v4`: 8-bit keys are near-lossless at half the key bytes.
3. `turbo8v3`: the sweet spot for most dense models.
4. `turbo8v2`: memory-bound long context, after validating step 3.
5. Symmetric `turbo4`, `turbo3`, `turbo2`: maximum compression. Key
   compression is where models break. Per-dimension key calibration
   (computed automatically from the prefill cache at compression time)
   equalizes post-rotation key variance and recovers most of the gap on
   sensitive families; sensitivity still varies, so validate on your model
   before deploying.

## Family sensitivity (measured)

WikiText-2 decode-time KL divergence against an fp16 cache with identical
weights, so the numbers isolate cache compression from weight quantization:

- Mistral-class models tolerate even symmetric `turbo4` (KLD 0.040 at 2.8x
  on Mistral-7B).
- qk-norm families (Qwen3) and small models are the most sensitive to key
  compression. Per-dimension key calibration recovers most of it: turbo4
  KLD 2.65 to 0.15 on Qwen3-1.7B, 2.76 to 0.036 on Phi-4-mini, 0.62 to
  0.060 on Qwen2.5-7B, all at about 3.3x. The asymmetric family stays
  healthy with or without calibration.
- Qwen2.5-class models carry large key outliers in their first and last
  layers; attention scores are computed in f32 for this reason, and affine8
  is notably weaker on this family (KLD 0.041) than the turbo value schemes
  (turbo0v4 KLD 0.005).
- Phi-family models are unusually affine-friendly (affine8 KLD 0.0004).
- Symmetric key compression improves with model size (KLD 2.7 at 1.7B, 0.6
  at 7B on the same family).

## Performance

TurboQuant decode runs through JIT-compiled Metal kernels; kernels are
compiled once per model shape, not per call. Single-token decode is a
SIMD-parallel flash kernel (packed, raw fp16, or 8-bit affine key scoring
with inline value dequantization). Measured on Qwen3-1.7B (M5 Max, fp16
150 tok/s): turbo8v3 114, turbo4 122, turbo0v4 102. Prefill stays raw
fp16, so prefill throughput is unaffected. Use TurboQuant when memory is
the constraint (long contexts, larger models per machine); use affine8
when exact fp16-parity decode matters more than footprint.

## Limitations

- Rotating and sliding-window cache layers (most Gemma layers) are not
  converted; their storage is already bounded by the window size and their
  eviction layout does not fit the sequential-append compression path. On
  mixed models the global (non-rotating) layers, where cache memory
  actually grows, do compress, and a one-time notice lists the layers that
  kept fp16 rotating caches. Hybrid recurrent layers are likewise left
  untouched.
- A bounded compressed rotating cache is not implemented.
- For memory estimation with wired limits see <doc:wired-memory>; effective
  bytes per element follow from the table above (for example `turbo8v3` is
  about 0.73 bytes per K/V element pair average against 4 for fp16).
