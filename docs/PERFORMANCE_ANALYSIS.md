# Gemma 4 Swift vs Python Performance Analysis

## Summary

Swift MLX achieves 88% of Python MLX generation speed for Gemma 4 E4B.
The 12% gap is caused by framework-level Metal buffer cache churn, not model code.

## Benchmark Results (Release, warmed up, 128 tokens)

| Metric | Python MLX | Swift MLX | Ratio |
|--------|-----------|-----------|-------|
| Generation tok/s | 75.5 | 66.3 | 0.88x |
| Prompt tok/s | 619.8 | 419.6 | 0.68x |
| Per-token latency | 14.62 ms | 16.41 ms | +1.79 ms |

## Root Cause: Metal Cache Churn

The MLX evaluation engine allocates significantly more temporary Metal
buffers in Swift than in Python for identical computations:

| Model | Python cache/token | Swift cache/token | Ratio |
|-------|-------------------|-------------------|-------|
| Gemma 4 E4B 8-bit | 2.3 MB | 237 MB | 103x |
| Gemma 3 1B QAT 4-bit | 0.9 MB | 11 MB | 12x |

Key observations:
- Cache is 0 MB before MLX.eval(), jumps to 237 MB after
- Graph building (lazy) allocates nothing
- Both Python and Swift use identical MLX C++ core v0.31.1
- Both use the same MLXFast primitives (rmsNorm, scaledDotProductAttention, quantizedMM)
- The gap scales with model size (more layers = more intermediates)

## What's NOT the Cause

Verified through isolation benchmarks:
- Detokenizer overhead (0% impact)
- AsyncStream vs synchronous generation (identical tok/s)
- Missing quantization (all modules correctly QuantizedLinear/QuantizedEmbedding)
- Missing C++ optimizations (both embed same MLX 0.31.1 with SDPA donation + fence fixes)
- Gemma 4 model code specifically (Gemma 3 shows same pattern at 12x)

## Related Upstream Issue

ml-explore/mlx-swift-lm#124 documents the same class of problem for MoE models.
Contributor analysis suggests the gap is in how Swift constructs the computation
graph, possibly due to ARC reference counting preventing buffer donation/aliasing
optimizations in the C++ evaluation engine.

## Model Architecture

The Gemma 4 text model implementation supports:
- 42-layer transformer with sliding + full attention pattern
- Dual head dimensions (256 sliding, 512 full)
- ProportionalRoPE with partial_rotary_factor
- KV cache sharing (18 shared layers)
- Per-layer auxiliary embeddings with gating
- Logit softcapping
- 8-bit quantized inference
