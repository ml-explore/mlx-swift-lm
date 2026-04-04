// Minimal buffer donation test: does a simple chain of ops reuse buffers?

import Foundation
import MLX
import MLXNN

func runDonationTest() {
    print("\n=== Buffer Donation Test ===")

    // Test 1: Chain of 10 rmsNorm with var reassignment
    let x = MLXArray.ones([1, 1, 2560])
    let w = MLXArray.ones([2560])
    MLX.eval(x, w)
    MLX.Memory.clearCache()

    var h = x
    for _ in 0..<10 {
        h = MLXFast.rmsNorm(h, weight: w, eps: 1e-6)
    }

    let beforeNorm = MLX.Memory.snapshot().cacheMemory
    MLX.eval(h)
    let afterNorm = MLX.Memory.snapshot().cacheMemory
    print("RMSNorm var-chain: before=\(beforeNorm/1_000_000) MB, after=\(afterNorm/1_000_000) MB  (Python: 0 MB)")

    // Test 2: Same chain but with let (all intermediates alive)
    MLX.Memory.clearCache()
    let n0 = MLXFast.rmsNorm(x, weight: w, eps: 1e-6)
    let n1 = MLXFast.rmsNorm(n0, weight: w, eps: 1e-6)
    let n2 = MLXFast.rmsNorm(n1, weight: w, eps: 1e-6)
    let n3 = MLXFast.rmsNorm(n2, weight: w, eps: 1e-6)
    let n4 = MLXFast.rmsNorm(n3, weight: w, eps: 1e-6)
    let n5 = MLXFast.rmsNorm(n4, weight: w, eps: 1e-6)
    let n6 = MLXFast.rmsNorm(n5, weight: w, eps: 1e-6)
    let n7 = MLXFast.rmsNorm(n6, weight: w, eps: 1e-6)
    let n8 = MLXFast.rmsNorm(n7, weight: w, eps: 1e-6)
    let n9 = MLXFast.rmsNorm(n8, weight: w, eps: 1e-6)
    MLX.eval(n9)
    let afterLet = MLX.Memory.snapshot().cacheMemory
    print("RMSNorm let-chain: after=\(afterLet/1_000_000) MB  (expect higher if ARC prevents donation)")

    // Test 3: Matmul chain with var
    MLX.Memory.clearCache()
    let weight = MLXArray.ones([10240, 2560])
    MLX.eval(weight)
    MLX.Memory.clearCache()

    h = x
    for _ in 0..<10 {
        h = matmul(h, weight.transposed())
        h = h[.ellipsis, ..<2560]
    }
    let beforeMM = MLX.Memory.snapshot().cacheMemory
    MLX.eval(h)
    let afterMM = MLX.Memory.snapshot().cacheMemory
    print("Matmul var-chain: before=\(beforeMM/1_000_000) MB, after=\(afterMM/1_000_000) MB  (Python: 0 MB)")

    // Test 4: Residual connection pattern (x used twice)
    MLX.Memory.clearCache()
    h = MLXFast.rmsNorm(x, weight: w, eps: 1e-6)
    h = x + h  // residual
    MLX.eval(h)
    print("Residual pattern: cache = \(MLX.Memory.snapshot().cacheMemory / 1_000_000) MB  (Python: 0 MB)")

    // Test 5: Quantized matmul chain
    MLX.Memory.clearCache()
    let fullW = MLXArray.ones([10240, 2560])
    let (qw, scales, biases) = MLX.quantized(fullW, groupSize: 64, bits: 8)
    MLX.eval(qw, scales, biases)
    MLX.Memory.clearCache()

    h = x
    for _ in 0..<10 {
        h = quantizedMatmul(h, qw, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 8)
        h = h[.ellipsis, ..<2560]
    }
    let beforeQMM = MLX.Memory.snapshot().cacheMemory
    MLX.eval(h)
    let afterQMM = MLX.Memory.snapshot().cacheMemory
    print("Quantized matmul chain: before=\(beforeQMM/1_000_000) MB, after=\(afterQMM/1_000_000) MB  (Python: 0 MB)")

    // Test 6: Full layer pattern (norm + matmul + residual + norm + matmul + residual)
    MLX.Memory.clearCache()
    h = x
    for _ in 0..<10 {
        // Attention-like: residual + norm + proj + norm + residual
        var residual = h
        h = MLXFast.rmsNorm(h, weight: w, eps: 1e-6)
        h = quantizedMatmul(h, qw, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 8)
        h = h[.ellipsis, ..<2560]
        h = MLXFast.rmsNorm(h, weight: w, eps: 1e-6)
        h = residual + h

        // MLP-like
        residual = h
        h = MLXFast.rmsNorm(h, weight: w, eps: 1e-6)
        h = quantizedMatmul(h, qw, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 8)
        h = h[.ellipsis, ..<2560]
        h = MLXFast.rmsNorm(h, weight: w, eps: 1e-6)
        h = residual + h
    }
    let beforeFull = MLX.Memory.snapshot().cacheMemory
    MLX.eval(h)
    let afterFull = MLX.Memory.snapshot().cacheMemory
    print("Full layer pattern (10 layers): before=\(beforeFull/1_000_000) MB, after=\(afterFull/1_000_000) MB")
}
