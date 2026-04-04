// Progressive isolation test: add Module components one at a time

import Foundation
import MLX
import MLXNN
import MLXLMCommon

// Test helper: measure cache from a single eval
func measureCache(_ label: String, _ body: () -> MLXArray) {
    MLX.Memory.clearCache()
    let result = body()
    let beforeEval = MLX.Memory.snapshot().cacheMemory
    MLX.eval(result)
    let afterEval = MLX.Memory.snapshot().cacheMemory
    print("  \(label): \(afterEval / 1_000_000) MB (before eval: \(beforeEval / 1_000_000) MB)")
}

func runDonationTest() {
    print("\n=== Progressive Module Isolation Test ===")

    let x = MLXArray.ones([1, 1, 2560])
    let w = MLXArray.ones([2560])
    MLX.eval(x, w)

    // Create quantized weight for Linear tests
    let fullW = MLXRandom.normal([2560, 2560])
    let (qw, scales, biases) = MLX.quantized(fullW, groupSize: 64, bits: 8)
    MLX.eval(qw, scales, biases)

    // ========== Test 1: Raw ops (baseline — expect 0 MB) ==========
    print("\n--- Raw ops (no Module) ---")
    measureCache("10x rmsNorm chain") {
        var h = x
        for _ in 0..<10 {
            h = MLXFast.rmsNorm(h, weight: w, eps: 1e-6)
        }
        return h
    }

    measureCache("10x quantizedMatmul chain") {
        var h = x
        for _ in 0..<10 {
            h = quantizedMatmul(h, qw, scales: scales, biases: biases,
                                transpose: true, groupSize: 64, bits: 8)
        }
        return h
    }

    measureCache("10x layer pattern (norm+qmm+residual)") {
        var h = x
        for _ in 0..<10 {
            let residual = h
            h = MLXFast.rmsNorm(h, weight: w, eps: 1e-6)
            h = quantizedMatmul(h, qw, scales: scales, biases: biases,
                                transpose: true, groupSize: 64, bits: 8)
            h = MLXFast.rmsNorm(h, weight: w, eps: 1e-6)
            h = residual + h
        }
        return h
    }

    // ========== Test 2: QuantizedLinear Module (vs raw quantizedMatmul) ==========
    print("\n--- QuantizedLinear Module ---")
    let qLinear = QuantizedLinear(
        weight: qw, bias: nil, scales: scales, biases: biases,
        groupSize: 64, bits: 8)
    MLX.eval(qLinear)

    measureCache("10x QuantizedLinear.callAsFunction") {
        var h = x
        for _ in 0..<10 {
            h = qLinear(h)
        }
        return h
    }

    // ========== Test 3: MLXNN.RMSNorm Module (vs raw MLXFast.rmsNorm) ==========
    print("\n--- RMSNorm Module ---")
    let rmsNorm = RMSNorm(dimensions: 2560)
    MLX.eval(rmsNorm)

    measureCache("10x RMSNorm Module") {
        var h = x
        for _ in 0..<10 {
            h = rmsNorm(h)
        }
        return h
    }

    // ========== Test 4: Combined Module layer (norm + linear + residual) ==========
    print("\n--- Combined Module layer ---")
    measureCache("10x (RMSNorm + QuantizedLinear + residual)") {
        var h = x
        for _ in 0..<10 {
            let residual = h
            h = rmsNorm(h)
            h = qLinear(h)
            h = rmsNorm(h)
            h = residual + h
        }
        return h
    }

    // ========== Test 5: Custom Module subclass ==========
    print("\n--- Custom Module subclass ---")

    class TestLayer: Module {
        @ModuleInfo(key: "norm") var norm: RMSNorm
        @ModuleInfo(key: "proj") var proj: QuantizedLinear

        init(norm: RMSNorm, proj: QuantizedLinear) {
            self._norm.wrappedValue = norm
            self._proj.wrappedValue = proj
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let residual = x
            var h = norm(x)
            h = proj(h)
            h = norm(h)
            return residual + h
        }
    }

    let testLayer = TestLayer(norm: rmsNorm, proj: qLinear)
    MLX.eval(testLayer)

    measureCache("10x TestLayer(Module) calls") {
        var h = x
        for _ in 0..<10 {
            h = testLayer(h)
        }
        return h
    }

    // ========== Test 6: Array of Module layers (like model.layers) ==========
    print("\n--- Array of Module layers ---")
    let layers = (0..<10).map { _ in TestLayer(norm: rmsNorm, proj: qLinear) }
    MLX.eval(layers)

    measureCache("10 different TestLayer instances") {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }

    // ========== Test 7: KV Cache update ==========
    print("\n--- KV Cache ---")
    let cache = RotatingKVCache(maxSize: 512, keep: 0)

    // Prefill cache with some data
    let kInit = MLXArray.ones([1, 2, 5, 256])
    let vInit = MLXArray.ones([1, 2, 5, 256])
    let _ = cache.update(keys: kInit, values: vInit)
    MLX.eval(cache)

    measureCache("10x KV cache updates (single token)") {
        var lastK = MLXArray.ones([1, 2, 1, 256])
        var lastV = MLXArray.ones([1, 2, 1, 256])
        for _ in 0..<10 {
            let (k, v) = cache.update(keys: lastK, values: lastV)
            lastK = k[.ellipsis, (-1)..., 0...]
            lastV = v[.ellipsis, (-1)..., 0...]
        }
        return lastK
    }

    // ========== Test 8: Module layer + KV Cache together ==========
    print("\n--- Module + KV Cache combined ---")
    let cache2 = RotatingKVCache(maxSize: 512, keep: 0)
    let _ = cache2.update(keys: kInit, values: vInit)
    MLX.eval(cache2)

    measureCache("10x (Module layer + cache update)") {
        var h = x
        for layer in layers {
            h = layer(h)
            // Simulate attention cache update
            let keys = h.reshaped(1, 2, 1, -1)[.ellipsis, ..<256]
            let vals = h.reshaped(1, 2, 1, -1)[.ellipsis, ..<256]
            let (_, _) = cache2.update(keys: keys, values: vals)
        }
        return h
    }

    print("\n--- Summary ---")
    print("If any test above shows >0 MB, that component is the source of cache churn.")
}
