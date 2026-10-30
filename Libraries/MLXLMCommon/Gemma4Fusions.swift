import MLX
import MLXNN

// MARK: - Compiled fusion fragments
//
// Fuses (residual + RMSNorm(x) * weight) and gelu(g) * other into single
// compiled graphs, mirroring upstream mlx-lm's gemma4_text.py optimization.
// All Gemma 4 checkpoints share one rms_norm_eps (1e-6), so hardcoding it
// lets one graph serve every layer; callers assert it with
// gemma4RequireFusedNormEps.

/// RMSNorm epsilon baked into the Gemma 4 fusion graphs.
package let gemma4RMSEps: Float = 1e-6

/// Fails when a checkpoint's eps would reach the fused graphs with the
/// wrong constant.
package func gemma4RequireFusedNormEps(_ eps: Float) {
    precondition(
        eps == gemma4RMSEps,
        "Gemma4 fused decode path requires rmsNormEps == \(gemma4RMSEps), got \(eps)")
}

/// Fused `residual + RMSNorm(x) * weight`.
package let gemma4AddRMSNorm: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { residual, x, weight in
    residual + MLXFast.rmsNorm(x, weight: weight, eps: gemma4RMSEps)
}

/// Fused `geluApproximate(gate) * other`.
package let gemma4GeluMul: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, other in
    geluApproximate(gate) * other
}

private let compiledGemma4Softcap = compile(shapeless: true) {
    (logits: MLXArray, cap: MLXArray) in tanh(logits / cap) * cap
}

package func gemma4LogitSoftcap(_ logits: MLXArray, _ cap: Float) -> MLXArray {
    guard Device.defaultDevice().deviceType == .gpu else { return tanh(logits / cap) * cap }
    // Swift scalars use the logits' dtype; a float32 tensor would promote low-precision logits.
    return compiledGemma4Softcap(logits, cap.asMLXArray(dtype: logits.dtype))
}

package func gemma4LogitSoftcap(_ logits: MLXArray, _ cap: MLXArray) -> MLXArray {
    guard Device.defaultDevice().deviceType == .gpu, cap.dtype == .float32 else {
        return tanh(logits / cap) * cap
    }
    // Promote before tracing: mixed float16/float32 division can round differently inside compile.
    if logits.dtype == .float16 || logits.dtype == .bfloat16 {
        return compiledGemma4Softcap(logits.asType(.float32), cap)
    }
    return compiledGemma4Softcap(logits, cap)
}
