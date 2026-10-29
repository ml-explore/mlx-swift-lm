import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Reads the expert ids on the CPU, as a module that pages experts in would. Illegal inside a compiled trace.
private final class HostReadSwitchGLU: SwitchGLU {
    var hostReads = 0

    override func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        _ = indices.asType(.uint32).asArray(UInt32.self)
        hostReads += 1
        return super.callAsFunction(x, indices)
    }
}

final class Qwen35RoutedExpertTracingTests: XCTestCase {
    private func configuration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 128,
                "full_attention_interval": 2,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(json.utf8))
    }

    /// Logits of three greedy single-token decode steps after a short prefill.
    private func decodeLogits(_ model: Qwen35TextModel) throws -> [MLXArray] {
        let cache = try model.newCache(parameters: nil)
        var logits = model(MLXArray([1, 2, 3, 4, 5] as [Int32])[.newAxis], cache: cache)
        var steps: [MLXArray] = []
        for _ in 0 ..< 3 {
            let next = argMax(logits[0..., -1, 0...], axis: -1).asType(.int32)
            logits = model(next[.newAxis], cache: cache)
            eval(logits)
            steps.append(logits)
        }
        return steps
    }

    func testDecodeWithUntracedRoutedExpertsMatchesTracedDecode() throws {
        MLXRandom.seed(11)
        let configuration = try configuration()
        let model = Qwen35TextModel(configuration)
        eval(model)
        XCTAssertTrue(model.tracesRoutedExperts)
        let traced = try decodeLogits(model)

        var replacements: [HostReadSwitchGLU] = []
        for layer in model.model.layers {
            guard let moe = layer.mlp as? Qwen35SparseMoeBlock else { continue }
            let replacement = HostReadSwitchGLU(
                inputDims: configuration.hiddenSize,
                hiddenDims: configuration.moeIntermediateSize,
                numExperts: configuration.numExperts)
            try replacement.update(parameters: moe.switchMLP.parameters(), verify: .all)
            moe.update(modules: ModuleChildren.unflattened(["switch_mlp": replacement]))
            replacements.append(replacement)
        }
        model.tracesRoutedExperts = false
        XCTAssertFalse(model.tracesRoutedExperts)
        let untraced = try decodeLogits(model)

        XCTAssertEqual(replacements.count, 2)
        XCTAssertTrue(replacements.allSatisfy { $0.hostReads > 0 })
        for (a, b) in zip(traced, untraced) {
            XCTAssertTrue(allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }
}
