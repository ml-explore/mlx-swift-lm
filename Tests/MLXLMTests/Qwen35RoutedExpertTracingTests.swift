import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Reads the expert ids on the CPU, as a module that pages experts in would. Illegal inside a compiled trace.
private final class HostReadSwitchGLU: SwitchGLU, UntraceableModule {
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

    private func replaceExperts(
        in model: Qwen35TextModel, _ make: (SwitchGLU) throws -> SwitchGLU
    ) throws -> [SwitchGLU] {
        var replacements: [SwitchGLU] = []
        for layer in model.model.layers {
            guard let moe = layer.mlp as? Qwen35SparseMoeBlock else { continue }
            let replacement = try make(moe.switchMLP)
            moe.update(modules: ModuleChildren.unflattened(["switch_mlp": replacement]))
            replacements.append(replacement)
        }
        return replacements
    }

    private func assertSameLogits(
        _ a: [MLXArray], _ b: [MLXArray], file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.count, b.count, file: file, line: line)
        for (x, y) in zip(a, b) {
            XCTAssertTrue(
                allClose(x, y, rtol: 1e-5, atol: 1e-5).item(Bool.self), file: file, line: line)
        }
    }

    func testUntraceableExpertsSplitTheCompiledDecodeStep() throws {
        MLXRandom.seed(11)
        let configuration = try configuration()
        let model = Qwen35TextModel(configuration)
        eval(model)
        let traced = try decodeLogits(model)
        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 2)

        let stock = model.model.layers.compactMap { ($0.mlp as? Qwen35SparseMoeBlock)?.switchMLP }
        let paging = try replaceExperts(in: model) { original in
            let replacement = HostReadSwitchGLU(
                inputDims: configuration.hiddenSize,
                hiddenDims: configuration.moeIntermediateSize,
                numExperts: configuration.numExperts)
            try replacement.update(parameters: original.parameters(), verify: .all)
            return replacement
        }
        let untraced = try decodeLogits(model)

        // Both MoE blocks became boundaries: 4 compiled segments instead of 2,
        // with the routed experts run eagerly between them.
        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 4)
        XCTAssertTrue(paging.allSatisfy { ($0 as! HostReadSwitchGLU).hostReads > 0 })
        assertSameLogits(traced, untraced)

        // Swapping the stock experts back rebuilds the schedule instead of reusing stale traces.
        var index = 0
        _ = try replaceExperts(in: model) { _ in
            defer { index += 1 }
            return stock[index]
        }
        let restored = try decodeLogits(model)
        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 2)
        assertSameLogits(traced, restored)
    }

    func testScheduleClosesAtEagerExperts() {
        let stock = CompiledDecodeSegment.schedule(linearLayers: [true, false])
        XCTAssertEqual(
            CompiledDecodeSegment.schedule(
                linearLayers: [true, false], eagerExperts: [false, false]),
            stock)

        XCTAssertEqual(
            CompiledDecodeSegment.schedule(
                linearLayers: [true, false], eagerExperts: [true, true]),
            [
                CompiledDecodeSegment(linearLayers: [0], expertsHeadLayer: 0),
                CompiledDecodeSegment(expertsTailLayer: 0, attentionPreLayer: 1),
                CompiledDecodeSegment(attentionPostLayer: 1, expertsHeadLayer: 1),
                CompiledDecodeSegment(expertsTailLayer: 1),
            ])
    }
}
