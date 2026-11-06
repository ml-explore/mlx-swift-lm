import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class LFM3CheckpointTests: XCTestCase {
    private var usesReducedFloat32: Bool {
        let a = withRandomState(MLXRandom.RandomState(seed: 0)) { MLXRandom.normal([64, 512]) }
        let b = withRandomState(MLXRandom.RandomState(seed: 1)) { MLXRandom.normal([512, 64]) }
        let cpu = matmul(a, b, stream: .cpu)
        let gpu = matmul(a, b, stream: .gpu)
        return abs(gpu - cpu).max().item(Float.self) / abs(cpu).max().item(Float.self) > 1e-5
    }

    private func loadModel() throws -> LFM3Model {
        guard let path = ProcessInfo.processInfo.environment["MLX_LFM3_CHECKPOINT"] else {
            throw XCTSkip("Set MLX_LFM3_CHECKPOINT to a local published LFM3 checkpoint")
        }
        let directory = URL(filePath: path)
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let config = try JSONDecoder().decode(LFM3Configuration.self, from: data)
        let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let start = Date.timeIntervalSinceReferenceDate
        let model = LFM3Model(config)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)
        eval(model)
        print("LFM3 checkpoint load_seconds=\(Date.timeIntervalSinceReferenceDate - start)")
        return model
    }

    func testPublishedCheckpointPrefillAndDecode() throws {
        let model = try loadModel()
        Memory.peakMemory = 0
        let count = 1024
        let tokens = MLXArray((0 ..< count).map { Int32(($0 * 7 + 3) % model.vocabularySize) })
            .reshaped(1, count)
        let cache = try model.newCache(parameters: nil)
        let start = Date.timeIntervalSinceReferenceDate
        var output: MLXArray?
        for offset in stride(from: 0, to: count, by: 256) {
            output =
                model.nextTokenLogits(
                    .init(tokens: tokens[0..., offset ..< offset + 256]), cache: cache, state: nil
                ).logits
            eval(output!)
            eval(cache)
        }
        var logits = try XCTUnwrap(output)[0..., (-1)...]
        eval(logits)
        let prefillSeconds = Date.timeIntervalSinceReferenceDate - start
        let decodeStart = Date.timeIntervalSinceReferenceDate
        for _ in 0 ..< 16 {
            let next = argMax(logits[0, -1]).reshaped(1, 1)
            logits = model(next, cache: cache)
            eval(logits)
        }
        let decodeSeconds = Date.timeIntervalSinceReferenceDate - decodeStart
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
        print(
            "LFM3 checkpoint prefill_tok_s=\(Double(count) / prefillSeconds), decode_tok_s=\(16 / decodeSeconds), peak_bytes=\(Memory.peakMemory)"
        )
    }

    func testPublishedCheckpointReferenceParity() throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_LFM3_REFERENCE_FILE"] else {
            throw XCTSkip("Set MLX_LFM3_REFERENCE_FILE to the matching FP32 reference outputs")
        }
        guard !usesReducedFloat32 else {
            throw XCTSkip("Set MLX_ENABLE_TF32=0 to compare the published model in IEEE FP32")
        }
        let reference = try loadArrays(url: URL(filePath: path))
        let model = try loadModel()
        model.apply { $0.dtype.isFloatingPoint ? $0.asType(.float32) : $0 }
        for length in [19, 257, 1024] {
            let tokens = try XCTUnwrap(reference["tokens_\(length)"])
            let cache = try model.newCache(parameters: nil)
            var output: MLXArray?
            for start in stride(from: 0, to: length, by: 127) {
                output =
                    model.nextTokenLogits(
                        .init(tokens: tokens[0..., start ..< min(start + 127, length)]),
                        cache: cache, state: nil
                    ).logits
                eval(output!)
                eval(cache)
            }
            let logits = try XCTUnwrap(output)[0..., (-1)...]
            let expected = try XCTUnwrap(reference["logits_\(length)"])
            let error = abs(logits - expected).max().item(Float.self)
            print(
                "LFM3 published FP32 \(length): logits_max_abs=\(error)"
            )
            XCTAssertLessThan(error, 2e-4)
            XCTAssertEqual(
                argMax(logits, axis: -1).item(Int.self), argMax(expected, axis: -1).item(Int.self))
        }
    }

    private func tinyConfiguration(
        tieWordEmbeddings: Bool = true, intermediateSize: Int = 48
    ) throws -> LFM3Configuration {
        try JSONDecoder().decode(
            LFM3Configuration.self,
            from: Data(
                """
                {
                    "model_type": "lfm3", "vocab_size": 64, "hidden_size": 32,
                    "intermediate_size": \(intermediateSize), "num_hidden_layers": 6,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                    "norm_eps": 1e-5, "conv_L_cache": 3, "conv_bias": false,
                    "sliding_window": 4, "use_sandwich_norm": true,
                    "tie_word_embeddings": \(tieWordEmbeddings),
                    "layer_types": ["conv", "sliding_attention", "full_attention",
                                    "conv", "sliding_attention", "full_attention"]
                }
                """.utf8))
    }

    private func tinyModel(_ configuration: LFM3Configuration) -> LFM3Model {
        withRandomState(MLXRandom.RandomState(seed: 7)) { LFM3Model(configuration) }
    }

    func testFactoryRejectsIncompatibleTokenizer() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(
            component: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try tinyConfiguration()
        try JSONEncoder().encode(config).write(to: directory.appending(component: "config.json"))
        let model = tinyModel(config)
        try save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            url: directory.appending(component: "model.safetensors"))
        do {
            _ = try await LLMModelFactory.shared.load(
                from: directory, using: ProbeTokenizerLoader())
            XCTFail("The factory accepted a tokenizer that ignores ignore_merges")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("ignore_merges"), "\(error)")
        }
    }

    private struct ProbeTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer {
            ProbeTokenizer(broken: ".reduce")
        }
    }

    private struct ProbeTokenizer: Tokenizer {
        var broken: String? = nil
        var hasProbeVocabulary = true
        let bosToken: String? = nil
        let eosToken: String? = nil
        let unknownToken: String? = nil
        private let vocabulary = [
            "<|startoftext|>": 124894, ".reduce": 96744, "line": 1421, ".čĊ": 64122, "This": 1942,
        ]

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            if text == broken { return [1, 2] }
            return text == ".reduce" ? [96744] : [1421, 64122, 1942]
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? {
            hasProbeVocabulary ? vocabulary[token] : nil
        }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

}
