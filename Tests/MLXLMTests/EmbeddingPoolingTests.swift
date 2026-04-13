import Foundation
import MLX
import Testing

@testable import MLXEmbedders

struct EmbeddingPoolingTests {

    @Test("Last-token pooling uses the final non-padding token")
    func testLastPoolingRespectsMask() {
        let hiddenStates = MLXArray([
            1.0 as Float, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            10.0, 20.0,
            30.0, 40.0,
            50.0, 60.0,
        ]).reshaped(2, 3, 2)
        let mask =
            MLXArray([
                1 as Int32, 1, 0,
                1, 1, 1,
            ]).reshaped(2, 3) .== MLXArray(Int32(1))

        let pooled = Pooling(strategy: .last)(
            EmbeddingModelOutput(hiddenStates: hiddenStates, pooledOutput: nil),
            mask: mask
        )
        pooled.eval()

        #expect(pooled.shape == [2, 2])
        #expect(pooled[0].asArray(Float.self) == [3.0, 4.0])
        #expect(pooled[1].asArray(Float.self) == [50.0, 60.0])
    }

    @Test("Qwen3 falls back to model-defined pooling when 1_Pooling metadata is missing")
    func testQwen3FallbackPoolingStrategy() throws {
        let config = try JSONDecoder().decode(
            Qwen3Configuration.self,
            from: Data(
                """
                {
                  "hidden_size": 1024,
                  "num_hidden_layers": 1,
                  "intermediate_size": 1536,
                  "num_attention_heads": 16,
                  "rms_norm_eps": 0.000001,
                  "vocab_size": 1024,
                  "num_key_value_heads": 8,
                  "head_dim": 64
                }
                """.utf8))
        let model = Qwen3Model(config)
        let modelDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)

        let pooling = loadPooling(modelDirectory: modelDirectory, model: model)

        #expect(pooling.strategy == .last)
        #expect(pooling.dimension == nil)
    }
}
