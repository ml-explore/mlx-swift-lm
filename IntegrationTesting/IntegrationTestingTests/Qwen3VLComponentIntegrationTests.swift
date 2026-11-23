// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
import Tokenizers

@Suite(.serialized)
struct Qwen3VLComponentIntegrationTests {
    /// The real quantized backbone must produce identical hidden states under both load policies.
    @Test func onDemandVisionMatchesImmediateOnQuantizedCheckpoint() async throws {
        let directory = try await #hubDownloader().download(
            id: "mlx-community/Qwen3-VL-4B-Instruct-4bit",
            revision: "2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b",
            matching: ["*.safetensors", "*.json", "*.jinja"], useLatest: false,
            progressHandler: { _ in })
        let context = try await VLMModelFactory.shared.load(
            from: directory, using: #huggingFaceTokenizerLoader())
        let model = try #require(context.model as? Qwen3VL)
        let holder = try #require(model.onDemandComponents.first)
        let paths = model.parameters().flattened().map(\.0)
        let textBytes = model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
        #expect(!holder.isLoaded)
        #expect(paths.allSatisfy { $0.hasPrefix("language_model.") })
        let text = try await context.processor.prepare(input: UserInput(prompt: "A red bicycle."))
        let textReference = try model.hiddenStates(text)[0, -1, 0...].asArray(Float.self)
        #expect(!holder.isLoaded)
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 56, height: 56))
        let visual = try await context.processor.prepare(
            input: UserInput(chat: [.user("A red square.", images: [.ciImage(image)])]))
        let visualReference = try model.hiddenStates(visual)[0, -1, 0...].asArray(Float.self)
        #expect(holder.isLoaded)
        #expect(model.parameters().flattened().map(\.0) == paths)
        let base = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization,
            componentLoading: .immediate)
        #expect(try model.hiddenStates(text)[0, -1, 0...].asArray(Float.self) == textReference)
        #expect(try model.hiddenStates(visual)[0, -1, 0...].asArray(Float.self) == visualReference)
        let fullBytes = model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
        #expect(fullBytes > textBytes)
        print("Qwen3VL text weights: \(textBytes) bytes; immediate weights: \(fullBytes) bytes")
    }
}
