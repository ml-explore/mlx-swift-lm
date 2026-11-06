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
struct LFM2VLComponentIntegrationTests {
    @Test func firstImageMatchesImmediateLoading() async throws {
        let downloader: any Downloader = #hubDownloader()
        let tokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()
        let context = try await VLMModelFactory.shared.load(
            from: downloader, using: tokenizerLoader,
            configuration: .init(
                id: "mlx-community/LFM2-VL-1.6B-4bit",
                revision: "be587c055231b0905e846f2c99803dd6d7f33a7a"))
        let model = try #require(context.model as? LFM2VL)
        let holder = try #require(model.onDemandComponents.first)
        let parameters = model.parameters().flattened().map(\.0)
        let textWeightBytes = model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
        #expect(!holder.isLoaded)
        #expect(!parameters.contains { $0.hasPrefix("vision_tower.") })

        let text = try await context.processor.prepare(input: UserInput(prompt: "Hello"))
        let textCache = try model.newCache(parameters: nil)
        guard
            case .logits(let textOutput) = try model.prepare(
                text, cache: textCache, state: nil, prefill: .init())
        else { throw VLMError.processing("Expected text logits") }
        eval(textOutput.logits)
        #expect(!holder.isLoaded)

        let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))
        let input = try await context.processor.prepare(
            input: UserInput(prompt: "What color is the image?", images: [.ciImage(image)]))
        let lazyCache = try model.newCache(parameters: nil)
        guard
            case .logits(let lazyOutput) = try model.prepare(
                input, cache: lazyCache, state: nil, prefill: .init())
        else { throw VLMError.processing("Expected image logits") }
        eval(lazyOutput.logits)
        #expect(holder.isLoaded)
        #expect(model.parameters().flattened().map(\.0) == parameters)

        guard case .directory(let directory) = context.configuration.id else {
            throw VLMError.processing("Expected a resolved model directory")
        }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        let eager = LFM2VL(model.config)
        try await loadWeights(
            modelDirectory: directory, model: eager,
            perLayerQuantization: base.perLayerQuantization,
            componentLoading: .immediate)
        let fullWeightBytes = eager.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
        #expect(fullWeightBytes > textWeightBytes)
        print(
            "LFM2VL checkpoint weights: text \(textWeightBytes) bytes, full \(fullWeightBytes) bytes"
        )
        let eagerCache = try eager.newCache(parameters: nil)
        guard
            case .logits(let eagerOutput) = try eager.prepare(
                input, cache: eagerCache, state: nil, prefill: .init())
        else { throw VLMError.processing("Expected immediate image logits") }
        eval(eagerOutput.logits)
        #expect(allClose(lazyOutput.logits, eagerOutput.logits).item(Bool.self))
        #expect(
            argMax(lazyOutput.logits, axis: -1).asArray(Int.self)
                == argMax(eagerOutput.logits, axis: -1).asArray(Int.self))
    }
}
