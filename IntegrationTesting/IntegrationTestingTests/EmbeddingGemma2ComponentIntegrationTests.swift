// Copyright © 2026 Apple Inc.

import CoreImage
import CoreMedia
import Foundation
import HuggingFace
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
import Tokenizers

@Suite(.serialized)
struct EmbeddingGemma2ComponentIntegrationTests {
    @Test
    func realCheckpointMatchesImmediateLoadingAcrossModalities() async throws {
        let resolved = try await resolve(
            configuration: ModelConfiguration(id: "mlx-community/embeddinggemma-2-8bit"),
            from: #hubDownloader(), useLatest: false, progressHandler: { _ in })
        let before = Memory.activeMemory
        let start = Date()
        let lazy = try await MultimodalEmbeddingFactory.shared.loadContainer(
            from: resolved.modelDirectory, using: #huggingFaceTokenizerLoader())
        let lazyBytes = Memory.activeMemory - before
        let lazySeconds = Date().timeIntervalSince(start)
        let beforeImmediate = Memory.activeMemory
        let immediateStart = Date()
        let immediate = try await EmbeddingGemma2Embedding(
            modelDirectory: resolved.modelDirectory, tokenizerLoader: #huggingFaceTokenizerLoader(),
            componentLoading: .immediate)
        let immediateBytes = Memory.activeMemory - beforeImmediate
        print(
            "EmbeddingGemma 2 load: on-demand \(lazySeconds)s / \(lazyBytes) live MLX bytes; immediate \(Date().timeIntervalSince(immediateStart))s / \(immediateBytes) additional live MLX bytes"
        )
        #expect(lazyBytes < immediateBytes)
        #expect(await lazy.supportedModalities == [.text, .image, .video, .audio])

        for kind in Kind.allCases {
            print("EmbeddingGemma 2: immediate \(kind)")
            let expected = try await immediate.embed(Self.input(kind), task: .document)
            print("EmbeddingGemma 2: on-demand \(kind)")
            let actual = try await lazy.embed([Self.input(kind)], task: .document)[0]
            #expect(actual.count == 768 && actual.allSatisfy { $0.isFinite })
            #expect(zip(actual, expected).allSatisfy { abs($0 - $1) < 1e-5 }, "\(kind)")
        }

        // The factory path also keeps media outside the text model's module tree.
        print("EmbeddingGemma 2: text factory")
        let context = try await EmbedderModelFactory.shared.load(
            from: resolved.modelDirectory, using: #huggingFaceTokenizerLoader())
        let model = try #require(context.model as? MLXLMCommon.EmbeddingGemma2)
        #expect(model.parameters().flattened().allSatisfy { $0.0.hasPrefix("language_model.") })
        #expect(model.onDemandComponents.allSatisfy { !$0.isLoaded })
    }

    private enum Kind: CaseIterable { case text, image, video, audio, interleaved }

    private static func input(_ kind: Kind) -> EmbeddingGemma2Embedding.Input {
        let image = UserInput.Image.ciImage(
            CIImage(color: CIColor(red: 0.15, green: 0.45, blue: 0.8))
                .cropped(to: CGRect(x: 0, y: 0, width: 128, height: 96)))
        func audio() -> UserInput.Audio {
            let step = 2 * Double.pi * 440 / 16_000
            let samples: [Float] = (0 ..< 16_000).map { Float(sin(Double($0) * step)) * 0.1 }
            return .array(MLXArray(samples))
        }
        switch kind {
        case .text:
            return .init(text: "The sky is blue on a clear day.")
        case .image:
            return .init([.image(image)])
        case .video:
            return .init([
                .video(
                    .frames([
                        .init(image: image, timeStamp: .zero),
                        .init(image: image, timeStamp: CMTime(seconds: 1, preferredTimescale: 600)),
                    ]))
            ])
        case .audio:
            return .init([.audio(audio())])
        case .interleaved:
            return .init([.text("A blue image and a tone: "), .image(image), .audio(audio())])
        }
    }
}
