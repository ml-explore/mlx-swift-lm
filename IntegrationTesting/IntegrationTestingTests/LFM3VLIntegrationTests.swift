import CoreImage
import CoreMedia
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
struct LFM3VLIntegrationTests {
    @Test func officialCheckpointImageGeneration() async throws {
        let loader: any TokenizerLoader = #huggingFaceTokenizerLoader()
        let container: ModelContainer
        if let path = ProcessInfo.processInfo.environment["MLX_LFM3_VL_CHECKPOINT"] {
            container = try await VLMModelFactory.shared.loadContainer(
                from: URL(fileURLWithPath: path), using: loader)
        } else {
            container = try await VLMModelFactory.shared.loadContainer(
                from: #hubDownloader(), using: loader,
                configuration: .init(
                    id: "LiquidAI/LFM3-VL-2B-MLX-4bit",
                    revision: "d0d7edaf1b462161720898e46687c6fdaae6caee"))
        }
        let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: 256, height: 256))
        let blue = CIImage(color: CIColor(red: 0, green: 0, blue: 1))
            .cropped(to: CGRect(x: 0, y: 0, width: 256, height: 256))
        let frames = (0 ..< 4).map {
            VideoFrame(
                image: .ciImage(image),
                timeStamp: .init(seconds: Double($0), preferredTimescale: 600))
        }
        var videoInput = UserInput(
            prompt: "What color is visible in this video? Answer with one word.",
            videos: [.frames(frames)], additionalContext: ["enable_thinking": false])
        videoInput.processing.video = .init(targetFrames: 4)
        for (userInput, color) in [
            (
                UserInput(
                    prompt: "What is the color of this image? Answer with one word.",
                    images: [.ciImage(image)], additionalContext: ["enable_thinking": false]), "red"
            ),
            (
                UserInput(
                    prompt: "What is the color of the second image? Answer with one word.",
                    images: [.ciImage(image), .ciImage(blue)],
                    additionalContext: ["enable_thinking": false]), "blue"
            ),
            (videoInput, "red"),
        ] {
            let input = try await container.prepare(input: userInput)
            let stream = try await container.generate(
                input: input,
                parameters: .init(maxTokens: 32, temperature: 0))
            var answer = ""
            for await event in stream {
                if case .chunk(let text) = event { answer += text }
            }
            print("LFM3-VL official checkpoint answer for \(color): \(answer)")
            #expect(answer.lowercased().contains(color))
        }
        if let referencePath = ProcessInfo.processInfo.environment["MLX_LFM3_VL_REFERENCE"] {
            try await container.perform { context in
                context.model.apply { value in
                    value.dtype.isFloatingPoint ? value.asType(.float32) : value
                }
                let reference = try MLX.loadArrays(url: URL(fileURLWithPath: referencePath))
                let tokens = try #require(reference["tokens"])
                let pixels = try #require(reference["pixels"])
                let grid = try #require(reference["grid"]).asArray(Int.self)
                let expected = try #require(reference["logits"])
                let prepared = LMInput(
                    text: .init(tokens: tokens),
                    image: .init(pixels: pixels, frames: [THW(grid[0], grid[1], grid[2])]))
                var previous: MLXArray?
                for step in [32, 512] {
                    let cache = try context.model.newCache(parameters: nil)
                    guard
                        case .logits(let output) = try context.model.prepare(
                            prepared,
                            cache: cache, state: nil, prefill: .init(stepSize: step))
                    else {
                        throw VLMError.processing("Expected LFM3-VL prefill logits")
                    }
                    let error = abs(output.logits.asType(.float32) - expected).max().item(
                        Float.self)
                    print("LFM3-VL reference max logit difference, prefill \(step): \(error)")
                    #expect(
                        argMax(output.logits, axis: -1).item(Int.self)
                            == argMax(expected, axis: -1).item(Int.self))
                    #expect(
                        allClose(output.logits, expected, rtol: 0.001, atol: 0.001).item(Bool.self))
                    if let previous {
                        #expect(
                            allClose(output.logits, previous, rtol: 0.001, atol: 0.001).item(
                                Bool.self))
                    }
                    previous = output.logits
                }
            }
        }
    }
}
