// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import MLXScriptedLM
import XCTest

@testable import MLXLMCommon

final class ChatSessionObservationPerformanceTests: XCTestCase {
    func testDefaultGenerationPerformance() async throws {
        MLXRandom.seed(42)
        let tokenizer = PseudoWordTokenizer(vocabularySize: 32768)
        let processor = TestInputProcessor(
            tokenizer: tokenizer, configuration: .init(id: "observation-benchmark"),
            messageGenerator: DefaultMessageGenerator())
        let model = Gemma3TextModel(
            Gemma3TextConfiguration(
                modelType: "text", hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64,
                attentionHeads: 4, headDim: 64, rmsNormEps: 0.00001, vocabularySize: 32768,
                kvHeads: 4, ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
                ropeTraditional: false, queryPreAttnScalar: 256, slidingWindow: 512,
                slidingWindowPattern: 6, maxPositionEmbeddings: 32768))
        eval(model)
        let container = ModelContainer(
            context: ModelContext(
                configuration: processor.configuration, model: model,
                processor: processor, tokenizer: tokenizer))
        var decodeTimes: [Double] = []
        var firstTokenTimes: [Double] = []
        var peaks: [Int] = []
        var reference: String?
        for round in 0 ..< 7 {
            let session = ChatSession(
                container, generateParameters: GenerateParameters(maxTokens: 96, temperature: 0))
            Memory.peakMemory = 0
            let start = ContinuousClock.now
            var firstToken: Duration?
            var text = ""
            var info: GenerateCompletionInfo?
            for try await event in session.streamDetails(to: "hello world") {
                if let chunk = event.chunk {
                    if firstToken == nil { firstToken = start.duration(to: .now) }
                    text += chunk
                }
                if let completion = event.info { info = completion }
            }
            let completion = try XCTUnwrap(info)
            XCTAssertEqual(completion.generationTokenCount, 96)
            if let reference { XCTAssertEqual(text, reference) } else { reference = text }
            if round > 0 {
                peaks.append(Memory.peakMemory)
                decodeTimes.append(completion.generateTime)
                let duration = try XCTUnwrap(firstToken).components
                firstTokenTimes.append(
                    Double(duration.seconds) + Double(duration.attoseconds) / 1e18)
            }
        }
        print(
            "CHAT_SESSION_BENCHMARK decode_seconds=\(decodeTimes.sorted()[decodeTimes.count / 2]) first_token_seconds=\(firstTokenTimes.sorted()[firstTokenTimes.count / 2])"
        )
        print("CHAT_SESSION_PEAK_BYTES \(peaks.sorted()[peaks.count / 2])")
        print("CHAT_SESSION_TOKENS \(tokenizer.encode(text: try XCTUnwrap(reference)))")
    }
}
