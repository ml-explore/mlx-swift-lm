// Copyright © 2026 Apple Inc.
//
// `Qwen35`'s `PreparedInputSplitting` conformance, on tiny random-weight models.
// The split must cut a prepared input so that prefilling the suffix on a warm
// cache lands where a cold prefill of the whole prompt does.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Qwen35PreparedInputSplittingTests: XCTestCase {

    // Token ids from `Qwen35ContinuationTests.makeTinyModel()`.
    private static let imageTokenId = 500
    private static let visionStartTokenId = 502
    private static let visionEndTokenId = 503

    private let continuation = ContinuationAssertions(
        imageTokenId: Int32(imageTokenId), visionStartTokenId: Int32(visionStartTokenId))

    // MARK: - Model level

    func testSplitMediaSuffixContinuationMatchesFullPrefill() throws {
        try continuation.assertSplitMediaSuffixContinuation(
            Qwen35ContinuationTests.makeTinyModel())
    }

    /// The MoE models load as `Qwen35MoE`, which inherits the conformance.
    func testMoESplitMediaSuffixContinuationMatchesFullPrefill() throws {
        try continuation.assertSplitMediaSuffixContinuation(
            Qwen35ContinuationTests.makeTinyMoEModel())
    }

    /// Two images with the same grid have the same row and placeholder counts, so only
    /// their pixels tell them apart. The suffix must carry the second image's rows.
    func testSplitKeepsTheSuffixImageWhenGridsMatch() throws {
        let model = try Qwen35ContinuationTests.makeTinyModel()
        try withRandomState(MLXRandom.RandomState(seed: 23)) {
            let imageA = continuation.image()
            let imageB = continuation.image()
            let t1 = concatenated(
                [continuation.textTokens(10), continuation.imageRun(), continuation.textTokens(6)],
                axis: 1)
            let t2 = concatenated(
                [
                    continuation.textTokens(4, seed: 8), continuation.imageRun(),
                    continuation.textTokens(5, seed: 9),
                ], axis: 1)
            let full = LMInput(
                text: .init(tokens: concatenated([t1, t2], axis: 1)),
                image: .init(
                    pixels: concatenated([imageA.pixels, imageB.pixels], axis: 0),
                    frames: [THW(1, 4, 4), THW(1, 4, 4)]))

            let suffix = try XCTUnwrap(model.splitPreparedInput(full, droppingFirst: t1.dim(1)))
            let suffixImage = try XCTUnwrap(suffix.image)
            XCTAssertEqual(suffixImage.frames?.map { [$0.t, $0.h, $0.w] }, [[1, 4, 4]])
            XCTAssertTrue(arrayEqual(suffixImage.pixels, imageB.pixels).item(Bool.self))

            let coldCache = try model.newCache(parameters: nil)
            let (cold, _) = try continuation.lastLogits(
                model.prepare(full, cache: coldCache, state: nil, prefill: .init()))

            func warmLogits(_ image: LMInput.ProcessedImage) throws -> MLXArray {
                let cache = try model.newCache(parameters: nil)
                let (_, state) = try continuation.lastLogits(
                    model.prepare(
                        LMInput(text: .init(tokens: t1), image: imageA), cache: cache,
                        state: nil, prefill: .init()))
                let (logits, _) = try continuation.lastLogits(
                    model.prepare(
                        LMInput(text: suffix.text, image: image), cache: cache, state: state,
                        prefill: .init()))
                return logits
            }

            XCTAssertLessThanOrEqual(
                continuation.maxAbsDiff(try warmLogits(suffixImage), cold),
                MatmulPrecision.splitTolerance)
            // Control: the logits do tell the images apart, so the match above is not luck.
            XCTAssertGreaterThan(
                continuation.maxAbsDiff(try warmLogits(imageA), cold),
                MatmulPrecision.splitTolerance)
        }
    }

    func testSplitRefusesBoundaryInsideAnImageBlock() throws {
        let model = try Qwen35ContinuationTests.makeTinyModel()
        let ids =
            [1, 2, Self.visionStartTokenId] + Array(repeating: Self.imageTokenId, count: 4)
            + [Self.visionEndTokenId, 3]
        let input = LMInput(
            text: .init(tokens: MLXArray(ids).expandedDimensions(axis: 0)),
            image: continuation.image())

        // Placeholders sit at indices 3 ..< 7.
        for boundary in 4 ... 6 {
            XCTAssertNil(
                model.splitPreparedInput(input, droppingFirst: boundary),
                "boundary \(boundary) is inside the image's placeholders")
        }
        XCTAssertNotNil(model.splitPreparedInput(input, droppingFirst: 2))
    }

    /// A suffix without an image is ChatSession's plain-token path, not a split. Video
    /// positions are not computed for Qwen 3.5 splits, so video is refused.
    func testSplitRefusesSuffixWithoutImageAndVideo() throws {
        let model = try Qwen35ContinuationTests.makeTinyModel()
        let ids =
            [1, Self.visionStartTokenId] + Array(repeating: Self.imageTokenId, count: 4)
            + [Self.visionEndTokenId, 2, 3]
        let tokens = MLXArray(ids).expandedDimensions(axis: 0)

        let imageOnly = LMInput(text: .init(tokens: tokens), image: continuation.image())
        XCTAssertNil(model.splitPreparedInput(imageOnly, droppingFirst: 7))

        let withVideo = LMInput(
            text: .init(tokens: tokens), image: continuation.image(),
            video: .init(pixels: continuation.image().pixels, frames: [THW(1, 4, 4)]))
        XCTAssertNil(model.splitPreparedInput(withVideo, droppingFirst: 1))
    }

    // MARK: - ChatSession

    /// Text, then an image, then text. Turn 2 must split the prepared input and prefill
    /// only its suffix; turn 3 must extend the cache on the state turn 2 handed back.
    /// Each turn's first logits must match a cold prefill of the same prompt.
    func testChatSessionAppendsImageTurnToWarmCache() async throws {
        let model = try Qwen35ContinuationTests.makeTinyModel()
        let pixels = withRandomState(MLXRandom.RandomState(seed: 29)) {
            continuation.image().pixels
        }
        let renders = PromptLog()
        let firstLogits = LogitsLog()
        let tokenizer = PrivateUseTokenizer()
        let context = ModelContext(
            configuration: ModelConfiguration(id: "qwen35-split-test"),
            model: model,
            processor: ToyQwenVLProcessor(tokenizer: tokenizer, renders: renders),
            tokenizer: tokenizer)
        let maxTokens = 3
        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: maxTokens),
            components: GenerationComponents {
                RecordFirstLogitsAndForceToken(log: firstLogits, token: 7)
            })

        let turn1 = try await completionInfo(session.streamDetails(to: "plan"))
        let turn2 = try await completionInfo(
            session.streamDetails(
                to: "look", role: .user, images: [.array(pixels)], videos: []))
        let turn3 = try await completionInfo(session.streamDetails(to: "more"))

        let prompts = renders.values
        let logits = firstLogits.values
        XCTAssertEqual(prompts.count, 3)
        XCTAssertEqual(logits.count, 3)
        guard prompts.count == 3, logits.count == 3 else { return }

        XCTAssertEqual(turn1.cachedPromptTokenCount, 0)

        // A rebuild would report zero cached tokens and prefill the whole prompt.
        XCTAssertEqual(turn2.cachedPromptTokenCount, prompts[0].count + maxTokens)
        XCTAssertEqual(
            turn2.promptTokenCount, prompts[1].count - turn2.cachedPromptTokenCount)

        XCTAssertEqual(turn3.cachedPromptTokenCount, prompts[1].count + maxTokens)
        XCTAssertEqual(
            turn3.promptTokenCount, prompts[2].count - turn3.cachedPromptTokenCount)

        let image = LMInput.ProcessedImage(pixels: pixels, frames: [THW(1, 4, 4)])
        for turn in 1 ... 2 {
            let cache = try model.newCache(parameters: nil)
            let (cold, _) = try continuation.lastLogits(
                model.prepare(
                    LMInput(
                        text: .init(tokens: MLXArray(prompts[turn]).expandedDimensions(axis: 0)),
                        image: image),
                    cache: cache, state: nil, prefill: .init()))
            XCTAssertLessThanOrEqual(
                continuation.maxAbsDiff(MLXArray(logits[turn]), cold.reshaped(-1)),
                MatmulPrecision.splitTolerance, "turn \(turn + 1) diverged from a cold prefill")
        }
    }

    private func completionInfo(
        _ stream: AsyncThrowingStream<Generation, Error>
    ) async throws -> GenerateCompletionInfo {
        var info: GenerateCompletionInfo?
        for try await item in stream {
            if let value = item.info { info = value }
        }
        return try XCTUnwrap(info)
    }
}

// MARK: - ChatSession fixtures

/// Maps generated ids to private-use scalars and back, so a reply re-renders to the
/// tokens that produced it and each turn's prompt extends the last.
private struct PrivateUseTokenizer: Tokenizer {
    private static let base = 0xE000
    private static let roundTripCount = 100

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.unicodeScalars.map { scalar in
            let id = Int(scalar.value) - Self.base
            return (0 ..< Self.roundTripCount).contains(id) ? id : Int(scalar.value) % 80 + 1
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(
            String.UnicodeScalarView(
                tokenIds.compactMap { UnicodeScalar(Self.base + $0) }))
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? {
        UnicodeScalar(Self.base + id).map(String.init)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw ToyProcessorError.unsupported
    }
}

private enum ToyProcessorError: Error {
    case unsupported
}

/// Prepares input in the layout `Qwen3VLProcessor` produces: a `[1, n]` token row, one
/// vision-start / placeholders / vision-end run per image, and every image's patch rows
/// concatenated in prompt order.
///
/// Like `Qwen3VLProcessor` it attaches an int8 all-ones mask. ChatSession treats that mask
/// as absent, so turn 3 still extends the cache.
private struct ToyQwenVLProcessor: UserInputProcessor {
    let tokenizer: PrivateUseTokenizer
    let renders: PromptLog

    func prepare(input: UserInput) throws -> LMInput {
        guard case .chat(let messages) = input.prompt else { throw ToyProcessorError.unsupported }
        var tokens: [Int] = []
        var pixels: [MLXArray] = []
        for message in messages {
            switch message.role {
            case .system: tokens.append(96)
            case .user: tokens.append(97)
            case .assistant: tokens.append(94)
            case .tool: tokens.append(98)
            }
            for image in message.images {
                guard case .array(let rows) = image.source else {
                    throw ToyProcessorError.unsupported
                }
                pixels.append(rows)
                tokens.append(502)
                tokens.append(contentsOf: Array(repeating: 500, count: 4))
                tokens.append(503)
            }
            tokens.append(contentsOf: tokenizer.encode(text: message.content))
            if message.role != .assistant { tokens.append(95) }
        }
        tokens.append(94)
        renders.append(tokens)

        let image =
            pixels.isEmpty
            ? nil
            : LMInput.ProcessedImage(
                pixels: concatenated(pixels, axis: 0),
                frames: Array(repeating: THW(1, 4, 4), count: pixels.count))
        let promptArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)),
            image: image)
    }
}

/// `@unchecked Sendable` because the lock serialises every access.
private final class PromptLog: @unchecked Sendable {
    private let lock = NSLock()
    private var prompts: [[Int]] = []

    func append(_ prompt: [Int]) { lock.withLock { prompts.append(prompt) } }
    var values: [[Int]] { lock.withLock { prompts } }
}

/// `@unchecked Sendable` because the lock serialises every access.
private final class LogitsLog: @unchecked Sendable {
    private let lock = NSLock()
    private var logits: [[Float]] = []

    func append(_ row: [Float]) { lock.withLock { logits.append(row) } }
    var values: [[Float]] { lock.withLock { logits } }
}

/// Records the logits of a generation's first step, which come from the prefill's last
/// position, then forces `token` so every reply re-renders to the same ids.
private struct RecordFirstLogitsAndForceToken: LogitProcessor {
    private final class Flag {
        var isSet = false
    }

    let log: LogitsLog
    let token: Int32
    private let recorded = Flag()

    init(log: LogitsLog, token: Int32) {
        self.log = log
        self.token = token
    }

    func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        if !recorded.isSet {
            recorded.isSet = true
            log.append(logits.asType(.float32).reshaped(-1).asArray(Float.self))
        }
        let ids = MLXArray(0 ..< Int32(logits.dim(-1)))
        return MLX.where(ids .== MLXArray(token), logits, MLXArray(-Float.infinity))
    }

    func didSample(token: MLXArray) {}
}
