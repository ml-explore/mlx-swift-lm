// Copyright © 2026 Apple Inc.

import Foundation
import Testing
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Labeled-channel reasoning (``ReasoningChannel``), the protocol Gemma 4 uses:
/// `<|channel>LABEL\n ... <channel|>`, where the label decides whether the span is
/// reasoning or the answer.
@Suite
struct ReasoningChannelTests {

    // MARK: - Fixtures & helpers

    /// Gemma 4's protocol, spelled out rather than taken from
    /// ``ReasoningConfig/gemma4`` so these tests pin the emitter's behavior against a
    /// literal shape; the preset itself is pinned separately by `gemma4Preset()`.
    private static let gemma4 = ReasoningConfig(
        startDelimiter: "<|channel>", endDelimiter: "<channel|>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: false),
        isSpecialToken: true,
        channel: .gemma4)

    private typealias Segment = ReasoningEventEmitter.Segment

    private func run(
        config: ReasoningConfig = gemma4, primedInside: Bool = false, _ chunks: [String]
    ) -> [Segment] {
        var emitter = ReasoningEventEmitter(config: config, primedInside: primedInside)
        var segments: [Segment] = []
        for chunk in chunks { segments += emitter.process(chunk) }
        segments += emitter.finalize()
        return segments
    }

    private func text(_ segments: [Segment]) -> String {
        segments.map {
            switch $0 {
            case .reasoning(let s), .response(let s): s
            }
        }.joined()
    }

    private func reasoningText(_ segments: [Segment]) -> String {
        segments.compactMap { if case .reasoning(let s) = $0 { return s } else { return nil } }
            .joined()
    }

    private func responseText(_ segments: [Segment]) -> String {
        segments.compactMap { if case .response(let s) = $0 { return s } else { return nil } }
            .joined()
    }

    private func leaksMarker(_ segments: [Segment]) -> Bool {
        let all = text(segments)
        return all.contains("<|channel>") || all.contains("<channel|>")
    }

    // MARK: - The production case

    /// Gemma 4's 31B chat template prefills a closed, empty thought block after
    /// `<|turn>model` when thinking is off - but suppresses the whole generation
    /// prompt when the previous message was a tool call or tool response. The model
    /// is then the one to open the channel, on exactly the answer that follows a
    /// tool result. (The E2B template prefills nothing at all, so the same shape
    /// can arrive on a first answer.)
    @Test func emptyThoughtBlockThenAnswer() {
        let segments = run(["<|channel>thought\n<channel|>answer"])
        #expect(segments == [.response("answer")])
        #expect(!leaksMarker(segments))
    }

    /// The same input delivered one character at a time must resolve identically:
    /// both markers are 10 characters, so a splitter sized for `</think>` tears them.
    @Test func emptyThoughtBlockOneCharacterAtATime() {
        let segments = run("<|channel>thought\n<channel|>answer".map { String($0) })
        #expect(responseText(segments) == "answer")
        #expect(reasoningText(segments).isEmpty)
        #expect(!leaksMarker(segments))
    }

    @Test func thoughtChannelRoutesToReasoning() {
        let segments = run(["<|channel>thought\nweighing it up<channel|>the answer"])
        #expect(segments == [.reasoning("weighing it up"), .response("the answer")])
    }

    /// A label listed in `responseLabels` routes to the answer. Defensive rather
    /// than observed for Gemma 4 (no shipped template emits a `content` channel),
    /// but the routing has to exist: sending every channel to reasoning would
    /// swallow the reply.
    @Test func responseLabelledChannelIsNotReasoning() {
        let segments = run(["<|channel>content\nanswer<channel|>"])
        #expect(segments == [.response("answer")])
        #expect(reasoningText(segments).isEmpty)
    }

    @Test func labelIsNeverEmitted() {
        let segments = run(["<|channel>thought\nthinking<channel|>"])
        #expect(reasoningText(segments) == "thinking")
        #expect(!text(segments).contains("thought"))
    }

    // MARK: - Chunk-boundary robustness

    @Test func markersSplitAcrossChunks() {
        let segments = run(["<|chan", "nel>thou", "ght\nidea<chan", "nel|>done"])
        #expect(reasoningText(segments) == "idea")
        #expect(responseText(segments) == "done")
        #expect(!leaksMarker(segments))
    }

    @Test func labelTerminatorArrivesInALaterChunk() {
        let segments = run(["<|channel>thou", "ght", "\n", "idea<channel|>done"])
        #expect(reasoningText(segments) == "idea")
        #expect(responseText(segments) == "done")
    }

    /// State survives across `process(_:)` calls, so a block is still classified when
    /// it is split at any point - including by a caller that keeps one emitter across
    /// separate generation passes, which is how a reasoning block spanning a tool call
    /// stays intact.
    @Test func blockSpanningTwoProcessCalls() {
        var emitter = ReasoningEventEmitter(config: Self.gemma4, primedInside: false)
        var segments = emitter.process("<|channel>thought\nfirst half")
        #expect(emitter.isInsideReasoning)
        segments += emitter.process(" second half<channel|>answer")
        #expect(reasoningText(segments) == "first half second half")
        #expect(responseText(segments) == "answer")
        #expect(!emitter.isInsideReasoning)
    }

    /// Chunking must not change the classification. This feeds a corpus through
    /// every split shape - whole, one character at a time, and a spread of fixed
    /// widths that land inside both 10-character markers and inside the label - and
    /// requires the same text on the same side every time.
    ///
    /// Whitespace is normalized before comparing, and only whitespace: trimming
    /// around a delimiter is position-dependent (text before a marker loses its
    /// trailing whitespace only when both land in the same chunk), which is a
    /// pre-existing property of the pair path, not something the label state adds.
    @Test func chunkingNeverChangesClassification() {
        let corpus = [
            "<|channel>thought\nreasoning<channel|>answer",
            "<|channel>thought\n<channel|>answer",
            "<|channel>content\nanswer<channel|>",
            "<|channel>scratchpad\nnotes<channel|>answer",
            "plain answer with no channel at all",
            "answer<channel|>after a stray close",
            "<|channel|>analysis<|message|>harmony is inert",
            "<|channel>thought\na<channel|>b<|channel>thought\nc<channel|>d",
            "<|channel>thought\ncontains <|tool_call>syntax<channel|>answer",
            "trailing partial marker <|chan",
        ]

        for input in corpus {
            let whole = run([input])
            for width in [1, 2, 3, 5, 7, 9, 10, 11, 17] {
                let chunks = stride(from: 0, to: input.count, by: width).map { start -> String in
                    let from = input.index(input.startIndex, offsetBy: start)
                    let to = input.index(from, offsetBy: width, limitedBy: input.endIndex)
                    return String(input[from ..< (to ?? input.endIndex)])
                }
                let split = run(chunks)
                #expect(
                    normalized(reasoningText(split)) == normalized(reasoningText(whole)),
                    "reasoning differs at width \(width) for \(input.debugDescription)")
                #expect(
                    normalized(responseText(split)) == normalized(responseText(whole)),
                    "response differs at width \(width) for \(input.debugDescription)")
                #expect(!leaksMarker(split), "marker leaked at width \(width)")
            }
        }
    }

    private func normalized(_ text: String) -> String {
        text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    // MARK: - Adversarial input

    /// Harmony's markers (`<|channel|>analysis<|message|>`) have pipes on BOTH
    /// sides. Neither Gemma 4 token is a substring of them, and a sloppy `contains`
    /// implementation eats text here.
    @Test func harmonyMarkersAreInertText() {
        let segments = run(["<|channel|>analysis<|message|>hello"])
        #expect(segments == [.response("<|channel|>analysis<|message|>hello")])
    }

    /// Both delimiters are special tokens, so an end delimiter with no opener cannot
    /// be prose: the PROMPT opened the channel. The text before it has already
    /// streamed out and cannot be reclassified, but the marker itself must not reach
    /// the answer.
    @Test func strayEndDelimiterIsSwallowedNotLeaked() {
        let segments = run(["already answering<channel|>rest of the answer"])
        #expect(!leaksMarker(segments))
        #expect(responseText(segments) == "already answeringrest of the answer")
    }

    /// An opener whose label never terminates must not buffer without bound: past
    /// the label window the opener is malformed, and its text is shown (as
    /// reasoning) rather than silently swallowed up to some distant closer.
    @Test func unterminatedLabelDoesNotBufferUnboundedly() {
        let long = String(repeating: "x", count: 200)
        var emitter = ReasoningEventEmitter(config: Self.gemma4, primedInside: false)
        let segments = emitter.process("<|channel>" + long)
        #expect(!reasoningText(segments).isEmpty)
        #expect(reasoningText(segments).hasPrefix("x"))
        #expect(emitter.isInsideReasoning)
    }

    /// Hiding an unrecognized channel in the reasoning stream is the safe
    /// direction: the text stays readable and no scratchpad text reaches the answer.
    @Test func unknownLabelRoutesToReasoning() {
        let segments = run(["<|channel>scratchpad\nnotes<channel|>answer"])
        #expect(segments == [.reasoning("notes"), .response("answer")])
    }

    /// Adversarial rather than observed: no Gemma 4 template puts tool syntax inside
    /// a channel. But if a model ever writes it there, that text must not reach the
    /// answer, and therefore must not reach the tool-call parser.
    @Test func toolSyntaxInsideThoughtStaysReasoning() {
        let segments = run(["<|channel>thought\nI could emit <|tool_call>here<channel|>Hi"])
        #expect(reasoningText(segments) == "I could emit <|tool_call>here")
        #expect(responseText(segments) == "Hi")
    }

    // MARK: - Primed state

    @Test func primedInsideClosesOnEndDelimiter() {
        let segments = run(primedInside: true, ["carried thought<channel|>answer"])
        #expect(segments == [.reasoning("carried thought"), .response("answer")])
    }

    @Test func primedNeverClosesFlushesAsReasoning() {
        let segments = run(primedInside: true, ["thinking with no close"])
        #expect(segments == [.reasoning("thinking with no close")])
    }

    // MARK: - State signals

    @Test func isInsideReasoningIsTrueWhileReadingTheLabel() {
        var emitter = ReasoningEventEmitter(config: Self.gemma4, primedInside: false)
        #expect(!emitter.isInsideReasoning)
        _ = emitter.process("<|channel>tho")
        #expect(emitter.isInsideReasoning)
    }

    @Test func isInsideReasoningIsFalseInsideTheContentChannel() {
        var emitter = ReasoningEventEmitter(config: Self.gemma4, primedInside: false)
        _ = emitter.process("<|channel>content\npartial answer")
        #expect(!emitter.isInsideReasoning)
    }

    @Test func hasClosedReasoningLatchesForThoughtOnly() {
        var thought = ReasoningEventEmitter(config: Self.gemma4, primedInside: false)
        _ = thought.process("<|channel>thought\nx<channel|>")
        #expect(thought.hasClosedReasoning)

        var content = ReasoningEventEmitter(config: Self.gemma4, primedInside: false)
        _ = content.process("<|channel>content\nx<channel|>")
        #expect(!content.hasClosedReasoning)
    }

    /// A malformed opener that closes before its label ever terminates still closed a
    /// channel the scanner was reporting as reasoning, so the latch must fire - or a
    /// think-then-call collector waiting on it would never be released.
    @Test func hasClosedReasoningLatchesForAnUnterminatedLabel() {
        var emitter = ReasoningEventEmitter(config: Self.gemma4, primedInside: false)
        _ = emitter.process("<|channel>oops<channel|>answer")
        #expect(emitter.hasClosedReasoning)
        #expect(!emitter.isInsideReasoning)
    }

    /// The empty block opens and closes within one `process` call, so sampling
    /// `isInsideReasoning` afterwards cannot see it.
    @Test func hasClosedReasoningDetectsTheEmptyBlock() {
        var emitter = ReasoningEventEmitter(config: Self.gemma4, primedInside: false)
        _ = emitter.process("<|channel>thought\n<channel|>answer")
        #expect(emitter.hasClosedReasoning)
        #expect(!emitter.isInsideReasoning)
    }

    @Test func finalizeFlushesAnUnterminatedLabelAsReasoning() {
        var emitter = ReasoningEventEmitter(config: Self.gemma4, primedInside: false)
        _ = emitter.process("<|channel>tho")
        let flushed = emitter.finalize()
        #expect(flushed == [.reasoning("tho")])
    }

    // MARK: - Prompt priming detection

    /// The template's own prefill for thinking-off: a CLOSED empty block, so the
    /// stream that follows starts outside.
    @Test func closedPrefillIsNotPrimed() {
        #expect(
            !ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: "<|turn>model\n<|channel>thought\n<channel|>",
                config: Self.gemma4))
    }

    @Test func openChannelInPromptIsPrimed() {
        #expect(
            ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: "<|turn>model\n<|channel>thought\n", config: Self.gemma4))
    }

    /// After a tool response the template emits no generation prompt at all, so
    /// nothing is primed and the model opens the channel itself.
    @Test func toolResponseTailIsNotPrimed() {
        #expect(
            !ReasoningEventEmitter.promptEndsInsideReasoning(
                renderedPromptTail: "response:get_weather{value:sunny}<tool_response|>",
                config: Self.gemma4))
    }

    // MARK: - Delimiter-pair configs are unaffected

    @Test func pairedConfigIgnoresChannelMarkers() {
        let think = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .alwaysOn)
        let segments = run(config: think, ["<|channel>thought\nx<channel|>y"])
        #expect(segments == [.response("<|channel>thought\nx<channel|>y")])
    }

    // MARK: - Model declaration

    /// The Gemma 4 model types declare this through `ChatConventionsProviding`, so the
    /// shared preset is what those declarations resolve to.
    @Test func gemma4Preset() {
        let config = ReasoningConfig.gemma4
        #expect(config.startDelimiter == "<|channel>")
        #expect(config.endDelimiter == "<channel|>")
        #expect(config.channel == .gemma4)
        #expect(config.isSpecialToken)
        #expect(config.promptStrategy == .templateFlag(key: "enable_thinking", defaultOn: false))
    }

    /// The delimiter-pair presets keep a `nil` channel, so adding the channel shape
    /// leaves every family that used them on the path they were already on.
    @Test func pairedPresetsHaveNoChannel() {
        #expect(ReasoningConfig.thinkTagsWithEnableThinking.channel == nil)
        #expect(ReasoningConfig.alwaysOnThinking.channel == nil)
    }
}

// MARK: - Declaration scope
//
// XCTest, and colocated with `ChatConventionsTests`' style, because these go through
// `LLMTypeRegistry` the way that file does rather than constructing a model type
// directly. Tiny random-weight config, so they run in CI without downloads.
final class ReasoningChannelDeclarationTests: XCTestCase {

    /// Gemma 3 and earlier have no reasoning protocol, and nothing about this change
    /// gives them one. The risk this guards is an over-broad declaration: nothing stops
    /// an extension from landing on a shared base or on the wrong class, and a Gemma 3
    /// model that suddenly reported `.gemma4` would hide its answers in the thought
    /// stream. `Gemma4TextTests.declaresChannelReasoningConfig` is the positive half of
    /// this pair - it builds a real `Gemma4TextModel` and requires the declaration to be
    /// there - so this `nil` means "Gemma 3 declares none", not "nothing declares
    /// anything".
    ///
    /// Only `model_type` is required; the rest override Gemma 3's full-size defaults so
    /// the model is small enough to build in a test without downloading weights.
    func testGemma3DeclaresNoReasoningConfig() async throws {
        let json = """
            {
                "model_type": "gemma3",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rms_norm_eps": 1e-6,
                "vocab_size": 64
            }
            """
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(json.utf8), modelType: "gemma3")
        XCTAssertNil(model.reasoningConfig)
    }

    /// No registered resolver claims the ids these two families actually ship under, so
    /// nothing overrides the declarations above. Resolvers are keyed on id and type, so
    /// representative ids are the most this can assert.
    func testNoResolverClaimsGemma() {
        XCTAssertNil(
            ChatConventionsRegistry.shared.reasoningConfig(
                modelId: "mlx-community/gemma-3-27b-it-4bit", modelType: "gemma3"))
        XCTAssertNil(
            ChatConventionsRegistry.shared.reasoningConfig(
                modelId: "mlx-community/gemma-4-31b-it-4bit", modelType: "gemma4_text"))
    }
}
