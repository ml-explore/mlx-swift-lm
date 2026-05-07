import Foundation
import Testing

@testable import MLXLMCommon

private struct MockVocabulary {
    let bytesByTokenId: [Int: [UInt8]]
    let stringByTokenId: [Int: String]

    init(_ entries: [Int: [UInt8]]) {
        bytesByTokenId = entries
        stringByTokenId = entries.mapValues { String(decoding: $0, as: UTF8.self) }
    }
}

/// Tokenizer whose `decode` concatenates the bytes for the requested ids — byte-prefix
/// monotonic by construction, so every branch of the streaming algorithm is reachable
/// without a real model.
private struct MockTokenizer: MLXLMCommon.Tokenizer {
    let vocabulary: MockVocabulary
    let invalidIds: Set<Int>

    init(_ entries: [Int: [UInt8]], invalidIds: Set<Int> = []) {
        vocabulary = MockVocabulary(entries)
        self.invalidIds = invalidIds
    }

    private struct UnknownTokenId: Error { let id: Int }
    private struct InjectedFailure: Error {}

    func encode(text: String, addSpecialTokens: Bool) throws -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
        var bytes: [UInt8] = []
        for id in tokenIds {
            if invalidIds.contains(id) {
                throw InjectedFailure()
            }
            guard let chunk = vocabulary.bytesByTokenId[id] else {
                throw UnknownTokenId(id: id)
            }
            bytes.append(contentsOf: chunk)
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { vocabulary.stringByTokenId[id] }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite("StreamingDetokenizer")
struct StreamingDetokenizerSuite {
    /// "Hello" split into single-byte ASCII tokens.
    private static let asciiTokenizer = MockTokenizer([
        1: Array("H".utf8),
        2: Array("e".utf8),
        3: Array("l".utf8),
        4: Array("o".utf8),
        5: Array(" ".utf8),
        6: Array("w".utf8),
        7: Array("r".utf8),
        8: Array("d".utf8),
    ])

    /// 🌍 (U+1F30D, four UTF-8 bytes) split byte-by-byte across tokens 10–13.
    private static let fragmentedEmojiTokenizer = MockTokenizer([
        10: [0xF0],
        11: [0x9F],
        12: [0x8C],
        13: [0x8D],
        20: Array("!".utf8),
    ])

    @Test
    func emitsEachAsciiToken() throws {
        let stream = Self.asciiTokenizer.streamingDetokenizer()
        let ids = [1, 2, 3, 3, 4]  // "Hello"
        var out = ""
        for id in ids {
            if let chunk = try stream.consume(id) {
                out.append(chunk)
            }
        }
        #expect(out == "Hello")
    }

    @Test
    func multiByteScalarBufferedAcrossTokens() throws {
        let stream = Self.fragmentedEmojiTokenizer.streamingDetokenizer()
        var collected: [String?] = []
        for id in [10, 11, 12, 13] {
            collected.append(try stream.consume(id))
        }
        // The first three consumes return nil because the buffered bytes still
        // form an incomplete scalar; the fourth completes the scalar and emits.
        #expect(collected[0] == nil)
        #expect(collected[1] == nil)
        #expect(collected[2] == nil)
        #expect(collected[3] == "🌍")
    }

    @Test
    func bufferStaysBoundedAcrossLongStream() throws {
        // Repeat "Hello world" many times. The buffer should trim after each
        // emission to a small constant rather than growing with stream length.
        let helloIds = [1, 2, 3, 3, 4, 5, 6, 4, 7, 3, 8]  // "Hello world"
        let ids = Array(repeating: helloIds, count: 1000).flatMap { $0 }
        let stream = Self.asciiTokenizer.streamingDetokenizer()

        var output = ""
        for id in ids {
            if let chunk = try stream.consume(id) {
                output.append(chunk)
            }
        }
        let oneShot = try Self.asciiTokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
        #expect(output == oneShot)

        #expect(stream.ids.count < 5)
    }

    @Test
    func consumeBatchReturnsConcatenation() throws {
        let stream = Self.asciiTokenizer.streamingDetokenizer()
        let chunk = try stream.consume([1, 2, 3, 3, 4])
        #expect(chunk == "Hello")
    }

    @Test
    func seedingFromInitialIds() throws {
        let stream = Self.asciiTokenizer.streamingDetokenizer(
            initialTokenIds: [1, 2]
        )
        // First consume seeds the prefix from [1, 2] and emits only the
        // new token's byte. Subsequent consumes must continue to emit
        // only their own bytes — a regression here would re-emit the
        // seeded "He" once the buffer is trimmed.
        let first = try stream.consume(3)
        #expect(first == "l")
        let second = try stream.consume(3)
        #expect(second == "l")
        let third = try stream.consume(4)
        #expect(third == "o")

        // Trim after each emit keeps the buffer bounded regardless of stream length.
        #expect(stream.ids.count <= 1)
    }

    @Test
    func consumeReturnsNilWhenTokenAddsNoBytes() throws {
        // A skipped-special token adds no bytes to the cumulative decode, so
        // `consume` withholds rather than throwing — the cached prefix is
        // still a valid prefix of the unchanged decode.
        struct SkipSpecial: MLXLMCommon.Tokenizer {
            static let regularBytes: [Int: [UInt8]] = [
                0: Array("a".utf8),
                1: Array("b".utf8),
            ]
            func encode(text: String, addSpecialTokens: Bool) throws -> [Int] { [] }
            func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
                var bytes: [UInt8] = []
                for id in tokenIds {
                    if id == 99 {
                        if skipSpecialTokens { continue }
                        bytes.append(contentsOf: Array("<eos>".utf8))
                        continue
                    }
                    if let chunk = Self.regularBytes[id] {
                        bytes.append(contentsOf: chunk)
                    }
                }
                return String(decoding: bytes, as: UTF8.self)
            }
            func convertTokenToId(_ token: String) -> Int? { nil }
            func convertIdToToken(_ id: Int) -> String? { nil }
            var bosToken: String? { nil }
            var eosToken: String? { "<eos>" }
            var unknownToken: String? { nil }
            func applyChatTemplate(
                messages: [[String: any Sendable]],
                tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] { [] }
        }

        let stream = SkipSpecial().streamingDetokenizer(skipSpecialTokens: true)
        #expect(try stream.consume(0) == "a")
        #expect(try stream.consume(99) == nil)
        #expect(try stream.consume(1) == "b")
    }

    @Test
    func consumeRollsBackWhenDecodeThrowsForUnknownToken() throws {
        // When the tokenizer rejects the just-appended id, `consume`
        // must not mutate any internal state — the next valid id has
        // to behave as if the rejected one had never been fed.
        let tokenizer = MockTokenizer(
            [
                1: Array("Hi".utf8),
                2: Array(" ".utf8),
                3: Array("there".utf8),
            ],
            invalidIds: [99]
        )
        let stream = tokenizer.streamingDetokenizer()

        _ = try stream.consume(1)

        let beforeIds = stream.ids
        let beforePrefix = stream.prefix
        let beforeIndex = stream.prefixIndex

        // This token causes decode to throw — internal state must not change.
        do {
            _ = try stream.consume(99)
            Issue.record("Expected decode failure")
        } catch {
            // Expected.
        }

        #expect(stream.ids == beforeIds)
        #expect(stream.prefix == beforePrefix)
        #expect(stream.prefixIndex == beforeIndex)

        // Subsequent valid consumes still produce the right text — i.e. the
        // poison id was fully rolled back, not buffered.
        let chunk2 = try stream.consume(2)
        let chunk3 = try stream.consume(3)
        let combined = (chunk2 ?? "") + (chunk3 ?? "")
        #expect(combined == " there")
    }

    @Test
    func consumeRollsBackWhenLaterDecodeThrows() throws {
        // The algorithm calls `decode` twice per `consume` (working text, then
        // prefix refresh). This mock fails on the second call to verify the
        // rollback holds for the post-emit decode, not just the first one.
        struct PartialFailureTokenizer: MLXLMCommon.Tokenizer {
            struct Failure: Error {}
            func encode(text: String, addSpecialTokens: Bool) throws -> [Int] { [] }
            func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
                switch tokenIds {
                case [1]: return "Hello"
                case [1, 2]: return "Hello world"
                case [2]: throw Failure()
                default: return ""
                }
            }
            func convertTokenToId(_ token: String) -> Int? { nil }
            func convertIdToToken(_ id: Int) -> String? { nil }
            var bosToken: String? { nil }
            var eosToken: String? { nil }
            var unknownToken: String? { nil }
            func applyChatTemplate(
                messages: [[String: any Sendable]],
                tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] { [] }
        }

        let stream = PartialFailureTokenizer().streamingDetokenizer()
        let first = try stream.consume(1)
        #expect(first == "Hello")

        #expect(stream.ids == [1])
        #expect(stream.prefix == "Hello")
        #expect(stream.prefixIndex == 1)

        let beforeIds = stream.ids
        let beforePrefix = stream.prefix
        let beforeIndex = stream.prefixIndex

        do {
            _ = try stream.consume(2)
            Issue.record("Expected decode failure")
        } catch is PartialFailureTokenizer.Failure {
            // Expected.
        }

        #expect(stream.ids == beforeIds)
        #expect(stream.prefix == beforePrefix)
        #expect(stream.prefixIndex == beforeIndex)
    }

    @Test
    func invalidStreamingPrefixThrows() throws {
        // Non-monotonic decode: the one-token result is not a prefix of the two-token result.
        struct NonMonotonic: MLXLMCommon.Tokenizer {
            func encode(text: String, addSpecialTokens: Bool) throws -> [Int] { [] }
            func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
                if tokenIds == [1] { return "abc" }
                if tokenIds == [1, 2] { return "xyzlonger" }  // breaks the prefix invariant
                return ""
            }
            func convertTokenToId(_ token: String) -> Int? { nil }
            func convertIdToToken(_ id: Int) -> String? { nil }
            var bosToken: String? { nil }
            var eosToken: String? { nil }
            var unknownToken: String? { nil }
            func applyChatTemplate(
                messages: [[String: any Sendable]],
                tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] { [] }
        }

        let stream = NonMonotonic().streamingDetokenizer()
        _ = try stream.consume(1)
        do {
            _ = try stream.consume(2)
            Issue.record("Expected invalidStreamingPrefix throw")
        } catch let error as TokenizerError {
            guard
                case .invalidStreamingPrefix(let tokenId, let expectedPrefix, let actualString) =
                    error
            else {
                Issue.record("Expected .invalidStreamingPrefix, got \(error)")
                return
            }
            #expect(tokenId == 2)
            #expect(expectedPrefix == "abc")
            #expect(actualString == "xyzlonger")
        }
    }

    @Test
    func streamingDetokenizerPrefersRawDecode() throws {
        // A tokenizer where `decode` applies cleanup (uppercases) and
        // `rawDecode` returns the original bytes. The streaming detokenizer
        // should pick `rawDecode` and emit lowercase chunks.
        struct CleanupTokenizer: MLXLMCommon.StreamingDecodeTokenizer {
            func encode(text: String, addSpecialTokens: Bool) throws -> [Int] { [] }
            func decode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
                try rawDecode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
                    .uppercased()
            }
            func rawDecode(tokenIds: [Int], skipSpecialTokens: Bool) throws -> String {
                let map: [Int: String] = [1: "h", 2: "i"]
                return tokenIds.compactMap { map[$0] }.joined()
            }
            func convertTokenToId(_ token: String) -> Int? { nil }
            func convertIdToToken(_ id: Int) -> String? { nil }
            var bosToken: String? { nil }
            var eosToken: String? { nil }
            var unknownToken: String? { nil }
            func applyChatTemplate(
                messages: [[String: any Sendable]],
                tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] { [] }
        }

        let tokenizer = CleanupTokenizer()
        // Sanity-check the cleanup divergence is wired correctly.
        #expect(try tokenizer.decode(tokenIds: [1, 2], skipSpecialTokens: false) == "HI")
        #expect(try tokenizer.rawDecode(tokenIds: [1, 2], skipSpecialTokens: false) == "hi")

        let stream = tokenizer.streamingDetokenizer()
        let first = try stream.consume(1)
        let second = try stream.consume(2)
        #expect((first ?? "") + (second ?? "") == "hi")
    }

    @Test
    func generationStreamFinishesThrowingOnHandlerError() async throws {
        // Verify the throwing-stream wiring: when a TokenLoopHandler.onToken
        // throws, generateLoopTask catches it and finishes the AsyncThrowingStream
        // with that error rather than truncating silently.
        struct InjectedFailure: Error, Equatable {}

        struct MockIterator: TokenIteratorProtocol {
            var remaining: [Int]
            var maxTokens: Int? { nil }
            var tokenCount: Int { 0 }
            var promptPrefillTime: TimeInterval { 0 }

            mutating func next() -> Int? {
                guard !remaining.isEmpty else { return nil }
                return remaining.removeFirst()
            }
        }

        struct ThrowingHandler: TokenLoopHandler {
            typealias Output = String

            mutating func onToken(
                _ token: Int,
                emit: (sending String) ->
                    AsyncThrowingStream<String, Error>.Continuation
                    .YieldResult
            ) throws -> Bool {
                throw InjectedFailure()
            }

            mutating func onStopToken(
                _ token: Int,
                emit: (sending String) ->
                    AsyncThrowingStream<String, Error>.Continuation
                    .YieldResult
            ) throws -> Bool { true }

            mutating func onGenerationEnd(
                emit: (sending String) ->
                    AsyncThrowingStream<String, Error>.Continuation
                    .YieldResult
            ) {}

            func infoEvent(_ info: GenerateCompletionInfo) -> String { "info" }
        }

        let iterator = MockIterator(remaining: [1, 2, 3])
        let modelConfiguration = ModelConfiguration(id: "test")
        let (stream, _) = generateLoopTask(
            promptTokenCount: 0,
            modelConfiguration: modelConfiguration,
            tokenizer: Self.asciiTokenizer,
            iterator: iterator,
            handler: ThrowingHandler()
        )

        do {
            for try await _ in stream {
                Issue.record("Stream should have thrown before yielding")
            }
            Issue.record("Stream finished without throwing")
        } catch is InjectedFailure {
            // Expected.
        }
    }

    @Test
    func consumeReturnsNilWhenTokenAddsInvalidTrailingByte() throws {
        // Byte-fallback case: a token grows the cumulative decode but the new
        // bytes only extend it with a Unicode replacement character. The
        // length guard alone wouldn't withhold (length grew), but the
        // `\u{fffd}`-suffix check must — the trailing fragment is not yet a
        // valid scalar. Mirrors Rust's `decode_byte_fallback` doctest.
        let tokenizer = MockTokenizer([
            0: Array(" ".utf8),
            1: [0xC3],  // first byte of "é"
            2: [0xA9],  // second byte of "é"; alone, an invalid trailing byte
        ])
        let stream = tokenizer.streamingDetokenizer()

        #expect(try stream.consume(0) == " ")
        #expect(try stream.consume(1) == nil)
        #expect(try stream.consume(2) == "é")

        let prefixBeforeStrayByte = stream.prefix
        let indexBeforeStrayByte = stream.prefixIndex

        // Feeding 0xA9 again grows the decode but only with `\u{fffd}`.
        #expect(try stream.consume(2) == nil)

        // No emission, so the cached prefix and index must not advance.
        #expect(stream.prefix == prefixBeforeStrayByte)
        #expect(stream.prefixIndex == indexBeforeStrayByte)
    }
}
