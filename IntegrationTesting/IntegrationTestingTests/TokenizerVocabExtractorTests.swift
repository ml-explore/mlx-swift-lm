// Copyright © 2026 Apple Inc.

#if GuidedGenerationSupport

    import Testing
    import Foundation
    import MLXLMCommon
    @testable import MLXFoundationModels

    /// Golden contract tests for `TokenizerVocabExtractor`.
    ///
    /// The extractor produces a per-token byte table the guided-generation
    /// backend consumes to align its grammar state with the tokenizer's
    /// own decoding. For guided generation to advance correctly, the bytes
    /// the extractor produces for a token id `t` must agree with the bytes
    /// that token contributes to the tokenizer's own decode output when
    /// `t` appears in a sequence.
    ///
    /// Golden invariant:
    ///
    ///   For any text T and `ids = encode(T, specials: false)`:
    ///     concat(extractor.bytes(for: id) for id in ids)
    ///       == decode(ids, specials: false).utf8
    ///
    /// If this invariant breaks, the backend's grammar state diverges from
    /// the actual stream the model produces, masks reject every extending
    /// token, and generation appears to "freeze" while burning through its
    /// token budget.
    @Suite(.serialized)
    struct TokenizerVocabExtractorTests {

        // MARK: - Qwen (BPE with Ġ / Ċ conventions)

        @Test("Qwen BPE: ASCII text round-trips")
        func qwenBpeAsciiRoundTrip() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await assertRoundTrip(
                modelID: TestFixtures.defaultModelID,
                text: "Hello, world!"
            )
        }

        @Test("Qwen BPE: leading space round-trips (Ġ convention)")
        func qwenBpeLeadingSpaceRoundTrip() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await assertRoundTrip(
                modelID: TestFixtures.defaultModelID,
                text: " the quick brown fox"
            )
        }

        @Test("Qwen BPE: newlines round-trip (Ċ convention)")
        func qwenBpeNewlineRoundTrip() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await assertRoundTrip(
                modelID: TestFixtures.defaultModelID,
                text: "line 1\nline 2\nline 3"
            )
        }

        @Test("Qwen BPE: non-ASCII round-trips")
        func qwenBpeUnicodeRoundTrip() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await assertRoundTrip(
                modelID: TestFixtures.defaultModelID,
                text: "日本語"
            )
        }

        @Test("Qwen BPE: JSON-shaped text round-trips")
        func qwenBpeJsonRoundTrip() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await assertRoundTrip(
                modelID: TestFixtures.defaultModelID,
                text: #"{"title":"Itinerary","summary":"A brief overview"}"#
            )
        }

        @Test("Qwen BPE: text from the deeply-nested fixture round-trips")
        func qwenBpeDeeplyNestedFixtureRoundTrip() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            // This fragment exercises tokens where extractor bytes must match
            // the decode output; if they do not, the grammar cannot advance
            // beyond it.
            try await assertRoundTrip(
                modelID: TestFixtures.defaultModelID,
                text:
                    #"{"title":"Two-Section Itinerary", "summary":"This itinerary is designed to provide a structured plan for"#
            )
        }

        // MARK: - Gemma (SentencePiece with ▁ / <0xNN> conventions)

        @Test("Gemma SentencePiece: ASCII text round-trips")
        func gemmaSpAsciiRoundTrip() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await assertRoundTrip(
                modelID: TestFixtures.gemmaModelID,
                text: "Hello, world!"
            )
        }

        @Test("Gemma SentencePiece: leading space round-trips (▁ convention)")
        func gemmaSpLeadingSpaceRoundTrip() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await assertRoundTrip(
                modelID: TestFixtures.gemmaModelID,
                text: " the quick brown fox"
            )
        }

        @Test("Gemma SentencePiece: non-ASCII round-trips")
        func gemmaSpUnicodeRoundTrip() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            try await assertRoundTrip(
                modelID: TestFixtures.gemmaModelID,
                text: "日本語"
            )
        }

        // MARK: - Helpers

        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private func assertRoundTrip(
            modelID: String,
            text: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) async throws {
            let container = try await loadTestModelContainer(id: modelID)
            try await container.perform { context in
                let vocab = TokenizerVocabExtractor.extract(from: context.tokenizer)
                let offsets = Self.prefixOffsets(of: vocab.tokenLens)
                let ids = context.tokenizer.encode(text: text, addSpecialTokens: false)

                // Tokenizer self-consistency. If this fails, the problem is in
                // encode/decode themselves, not in our extractor.
                let tokenizerDecoded = context.tokenizer.decode(
                    tokenIds: ids,
                    skipSpecialTokens: false
                )

                // Extractor consistency: concatenated per-token bytes must match
                // what the tokenizer's own decode produces for the same id list.
                var extractorBytes: [UInt8] = []
                extractorBytes.reserveCapacity(tokenizerDecoded.utf8.count)
                for id in ids {
                    guard id >= 0 && id < vocab.vocabSize else {
                        Issue.record(
                            "encode() returned out-of-range id \(id) for vocabSize \(vocab.vocabSize) in \(modelID)",
                            sourceLocation: sourceLocation
                        )
                        return
                    }
                    let start = offsets[id]
                    let end = offsets[id + 1]
                    extractorBytes.append(contentsOf: vocab.tokenBytes[start ..< end])
                }

                let decodedBytes = Array(tokenizerDecoded.utf8)

                #expect(
                    extractorBytes == decodedBytes,
                    """
                    Extractor bytes diverge from tokenizer decode output for \(modelID).
                    text        : \(text.debugDescription)
                    ids         : \(ids)
                    decode(ids) : \(tokenizerDecoded.debugDescription)
                    expected (\(decodedBytes.count) bytes): \(Self.hex(decodedBytes))
                    got      (\(extractorBytes.count) bytes): \(Self.hex(extractorBytes))
                    first-divergence index: \(Self.firstDivergence(decodedBytes, extractorBytes) ?? -1)
                    """,
                    sourceLocation: sourceLocation
                )
            }
        }

        private static func prefixOffsets(of lens: [UInt32]) -> [Int] {
            var offsets: [Int] = []
            offsets.reserveCapacity(lens.count + 1)
            offsets.append(0)
            var running = 0
            for len in lens {
                running += Int(len)
                offsets.append(running)
            }
            return offsets
        }

        private static func hex(_ bytes: [UInt8]) -> String {
            let shown = bytes.prefix(80)
            let s = shown.map { String(format: "%02x", $0) }.joined(separator: " ")
            return bytes.count > shown.count ? s + " ..." : s
        }

        private static func firstDivergence(_ lhs: [UInt8], _ rhs: [UInt8]) -> Int? {
            let n = min(lhs.count, rhs.count)
            for i in 0 ..< n where lhs[i] != rhs[i] { return i }
            return lhs.count == rhs.count ? nil : n
        }
    }

#endif
