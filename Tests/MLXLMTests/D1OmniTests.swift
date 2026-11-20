// Copyright © 2026 Apple Inc.

import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLX
import MLXLMCommon
import MLXNN
import Testing
import UniformTypeIdentifiers

@testable import MLXDecisions
@testable import MLXVLM

// Ports of the d1-omni tests in mlx-vlm `tests/test_extraction_models.py`, on its tiny config.
// MLX work runs on the CPU, where the exact-equality checks hold.

/// The tiny `d1_omni` config from mlx-vlm `tests/extraction_cases.json`.
private let tinyOmniConfigJSON = """
    {
      "model_type": "d1_omni",
      "text_config": {
        "vocab_size": 320, "hidden_size": 64, "intermediate_size": 96,
        "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
        "layer_types": ["conv", "full_attention", "conv"], "block_multiple_of": 32
      },
      "vision_config": {
        "model_type": "siglip2_vision_model", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_patches": 16
      },
      "projector_hidden_size": 48,
      "head_layers": 2,
      "temperatures": {"choice:2": 1.5, "score": 2.0}
    }
    """

private func tinyConfiguration() throws -> D1OmniConfiguration {
    try JSONDecoder().decode(D1OmniConfiguration.self, from: Data(tinyOmniConfigJSON.utf8))
}

/// A tiny model with every parameter drawn from `normal * 0.5`, as Python's `_random_d1_omni`.
private func randomD1Omni() throws -> D1Omni {
    let model = D1Omni(try tinyConfiguration())
    var randomized = [String: MLXArray]()
    for (index, (key, value)) in model.parameters().flattened().enumerated() {
        randomized[key] = MLXRandom.normal(value.shape, key: MLXRandom.key(UInt64(index))) * 0.5
    }
    try model.update(parameters: ModuleParameters.unflattened(randomized), verify: [.all])
    eval(model)
    return model
}

/// Model parameters under the reference checkpoint's names and torch layouts, as mlx-vlm's
/// `test_sanitize_converts_reference_checkpoint` builds them.
private func referenceCheckpoint(_ parameters: [String: MLXArray], headLayers: Int)
    -> [String: MLXArray]
{
    var source = [String: MLXArray]()
    for (key, value) in parameters {
        let value = key.hasSuffix(".conv.conv.weight") ? value.transposed(0, 2, 1) : value
        let key =
            key
            .replacingOccurrences(of: "vision.tower.", with: "vision.tower.vision_model.")
            .replacingOccurrences(of: "head.scorer.layers.", with: "head.scorer.")
            .replacingOccurrences(of: ".attention.out_proj.", with: ".self_attn.out_proj.")
        source[key] = value
    }
    for layer in 0 ..< headLayers {
        let prefix = "head.head.layers.\(layer)"
        for suffix in ["weight", "bias"] {
            let parts = ["query", "key", "value"].map {
                source.removeValue(forKey: "\(prefix).attention.\($0)_proj.\(suffix)")!
            }
            source["\(prefix).self_attn.in_proj_\(suffix)"] = concatenated(parts)
        }
    }
    return source
}

/// Random RGB bytes, `height` rows of `width` pixels.
private func randomPixels(width: Int, height: Int, seed: UInt64) -> RGB8Image {
    let bytes = MLXRandom.randInt(
        Int32(0) ..< Int32(256), [height * width * 3], key: MLXRandom.key(seed)
    ).asType(.uint8).asArray(UInt8.self)
    return RGB8Image(width: width, height: height, bytes: bytes)
}

private func kind(_ answer: DecisionAnswer) -> DecisionQuestionKind {
    switch answer.value {
    case .choice: .choice
    case .score: .score
    case .bool: .bool
    }
}

/// `image` encoded losslessly as `type`, with an EXIF orientation when one is given.
private func encoded(_ image: RGB8Image, as type: UTType, orientation: Int? = nil) throws -> Data {
    let provider = try #require(CGDataProvider(data: Data(image.bytes) as CFData))
    let cgImage = try #require(
        CGImage(
            width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: image.width * 3, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    let encoded = NSMutableData()
    let destination = try #require(
        CGImageDestinationCreateWithData(encoded, type.identifier as CFString, 1, nil))
    let properties = orientation.map { [kCGImagePropertyOrientation: $0] as CFDictionary }
    CGImageDestinationAddImage(destination, cgImage, properties)
    #expect(CGImageDestinationFinalize(destination))
    return encoded as Data
}

/// A PNG of `image` with an eXIf chunk that asks viewers to turn it 90 degrees (orientation 6).
private func pngWithOrientation6(_ image: RGB8Image) throws -> Data {
    // Signature (8 bytes) and IHDR (25 bytes), then the eXIf chunk.
    let exif = try #require(
        Data(base64Encoded: "AAAAGmVYSWZNTQAqAAAACAABARIAAwAAAAEABgAAAAAAANZnS2k="))
    var png = try encoded(image, as: .png)
    png.insert(contentsOf: exif, at: 33)
    return png
}

/// Hands out the character tokenizer and records the directories it was asked for.
private actor CharTokenizerLoader: TokenizerLoader {
    private(set) var directories: [URL] = []

    func load(from directory: URL) async throws -> any Tokenizer {
        directories.append(directory)
        return CharTokenizer()
    }
}

private func temporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appending(component: "d1-omni-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

/// Counts the calls to a tokenizer and the characters it was asked to encode.
private final class EncodedCharacters: @unchecked Sendable {
    private let lock = NSLock()
    private var total = 0
    private var encodes = 0

    func add(_ count: Int) {
        lock.withLock {
            total += count
            encodes += 1
        }
    }
    var count: Int { lock.withLock { total } }
    var calls: Int { lock.withLock { encodes } }
}

/// One token per character (`100 + code point`) and d1-omni's delimiters as fixed ids, as
/// Python's `_CharTokenizer`.
private struct CharTokenizer: Tokenizer {
    var bos: String? = "<|startoftext|>"
    var special: [String: Int] = [
        "<|startoftext|>": 1, "<|reserved_7|>": 7, "<|reserved_8|>": 8, "<|reserved_9|>": 9,
        "<|reserved_10|>": 10, "<|reserved_11|>": 11, "<|mask|>": 16,
    ]
    var encodedCharacters: EncodedCharacters?

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        encodedCharacters?.add(text.unicodeScalars.count)
        return chars(text)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { special[token] }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { bos }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private func chars(_ text: String) -> [Int] {
    text.unicodeScalars.map { 100 + Int($0.value) }
}

/// One id per piece of the pre-tokenizer regex in d1-omni's `tokenizer.json`, so text cut
/// inside a piece encodes differently.
private struct PreTokenizedTokenizer: Tokenizer {
    static let pattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"#
        + #"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    var encodedCharacters = EncodedCharacters()

    static func pieces(_ text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let whole = text as NSString
        return regex.matches(in: text, range: NSRange(location: 0, length: whole.length)).map {
            whole.substring(with: $0.range)
        }
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        encodedCharacters.add(text.unicodeScalars.count)
        return Self.pieces(text).map { (piece: String) -> Int in
            piece.unicodeScalars.reduce(7) { (id: Int, scalar: Unicode.Scalar) -> Int in
                (id * 31 + Int(scalar.value)) % 1_000_003
            }
        }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// Prose, numbers, punctuation, CJK and odd whitespace.
private let mixedFragments = [
    "customer", "refund", "It's", "they're", "we'll", "12345", "3.14", "end.", "(note)",
    "x-ray", "東京都", "こんにちは", "<|im_start|>", "<|mask|>", "{\"k\": [1, 2]}", "naïve",
    "a", "🙂", "wait...", "#42", "I'd", "'quoted'", "line\nbreak", "tab\tstop", "e\u{301}",
    "x \u{301}y", "\u{301}x",
]

/// Words longer than four characters, so a text of them has few tokens per character.
private let longWords = [
    "subscription", "cancellation", "acknowledged", "international", "yesterday", "customer",
    "refunded", "Wednesday", "unfortunately", "configuration",
]

/// `count` of `fragments` in a fixed pseudo-random order, with `separators` between them.
private func mixedText(
    fragments count: Int, separators: [String], seed: UInt64,
    from fragments: [String] = mixedFragments
) -> String {
    var state = seed
    func next(_ bound: Int) -> Int {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int((state >> 33) % UInt64(bound))
    }
    var text = ""
    for _ in 0 ..< count {
        text += fragments[next(fragments.count)] + separators[next(separators.count)]
    }
    return text
}

/// An RMSNorm that records the shapes it is called with.
private final class RecordingNorm: RMSNorm {
    var shapes = [[Int]]()

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        shapes.append(x.shape)
        return super.callAsFunction(x)
    }
}

private func text(of ids: [Int]) -> String {
    String(String.UnicodeScalarView(ids.compactMap { Unicode.Scalar(UInt32($0 - 100)) }))
}

private func named(_ name: String, _ question: DecisionQuestion) -> DecisionRequest.NamedQuestion {
    .init(name: name, question: question)
}

struct D1OmniTests {

    // MARK: Configuration

    @Test("Configuration decodes with Python defaults and LFM2's feed-forward rule")
    func configurationDecodes() throws {
        let tiny = try tinyConfiguration()
        #expect(tiny.textConfiguration.feedForwardDimensions == 64)
        #expect(tiny.textConfiguration.headDimensions == 16)
        #expect(tiny.temperatures == ["choice:2": 1.5, "score": 2.0])
        #expect(tiny.maxLength == 16384)
        #expect(tiny.imageTextLength == 896)
        #expect(tiny.bosTokenId == 1)
        #expect(tiny.visionConfiguration.hiddenSize == 32)
        #expect(tiny.visionConfiguration.patchSize == 16)

        // The released checkpoint: int(2 * 6656 / 3) = 4437, rounded up to 4608.
        let real = try JSONDecoder().decode(
            D1OmniConfiguration.TextConfiguration.self,
            from: Data(#"{"intermediate_size": 6656, "block_ffn_dim_multiplier": 1.0}"#.utf8))
        #expect(real.feedForwardDimensions == 4608)
        #expect(real.headDimensions == 64)

        let defaults = try JSONDecoder().decode(
            D1OmniConfiguration.self,
            from: Data(
                #"""
                {"text_config": {"num_hidden_layers": 2}, "vision_config": {"model_type": "siglip2_vision_model",
                "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2}}
                """#.utf8))
        #expect(defaults.textConfiguration.layerTypes == ["conv", "conv"])
        #expect(defaults.textConfiguration.hiddenSize == 1024)
        #expect(defaults.headLayers == 2)

        Device.withDefaultDevice(.cpu) {
            let model = D1Omni(tiny)
            let shapes = Dictionary(
                uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
            #expect(shapes["encoder.layers.0.conv.conv.weight"] == [64, 3, 1])
            #expect(shapes["encoder.layers.0.feed_forward.w1.weight"] == [64, 64])
            #expect(shapes["encoder.layers.1.self_attn.k_proj.weight"] == [32, 64])
            #expect(shapes["head.head.layers.1.attention.query_proj.bias"] == [64])
            #expect(shapes["head.scorer.layers.3.weight"] == [1, 64])
            #expect(shapes["head.type_emb.weight"] == [3, 64])
            #expect(model.decisionCapabilities.maximumScoreLevels == 10)
            #expect(model.decisionCapabilities.media == [.images])
            // The tower, then 2x2 unshuffled patches through the projector into the trunk width.
            #expect(shapes["vision.tower.embeddings.patch_embedding.weight"] == [32, 768])
            #expect(shapes["vision.tower.embeddings.position_embedding.weight"] == [16, 32])
            #expect(shapes["vision.tower.post_layernorm.weight"] == [32])
            #expect(shapes["vision.projector.linear_1.weight"] == [48, 128])
            #expect(shapes["vision.projector.linear_2.weight"] == [64, 48])
            let roots = Set(shapes.keys.map { $0.split(separator: ".")[0] })
            #expect(roots == ["encoder", "head", "vision"])
            #expect(model.excludedCheckpointNamespaces == ["audio"])
        }
    }

    // MARK: Prompt

    @Test("Delimiters resolve by name, with the config BOS as a fallback")
    func tokensResolveByName() throws {
        let tokens = try D1OmniPrompt.Tokens(CharTokenizer(), fallbackBOS: 5)
        #expect(
            tokens
                == .init(
                    bos: 1, state: 7, question: 8, option: 9, optionEnd: 10, decide: 11, marker: 16)
        )
        #expect(try D1OmniPrompt.Tokens(CharTokenizer(bos: nil), fallbackBOS: 5).bos == 5)

        var missing = CharTokenizer()
        missing.special["<|mask|>"] = nil
        #expect(throws: DecisionError.missingToken("The tokenizer has no <|mask|> token")) {
            try D1OmniPrompt.Tokens(missing, fallbackBOS: 1)
        }
    }

    @Test("Encode lays out state, question and options as the reference does")
    func encodeMatchesReference() throws {
        let tokenizer = CharTokenizer()
        let tokens = try D1OmniPrompt.Tokens(tokenizer, fallbackBOS: 1)
        func encode(_ state: String, _ question: DecisionQuestion, _ maxLength: Int) throws -> (
            ids: [Int], markers: [Int]
        ) {
            try D1OmniPrompt.encode(
                tokenizer: tokenizer, tokens: tokens,
                stateIds: D1OmniPrompt.encoded(state, tokenizer: tokenizer), question: question,
                maxLength: maxLength)
        }

        let choice = DecisionQuestion.choice(
            "Pick", options: ["a", .init("b", description: "bee")])
        var (ids, markers) = try encode("xy", choice, 16384)
        let expected: [[Int]] = [
            [1, 7], chars("xy"), [8], chars("Pick"), [9, 16], chars(" a"), [10],
            [9, 16], chars(" b: bee"), [10, 11],
        ]
        #expect(ids == Array(expected.joined()))
        #expect(markers == [10, 15])

        // Three options leave 64 - 55 - 2 = 7 tokens of state.
        let score = DecisionQuestion.score("Rate", levels: ["low", "mid", "high"])
        (ids, markers) = try encode("abcdefghijklmnop", score, 64)
        #expect(Array(ids.prefix(9)) == [1, 7] + chars("abcdefg"))
        #expect(ids.count == 64)
        #expect(markers == [15, 31, 47])

        // Instructions keep max(16, budget) = 96 tokens, each option 45.
        let long = DecisionQuestion.choice(
            String(repeating: "Q", count: 200),
            options: [.init("a", description: String(repeating: "x", count: 100)), "b"])
        (ids, markers) = try encode("s", long, 16384)
        #expect(ids.count == 153)
        #expect(markers == [100, 148])
        #expect(Array(ids[3 ..< 101]) == [8] + chars(String(repeating: "Q", count: 95)) + [9, 16])
        #expect(Array(ids[101 ..< 147]) == chars(" a: " + String(repeating: "x", count: 41)) + [10])

        // Caller text cannot forge a delimiter or marker.
        let escaped = DecisionQuestion.bool("<|mask|>?", whenTrue: "<|reserved_9|>")
        (ids, _) = try encode("<|im_end|>", escaped, 16384)
        #expect(ids.filter { $0 < 100 } == [1, 7, 8, 9, 16, 10, 9, 16, 10, 11])
        #expect(
            text(of: ids.filter { $0 >= 100 })
                == "<¦im_end¦><¦mask¦>? false: no, the statement does not hold"
                + " true: <¦reserved_9¦>")

        #expect(throws: DecisionError.optionsDoNotFit(name: "q")) {
            try D1OmniPrompt.encode(
                tokenizer: tokenizer, tokens: tokens, stateIds: [], question: .bool("q"),
                maxLength: 20, name: "q")
        }
    }

    @Test("A bounded encode gives the first ids of the full encode and stops early")
    func boundedEncodeMatchesFullEncode() {
        // Every limit moves the cuts, so they land next to every kind of space run.
        let separators = [
            " ", " ", "  ", "   ", "\n", "\n\n", " \n ", "\t ", "", "\u{A0}", "\r\n", "\r",
            "\n\t", "\n\u{85}", "\u{B}", "\u{2028}",
        ]
        let lineBreaks = ["\n", "\r\n", "\r", "\n\n", ",", ""]
        // Per scalar: a space can sit inside a character, as in " \u{301}".
        func noSpaces(_ text: String, orLineBreaks: Bool = false) -> String {
            let removed: Set<Unicode.Scalar> = orLineBreaks ? [" ", "\n", "\r"] : [" "]
            return String(
                String.UnicodeScalarView(
                    text.unicodeScalars.map { (scalar: Unicode.Scalar) -> Unicode.Scalar in
                        removed.contains(scalar) ? "_" : scalar
                    }))
        }
        // Each text, and how many pieces its bounded encodes take at most. Words longer than
        // the first guess of four characters per token need several.
        let texts: [(String, ClosedRange<Int>)] = [
            (mixedText(fragments: 300, separators: separators, seed: 1), 1 ... .max),
            (
                mixedText(fragments: 300, separators: separators, seed: 2, from: longWords),
                2 ... .max
            ),
            // No space, so only line breaks cut.
            (noSpaces(mixedText(fragments: 300, separators: lineBreaks, seed: 3)), 1 ... .max),
            (
                noSpaces(
                    mixedText(fragments: 300, separators: lineBreaks, seed: 4, from: longWords)),
                2 ... .max
            ),
            // No space and no line break: no cut point, so one piece.
            (
                noSpaces(
                    mixedText(fragments: 150, separators: [",", ""], seed: 5), orLineBreaks: true),
                1 ... 1
            ),
        ]
        for (text, pieceCounts) in texts {
            let escaped = D1OmniPrompt.escape(text)
            let scalars = Array(escaped.unicodeScalars)
            let pieces = PreTokenizedTokenizer.pieces(escaped)
            let full = D1OmniPrompt.encoded(text, tokenizer: PreTokenizedTokenizer())
            #expect(full.count == pieces.count && full.count > 200)

            // Each cut point splits the ids of the whole text in two.
            let cuts = (1 ..< scalars.count).filter { (index: Int) -> Bool in
                D1OmniPrompt.startsToken(scalars, at: index)
            }
            func encodedPart(_ part: ArraySlice<Unicode.Scalar>) -> [Int] {
                PreTokenizedTokenizer().encode(
                    text: String(String.UnicodeScalarView(part)), addSpecialTokens: false)
            }
            let badCuts = cuts.filter { (cut: Int) -> Bool in
                encodedPart(scalars[..<cut]) + encodedPart(scalars[cut...]) != full
            }
            #expect(badCuts.isEmpty, "cuts \(badCuts.prefix(10))")
            #expect(cuts.isEmpty == (pieceCounts == 1 ... 1))

            var mismatches = [Int]()
            var overreads = [Int]()
            var mostPieces = 0
            // Characters of the first `limit` pieces.
            var needed = 0
            for limit in 1 ... full.count + 2 {
                needed += limit <= pieces.count ? pieces[limit - 1].unicodeScalars.count : 0
                let tokenizer = PreTokenizedTokenizer()
                let ids = D1OmniPrompt.encoded(text, limit: limit, tokenizer: tokenizer)
                if ids != Array(full.prefix(limit)) {
                    mismatches.append(limit)
                }
                if !cuts.isEmpty, tokenizer.encodedCharacters.count > 4 * needed + 300 {
                    overreads.append(limit)
                }
                mostPieces = max(mostPieces, tokenizer.encodedCharacters.calls)
            }
            #expect(mismatches.isEmpty, "limits \(mismatches.prefix(10))")
            #expect(overreads.isEmpty, "limits \(overreads.prefix(10))")
            #expect(pieceCounts.contains(mostPieces), "\(mostPieces) pieces")
        }
        let short = PreTokenizedTokenizer()
        #expect(D1OmniPrompt.encoded("", limit: 8, tokenizer: short) == [])
        #expect(
            D1OmniPrompt.encoded("<|mask|> it's", limit: 8, tokenizer: short)
                == D1OmniPrompt.encoded("<|mask|> it's", tokenizer: short))
    }

    @Test("A long state is encoded only as far as the model reads")
    func longStateIsEncodedInPart() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let state = mixedText(fragments: 60000, separators: [" "], seed: 3)
            let questions = [
                named("a", .bool("Is it short?")),
                named("b", .choice("Pick one", options: ["a", "b", "c"])),
            ]
            let tokenizer = CharTokenizer()
            let tokens = try D1OmniPrompt.Tokens(tokenizer, fallbackBOS: 1)
            let stateIds = D1OmniPrompt.encoded(state, tokenizer: tokenizer)
            #expect(stateIds.count > 300_000)
            // Text rows read up to 16384 positions, rows after an image up to 896.
            let cases: [(MLXArray?, DecisionMediaKind?, Int)] = [
                (nil, nil, 16384), (zeros([70, 64]), .images, 896),
            ]
            for (prefix, media, maxLength) in cases {
                let counted = CharTokenizer(encodedCharacters: EncodedCharacters())
                let rows = try model.rows(
                    state: .text(state), questions: questions, prefix: prefix, media: media,
                    tokenizer: counted, tokens: tokens
                ).rows
                let expected = try questions.map {
                    (named: DecisionRequest.NamedQuestion) -> [Int] in
                    try D1OmniPrompt.encode(
                        tokenizer: tokenizer, tokens: tokens, stateIds: stateIds,
                        question: named.question, maxLength: maxLength,
                        noulDefault: media == nil ? nil : D1OmniPrompt.yesNo
                    ).ids
                }
                #expect(rows.map(\.ids) == expected, "\(maxLength)")
                #expect(rows.allSatisfy { $0.ids.count == maxLength }, "\(maxLength)")
                let read = counted.encodedCharacters?.count ?? 0
                #expect(read <= 8 * maxLength, "\(maxLength): \(read) characters")
            }
        }
    }

    @Test("Option texts, state text and temperature buckets match the reference")
    func optionTextsAndTemperatureBuckets() {
        #expect(
            D1OmniPrompt.escape("<|mask|> <|reserved_7|> <|not one|> <||>")
                == "<¦mask¦> <¦reserved_7¦> <|not one|> <||>")
        #expect(D1OmniPrompt.escape("a<|<|b_1|>|>é") == "a<|<¦b_1¦>|>é")

        let noul = DecisionQuestion.bool("Q")
        let defaults = ["false: no, the statement does not hold", "true: yes, the statement holds"]
        let cases: [(DecisionQuestion, D1OmniPrompt.BoolCriteria?, [String])] = [
            (noul, nil, defaults),
            (noul, D1OmniPrompt.yesNo, ["false: no", "true: yes"]),
            (.bool("", whenTrue: "Y", whenFalse: "N"), nil, ["false: N", "true: Y"]),
            // One given side turns the default off; the other side reads the generic text.
            (
                .bool("", whenTrue: "0", whenFalse: ""), D1OmniPrompt.yesNo,
                ["false: no, the statement does not hold", "true: 0"]
            ),
            (
                .bool("", whenTrue: "very"), D1OmniPrompt.yesNo,
                ["false: no, the statement does not hold", "true: very"]
            ),
            (
                .choice(
                    "",
                    options: [
                        .init("zero", description: "0"), "none", .init("empty", description: ""),
                    ]),
                nil, ["zero: 0", "none", "empty"]
            ),
            (
                .score("", levels: ["low", "[0, false]"]), nil,
                ["level 0: low", "level 1: [0, false]"]
            ),
        ]
        for (question, noulDefault, expected) in cases {
            #expect(D1OmniPrompt.renderOptions(question, noulDefault: noulDefault) == expected)
        }

        let letters = "abcdefghijkl".map { DecisionOption(String($0)) }
        let buckets: [(DecisionQuestion, String)] = [
            (noul, "noul:2"),
            (.choice("", options: Array(letters.prefix(3))), "choice:3-5"),
            (.score("", levels: letters.prefix(6).map(\.label)), "score:6-10"),
            (.choice("", options: letters), "choice:11+"),
            (.choice("", options: Array(letters.prefix(2))), "choice:2"),
        ]
        for (question, key) in buckets {
            #expect(D1OmniPrompt.temperatureKey(question) == key)
        }
        #expect(D1OmniPrompt.questionType(noul) == 2)
        #expect(D1OmniPrompt.optionCount(noul) == 2)

        #expect(D1OmniPrompt.stateText(nil) == "")
        #expect(D1OmniPrompt.stateText("x") == "x")
        #expect(D1OmniPrompt.stateText(.json(["k": [1, 2]])) == #"{"k": [1, 2]}"#)
    }

    // MARK: Weights

    @Test("Sanitize maps the reference checkpoint and is idempotent")
    func sanitizeConvertsReferenceCheckpoint() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = D1Omni(try tinyConfiguration())
            var expected = [String: MLXArray]()
            var start: Int32 = 0
            for (key, value) in model.parameters().flattened() {
                let count = Int32(value.size)
                expected[key] = MLXArray(Array(start ..< start + count)).asType(.float32)
                    .reshaped(value.shape)
                start += count
            }

            var source = referenceCheckpoint(expected, headLayers: 2)
            for key in [
                "encoder.layers.0.conv.conv.weight", "head.head.layers.0.self_attn.in_proj_weight",
                "head.head.layers.0.self_attn.out_proj.bias", "head.scorer.3.weight",
                "vision.tower.vision_model.post_layernorm.weight",
            ] {
                #expect(source[key] != nil, "\(key)")
            }
            #expect(source["encoder.layers.0.conv.conv.weight"]?.shape == [64, 1, 3])
            #expect(source["head.head.layers.0.self_attn.in_proj_weight"]?.shape == [192, 64])
            source["vision.tower.vision_model.embeddings.position_ids"] = MLXArray(0 ..< 16)

            func check(_ converted: [String: MLXArray]) {
                #expect(Set(converted.keys) == Set(expected.keys))
                for (key, value) in expected {
                    guard let actual = converted[key] else { continue }
                    #expect(actual.shape == value.shape, "\(key)")
                    #expect(arrayEqual(actual, value).item(Bool.self), "\(key)")
                }
            }
            let converted = model.sanitize(weights: source)
            check(converted)
            check(model.sanitize(weights: converted))
            try model.update(parameters: ModuleParameters.unflattened(converted), verify: [.all])
        }
    }

    // MARK: Trunk

    @Test("The short convolution taps left, centre and right, with keep_right")
    func shortConvTapsLeftCentreRight() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let conv = try #require(model.encoder.layers.compactMap(\.conv).first)
            let taps = broadcast(
                MLXArray([1, 10, 100] as [Float]).reshaped(1, 3, 1), to: conv.conv.weight.shape)
            try conv.conv.update(
                parameters: ModuleParameters.unflattened(["weight": taps]), verify: [.all])

            let x = MLXRandom.normal([1, 6, 64], key: MLXRandom.key(0))
            // keep_right of a 3-position prefix.
            let keep = MLXArray([1, 1, 0, 1, 1, 1] as [Float]).reshaped(1, 6)
            let y = conv(x, pad: ones([1, 6]), keepRight: keep)

            let bcu = split(matmul(x, conv.inProj.weight.T), parts: 3, axis: -1)
            let z = padded(bcu[0] * bcu[2], widths: [0, [1, 1], 0])
            let mixed =
                z[0..., 0 ..< 6] + 10 * z[0..., 1 ..< 7]
                + 100 * expandedDimensions(keep, axis: -1) * z[0..., 2 ..< 8]
            let expected = matmul(bcu[1] * mixed, conv.outProj.weight.T)
            #expect(maxAbsDiff(y, expected) <= 1e-5 * abs(expected).max().item(Float.self))

            // The text after a 3-position prefix that ran alone reads it through `left`.
            let text = conv(
                x[0..., 3...], pad: ones([1, 3]), keepRight: ones([1, 3]),
                left: conv.tapInput(x[0..., 2 ..< 3]))
            #expect(
                maxAbsDiff(text, expected[0..., 3...])
                    <= 1e-5 * abs(expected).max().item(Float.self))
        }
    }

    @Test("A media prefix is encoded the same whatever text follows it")
    func mediaPrefixReadsOnlyMedia() throws {
        try Device.withDefaultDevice(.cpu) {
            let trunk = try randomD1Omni().encoder
            let media = MLXRandom.normal([5, 64], key: MLXRandom.key(1))
            func run(_ ids: [Int32]) -> MLXArray {
                let h = concatenated([media, trunk.embedTokens(MLXArray(ids))])
                return trunk(
                    expandedDimensions(h, axis: 0), pad: ones([1, h.dim(0)], dtype: .bool),
                    prefix: [5])[0]
            }
            let first = run([1, 2, 3, 4])
            let second = run([9, 8, 7, 6])
            #expect(arrayEqual(first[..<5], second[..<5]).item(Bool.self))
            #expect(!allClose(first[5...], second[5...]).item(Bool.self))
        }
    }

    // MARK: Batching

    @Test("Batched text rows equal single rows and one-row batches")
    func batchedEqualsSingle() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let tokenizer = CharTokenizer()
            let tokens = try D1OmniPrompt.Tokens(tokenizer, fallbackBOS: 1)
            let q1 = named("a", .bool("Is it short?"))
            let q2 = named("b", .choice("Pick one", options: ["a", "b", "c"]))
            let q3 = named("c", .score("Rate", levels: ["low", "mid", "high"]))
            let requests: [(DecisionState?, [DecisionRequest.NamedQuestion])] = [
                (.text(String(repeating: "x", count: 300)), [q1, q2]),
                ("hi", [q3]),
                (.json(["k": [1, 2]]), [q2]),
            ]
            let perRequest = try requests.map {
                try model.rows(
                    state: $0.0, questions: $0.1, prefix: nil, media: nil, tokenizer: tokenizer,
                    tokens: tokens)
            }
            let rows = perRequest.flatMap(\.rows)

            let batched = try model.run(rows, tokenBudget: 65536)
            let alone = try perRequest.flatMap { try model.run($0.rows, tokenBudget: 65536) }
            let single = try model.run(rows, tokenBudget: 1)
            #expect(maxAbsDiff(batched, alone) <= 1e-5)
            #expect(maxAbsDiff(batched, single) <= 1e-5)
            #expect(batched.map(\.count) == [2, 3, 3, 3])
            #expect(rows.map(\.calibrate) == [true, true, true, true])
        }
    }

    @Test("Row batches match mlx-vlm#2500's _batches")
    func batchesMatchPython() {
        let texts = [30, 10, 16, 15, 31]
        let long = [1127] + Array(repeating: 235, count: 63)
        let longAt30 =
            Array(repeating: 235, count: 30) + [1127] + Array(repeating: 235, count: 33)
        // Python's batches, each sorted into request order.
        let cases: [(lengths: [Int], trunk: Int, budget: Int, expected: [[Int]])] = [
            (texts, 0, 65536, [[0, 2, 4], [1, 3]]),
            (texts, 0, 64, [[0, 4], [1, 2, 3]]),
            (texts, 0, 1, [[4], [0], [2], [3], [1]]),
            // After a shared prefix of 70 positions, only the text counts for padding.
            (texts, 70, 65536, [[0, 2, 4], [1, 3]]),
            (texts.map { $0 + 70 }, 0, 65536, [[0, 1, 2, 3, 4]]),
            (texts, 70, 300, [[0, 4], [1, 2, 3]]),
            (long, 0, 16384, [[0], Array(1 ..< 64)]),
            (longAt30, 0, 16384, [[30], Array(0 ..< 30) + Array(31 ..< 64)]),
            ([], 0, 10, []),
        ]
        for (lengths, trunk, budget, expected) in cases {
            #expect(D1Prompt.chunks(lengths: lengths, trunk: trunk, budget: budget) == expected)
        }
    }

    @Test("Text rows run in batches of similar length")
    func textRowsRunInBatchesOfSimilarLength() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let norm = try recordTrunkShapes(model)
            let tokenizer = CharTokenizer()
            let (rows, _) = try model.rows(
                state: "hi", questions: Self.mixedQuestions, prefix: nil, media: nil,
                tokenizer: tokenizer, tokens: D1OmniPrompt.Tokens(tokenizer, fallbackBOS: 1))
            let lengths = rows.map(\.length)
            let batches = D1Prompt.chunks(lengths: lengths, trunk: 0, budget: 65536)
            try #require(batches.count > 1)

            let together = try model.run(rows, tokenBudget: 65536)
            let shapes = batches.map { (batch: [Int]) -> [Int] in
                [batch.count, batch.map { lengths[$0] }.max() ?? 0, 64]
            }
            #expect(norm.shapes == shapes)
            let single = try model.run(rows, tokenBudget: 1)
            #expect(maxAbsDiff(together, single) <= 1e-5)
        }
    }

    @Test("Calibration divides by temperatures[type:bucket], else temperatures[type], else 1")
    func calibrationPrefersTheBucket() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let tokenizer = CharTokenizer()
            let tokens = try D1OmniPrompt.Tokens(tokenizer, fallbackBOS: 1)
            let (rows, _) = try model.rows(
                state: "hi",
                questions: [
                    named("pick", .choice("Pick one", options: ["a", "b"])),
                    named("rate", .score("Rate", levels: ["low", "mid", "high"])),
                ],
                prefix: nil, media: nil, tokenizer: tokenizer, tokens: tokens)
            // One-row batches on both sides, so only the temperature differs.
            let logits = rows.map { model.logits([$0])[0] }
            func expected(_ choice: Float, _ score: Float) -> [[Double]] {
                zip(logits, [(2, choice), (3, score)]).map { z, option in
                    softmax(z[0 ..< option.0] / option.1, axis: -1)
                        .asArray(Float.self).map { Double($0) }
                }
            }
            let cases: [([String: Double], [[Double]])] = [
                (
                    ["choice:2": 1.5, "choice": 3, "score:3-5": 0.5, "score": 4],
                    expected(1.5, 0.5)
                ),
                (["choice:3-5": 9, "choice": 3, "score:6-10": 9, "score": 4], expected(3, 4)),
                (["noul:2": 9, "noul": 9], expected(1, 1)),
            ]
            #expect(maxAbsDiff(cases[0].1, cases[1].1) > 1e-3)
            #expect(maxAbsDiff(cases[1].1, cases[2].1) > 1e-3)
            for (temperatures, expected) in cases {
                model.configuration.temperatures = temperatures
                let difference = maxAbsDiff(try model.run(rows, tokenBudget: 1), expected)
                #expect(difference <= 1e-6, "\(temperatures)")
            }
        }
    }

    @Test("A row after a media prefix reads the trunk at the text positions")
    func prefixRowsReadTheirText() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let tokenizer = CharTokenizer()
            let tokens = try D1OmniPrompt.Tokens(tokenizer, fallbackBOS: 1)
            let image = MLXRandom.normal([7, 64], key: MLXRandom.key(4))
            let imageRows = try model.rows(
                state: "hi", questions: [named("pick", .choice("Pick one", options: ["a", "b"]))],
                prefix: image, media: .images, tokenizer: tokenizer, tokens: tokens
            ).rows
            let textRows = try model.rows(
                state: "a longer state", questions: [named("done", .bool("Done?"))], prefix: nil,
                media: nil, tokenizer: tokenizer, tokens: tokens
            ).rows
            let rows = imageRows + textRows

            // Each row alone: trunk, then positions P..<P+n, then the head.
            func reference(_ row: D1OmniRow) -> MLXArray {
                var h = model.encoder.embedTokens(MLXArray(row.ids.map { Int32($0) }))
                if let prefix = row.prefix {
                    h = concatenated([prefix, h])
                }
                let trunk = model.encoder(
                    expandedDimensions(h, axis: 0), pad: ones([1, h.dim(0)], dtype: .bool),
                    prefix: [row.prefixLength])
                return model.head(
                    trunk[0..., row.prefixLength...], pad: ones([1, row.ids.count], dtype: .bool),
                    markerPositions: MLXArray(
                        row.markers.map { Int32($0) }, [1, row.markers.count]),
                    markerMask: ones([1, row.markers.count], dtype: .bool),
                    questionTypes: MLXArray([Int32(D1OmniPrompt.questionType(row.question))]))[0]
            }
            let batched = model.logits(rows)
            for (index, row) in rows.enumerated() {
                let expected = reference(row)
                let scale = max(1, abs(expected).max().item(Float.self))
                #expect(
                    maxAbsDiff(batched[index, 0 ..< row.markers.count], expected)
                        <= 1e-5 * scale)
            }
        }
    }

    /// Questions whose prompts differ in length, so batches of them are padded.
    private static let mixedQuestions = [
        named("a", .bool("Is it short?")),
        named("b", .choice("Pick the one that fits best", options: ["a", "b", "c"])),
        named("c", .score("Rate", levels: ["low", "mid", "high"])),
        named("d", .bool("Q?", whenTrue: "it is")),
        named("e", .choice(String(repeating: "long ", count: 30), options: ["x", "y"])),
    ]

    /// Two image requests (70 prefix rows each), each with `mixedQuestions`.
    private static func mediaRequests() -> [DecisionRequest] {
        [
            DecisionRequest(
                state: "a state", questions: mixedQuestions,
                images: [randomPixels(width: 64, height: 48, seed: 0).userImage]),
            DecisionRequest(
                state: .json(["k": [1, 2]]), questions: mixedQuestions,
                images: [randomPixels(width: 48, height: 64, seed: 1).userImage]),
        ]
    }

    /// Swaps the first trunk layer's norm for one that records the shapes it sees.
    private func recordTrunkShapes(_ model: D1Omni) throws -> RecordingNorm {
        let layer = model.encoder.layers[0]
        let norm = RecordingNorm(dimensions: 64, eps: layer.operatorNorm.eps)
        try norm.update(parameters: layer.operatorNorm.parameters(), verify: [.all])
        layer.update(modules: .unflattened(["operator_norm": norm]))
        return norm
    }

    @Test("Several questions about one image run its prefix through the trunk once")
    func sharedPrefixRunsOnce() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let norm = try recordTrunkShapes(model)
            for (rows, _) in try await requestRows(model, Self.mediaRequests()) {
                let P = rows[0].prefixLength
                let texts = rows.map(\.ids.count)
                let batches = D1Prompt.chunks(lengths: texts, trunk: P, budget: 65536)
                #expect(batches.count > 1, "P=\(P)")
                let textShapes = batches.map { (batch: [Int]) -> [Int] in
                    [batch.count, batch.map { texts[$0] }.max() ?? 0, 64]
                }
                norm.shapes = []
                let together = try model.run(rows, tokenBudget: 65536)
                // The prefix once, then the text in batches of similar length.
                #expect(norm.shapes == [[1, P, 64]] + textShapes, "P=\(P)")

                // One question runs with its prefix.
                norm.shapes = []
                let alone = try rows.map { (row: D1OmniRow) -> [Double] in
                    try model.run([row], tokenBudget: 65536)[0]
                }
                #expect(
                    norm.shapes == rows.map { (row: D1OmniRow) -> [Int] in [1, row.length, 64] },
                    "P=\(P)")
                #expect(maxAbsDiff(together, alone) <= 1e-5, "P=\(P)")
            }
        }
    }

    @Test("A cancelled request throws before the trunk runs")
    func cancelledRunSkipsTheTrunk() async throws {
        let task = Task { () async throws -> [[Int]] in
            try await Device.withDefaultDevice(.cpu) { () async throws -> [[Int]] in
                let model = try randomD1Omni()
                let norm = try recordTrunkShapes(model)
                let rows = try await requestRows(model, Self.mediaRequests())[0].rows
                withUnsafeCurrentTask { (task: UnsafeCurrentTask?) in task?.cancel() }
                #expect(throws: CancellationError.self) {
                    _ = try model.run(rows, tokenBudget: 65536)
                }
                return norm.shapes
            }
        }
        #expect(try await task.value == [])
    }

    /// Checks that rows after `sharedPrefixCache` give the logits and probabilities of rows
    /// that each carry the prefix, on the default device.
    private func expectSharedPrefixMatchesPrefixedRows() async throws {
        let model = try randomD1Omni()
        let perRequest = try await requestRows(model, Self.mediaRequests())
        for (rows, _) in perRequest {
            let P = rows[0].prefixLength
            #expect(Set(rows.map(\.ids.count)).count == rows.count, "P=\(P)")

            // Every row carries the prefix through the trunk, in one padded batch.
            let expected = model.logits(rows)
            let cache = try #require(model.sharedPrefixCache(rows))
            #expect(cache.length == P && cache.layers.count == model.encoder.layers.count)
            let logits = model.logits(rows, after: cache)
            #expect(logits.dtype == .float32)
            for (index, row) in rows.enumerated() {
                let options = 0 ..< row.markers.count
                let scale = max(1, abs(expected[index, options]).max().item(Float.self))
                #expect(
                    maxAbsDiff(logits[index, options], expected[index, options])
                        <= 1e-4 * scale, "P=\(P) row \(index)")
            }
            // The prefix reaches the text: without it the logits are others.
            let bare = rows.map { (row: D1OmniRow) -> D1OmniRow in
                var row = row
                row.prefix = nil
                return row
            }
            #expect(maxAbsDiff(model.logits(bare), expected) > 1e-2, "P=\(P)")

            // All rows in a batch, two in a batch and one in a batch.
            let alone = try rows.map { (row: D1OmniRow) -> [Double] in
                try model.run([row], tokenBudget: 65536)[0]
            }
            let longest = rows.map(\.length).max() ?? 0
            for budget in [65536, 2 * longest] {
                let shared = try model.run(rows, tokenBudget: budget)
                #expect(maxAbsDiff(shared, alone) <= 1e-5, "P=\(P) budget \(budget)")
            }
        }

        // One row, rows without a prefix and rows after different prefixes share nothing.
        let (image, other) = (perRequest[0].rows, perRequest[1].rows)
        #expect(model.sharedPrefixCache([image[0]]) == nil)
        #expect(model.sharedPrefixCache(image + other) == nil)
        let text = try model.rows(
            state: "hi", questions: Self.mixedQuestions, prefix: nil, media: nil,
            tokenizer: CharTokenizer(),
            tokens: D1OmniPrompt.Tokens(CharTokenizer(), fallbackBOS: 1)
        ).rows
        #expect(model.sharedPrefixCache(text) == nil)

        // A half-precision trunk reads the prefix in its own type.
        let half = model.encoder.parameters().flattened().map { ($0.0, $0.1.asType(.float16)) }
        try model.encoder.update(
            parameters: ModuleParameters.unflattened(half), verify: [.all])
        let cache = try #require(model.sharedPrefixCache(image))
        #expect(cache.arrays.allSatisfy { $0.dtype == .float16 })
        let logits = model.logits(image, after: cache)
        let expected = model.logits(image)
        #expect(logits.dtype == .float32)
        #expect(maxAbsDiff(logits, expected) <= 1e-2)
    }

    @Test("Rows after a shared image prefix match rows that each carry it")
    func sharedPrefixMatchesPrefixedRows() async throws {
        try await Device.withDefaultDevice(.cpu) { () async throws in
            try await expectSharedPrefixMatchesPrefixedRows()
        }
    }

    @Test(
        "On the GPU, rows after a shared prefix match rows that each carry it",
        .enabled(if: Device.defaultDevice().deviceType == .gpu, "needs a GPU"))
    func sharedPrefixMatchesPrefixedRowsOnTheGPU() async throws {
        try await expectSharedPrefixMatchesPrefixedRows()
    }

    @Test("A trunk pass that waits for each layer gives the same logits after a shared prefix")
    func longPassAfterSharedPrefix() async throws {
        // `waitEachLayerPositions` or more; many short rows keep the attention cost low.
        let model = try randomD1Omni()
        let questions = (0 ..< 24).map { (index: Int) -> DecisionRequest.NamedQuestion in
            named("q\(index)", .bool("Is it \(index)?"))
        }
        let long = try await requestRows(
            model,
            [
                DecisionRequest(
                    state: .text(String(repeating: "a long state ", count: 26)),
                    questions: questions,
                    images: [randomPixels(width: 64, height: 48, seed: 0).userImage])
            ]
        )[0].rows
        let positions = long.count * (long.map(\.ids.count).max() ?? 0)
        #expect(positions >= D1OmniTrunk.waitEachLayerPositions)
        let longCache = try #require(model.sharedPrefixCache(long))
        let longExpected = model.logits(long)
        let longScale = max(1, abs(longExpected).max().item(Float.self))
        #expect(
            maxAbsDiff(model.logits(long, after: longCache), longExpected)
                <= 1e-4 * longScale)
    }

    @Test("Media longer than the context is rejected")
    func mediaTooLong() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            model.configuration.maxLength = 200
            let tokenizer = CharTokenizer()
            let tokens = try D1OmniPrompt.Tokens(tokenizer, fallbackBOS: 1)
            #expect(throws: DecisionError.mediaTooLong(positions: 137, maximum: 200)) {
                try model.rows(
                    state: nil, questions: [named("q", .bool("q"))], prefix: zeros([137, 64]),
                    media: .images, tokenizer: tokenizer, tokens: tokens)
            }
            let (rows, _) = try model.rows(
                state: nil, questions: [named("q", .bool("q"))], prefix: zeros([136, 64]),
                media: .images, tokenizer: tokenizer, tokens: tokens)
            #expect(rows[0].ids.count <= 64)
        }
    }

    // MARK: Requests

    @Test("Text requests answer through the decision container")
    func textRequestsAnswer() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let tokenizer = CharTokenizer()
            let questions: KeyValuePairs<String, DecisionQuestion> = [
                "finished": .bool("Finished"),
                "route": .choice("Route", options: ["A", "B"]),
                "quality": .score("Rate", levels: ["low", "mid", "high"]),
            ]
            let output = try await model.decisionProbabilities(
                DecisionRequest(state: "A B", questions: questions), tokenizer: tokenizer,
                options: .init())

            let tokens = try D1OmniPrompt.Tokens(tokenizer, fallbackBOS: 1)
            let (rows, inputTokens) = try model.rows(
                state: "A B", questions: questions.map { named($0.key, $0.value) }, prefix: nil,
                media: nil, tokenizer: tokenizer, tokens: tokens)
            let expected = try model.run(rows, tokenBudget: 65536)
            #expect(output.inputTokens == inputTokens)
            #expect(output.probabilities == expected)
            #expect(output.probabilities.map(\.count) == [2, 2, 3])
            for probabilities in output.probabilities {
                #expect(abs(probabilities.reduce(0, +) - 1) < 1e-5)
            }

            let container = DecisionModelContainer(model: model, tokenizer: tokenizer)
                .decisionContainer(modelID: "d1-omni-test")
            let response = try await container.predict(state: "A B", questions: questions)
            // Bool output is [P(false), P(true)].
            #expect(response["finished"]?.probabilities == output.probabilities[0])
            #expect(response["finished"]?.value == .bool(output.probabilities[0][1] >= 0.5))
            #expect(response["route"]?.probabilities == output.probabilities[1])
            #expect(response.inputTokens == inputTokens)
        }
    }

    // MARK: Images

    @Test("Image preprocessing matches the reference")
    func imagePreprocessingMatchesReference() throws {
        try Device.withDefaultDevice(.cpu) {
            let bytes = (0 ..< 64 * 48).flatMap { _ -> [UInt8] in [255, 0, 127] }
            let inputs = try d1OmniPatches(
                OrientedImage(RGB8Image(width: 64, height: 48, bytes: bytes)))
            #expect(inputs.pixels.shape == [1, 1024, 768])
            #expect(inputs.spatialShapes.map { [$0.0, $0.1] } == [[14, 20]])
            let values = inputs.pixels.asArray(Float.self)
            #expect(Array(values.prefix(6)) == [1, -1, -0.5 / 127.5, 1, -1, -0.5 / 127.5])
            #expect(values[(280 * 768)...].allSatisfy { $0 == 0 })
        }
    }

    @Test("The shared bilinear resize matches the reference's torchvision port")
    func resizeMatchesReference() {
        // (in, out, bits, first row) from d1_omni `_taps`. ATen's rows are at least as long.
        let tapCases: [(Int, Int, Int, [Int32])] = [
            (640, 576, 15, [28300, 4468, 0]),
            (
                4032, 576, 17,
                [12193, 15241, 18289, 21337, 18289, 15241, 12193, 9145, 6096, 3048, 0, 0, 0, 0]
            ),
            (53, 20, 16, [19137, 25952, 15466, 4981, 0, 0]),
            (100, 300, 14, [16384, 0]),
            (16, 14, 15, [27307, 5461, 0]),
        ]
        for (n, m, bits, row) in tapCases {
            let taps = ResampleTaps(inSize: n, outSize: m, filter: .bilinear)
            #expect(taps.shift == bits && taps.start[0] == 0, "\(n) -> \(m)")
            #expect(Array(taps.weights.prefix(row.count)) == row, "\(n) -> \(m)")
        }
        // (inW, inH, outW, outH, byte sum, first 6 bytes) from d1_omni `resize`.
        let resizeCases: [(Int, Int, Int, Int, Int, [UInt8])] = [
            (300, 200, 224, 320, 27_419_189, [9, 68, 127, 47, 106, 165]),
            (1001, 97, 512, 32, 6_265_889, [63, 118, 154, 127, 163, 138]),
            (64, 48, 320, 224, 27_544_056, [0, 59, 118, 0, 59, 118]),
        ]
        for (w, h, ow, oh, sum, first) in resizeCases {
            let out = RGB8Image.testPattern(width: w, height: h)
                .resized(width: ow, height: oh, filter: .bilinear)
            #expect(out.width == ow && out.height == oh)
            #expect(out.byteSum == sum, "\(w)x\(h) -> \(ow)x\(oh)")
            #expect(Array(out.bytes.prefix(6)) == first, "\(w)x\(h) -> \(ow)x\(oh)")
        }
    }

    @Test("The image prefix is each crop through the tower and projector; tiles come first")
    func imagePrefixMatchesEachCrop() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            // Two 512x512 tiles, then a 704x352 thumbnail of 968 patches, which is padded.
            let image = randomPixels(width: 1200, height: 600, seed: 11)
            let inputs = try d1OmniPatches(OrientedImage(image))
            #expect(
                inputs.spatialShapes.map { [$0.0, $0.1] } == [[32, 32], [32, 32], [22, 44]])
            let prefix = try model.vision([OrientedImage(image)]) { $0 }
            #expect(prefix.shape == [256 + 256 + 11 * 22, 64])
            let scale = max(1, abs(prefix).max().item(Float.self))

            var offset = 0
            for (index, (h, w)) in inputs.spatialShapes.enumerated() {
                let alone = model.vision.tower(
                    inputs.pixels[index ..< index + 1, 0 ..< h * w], spatialShapes: [(h, w)],
                    mask: nil)
                let expected = model.vision.projector(alone.reshaped(1, h, w, -1))
                let rows = prefix[offset ..< offset + expected.dim(0)]
                #expect(maxAbsDiff(rows, expected) <= 1e-5 * scale, "crop \(index)")
                offset += expected.dim(0)
            }

            // Several images: each image's crops, in order.
            let small = randomPixels(width: 64, height: 48, seed: 12)
            let both = try model.vision([OrientedImage(small), OrientedImage(image)]) { $0 }
            #expect(both.shape == [70 + prefix.dim(0), 64])
            #expect(maxAbsDiff(both[70...], prefix) <= 1e-5 * scale)
        }
    }

    @Test("Images are decoded one at a time and give the prefix of images all decoded first")
    func imagesAreDecodedOneAtATime() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let sizes = [(64, 48), (1200, 600), (300, 1400)]
            func image(_ index: Int) -> RGB8Image {
                randomPixels(width: sizes[index].0, height: sizes[index].1, seed: UInt64(index))
            }
            // `last` shares the bytes of the image decoded last. Writing to it copies them
            // only while something else still holds that image.
            var last: [UInt8] = []
            var held: [Bool] = []
            let decode = { (index: Int) -> OrientedImage in
                if !last.isEmpty {
                    let before = last.withUnsafeBufferPointer { $0.baseAddress }
                    last.withUnsafeMutableBufferPointer { _ in }
                    held.append(last.withUnsafeBufferPointer { $0.baseAddress } != before)
                }
                let decoded = image(index)
                last = decoded.bytes
                return OrientedImage(decoded)
            }
            let prefix = try model.vision(Array(sizes.indices), decode: decode)
            #expect(held == [false, false])

            // The rows of decoding every image first and then running each, as before.
            let decoded = sizes.indices.map { OrientedImage(image($0)) }
            let before = concatenated(try decoded.map { model.vision(try d1OmniPatches($0)) })
            #expect(arrayEqual(prefix, before).item(Bool.self))

            // A request gives the same prefix.
            let (fromRequest, media) = try await model.mediaPrefix(
                images: sizes.indices.map { image($0).userImage })
            #expect(media == .images)
            #expect(arrayEqual(try #require(fromRequest), before).item(Bool.self))
        }
    }

    @Test("Image questions batch like single rows, read 70 prefix rows each, and skip calibration")
    func imageDecisionsBatchingAndUsage() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let pixels = randomPixels(width: 64, height: 48, seed: 0)
            let q1 = named("a", .bool("Is it short?"))
            let q2 = named("b", .choice("Pick one", options: ["a", "b", "c"]))
            let q3 = named("c", .score("Rate", levels: ["low", "mid", "high"]))
            let requests = [
                DecisionRequest(
                    state: .text(String(repeating: "x", count: 300)), questions: [q1, q2]),
                DecisionRequest(state: "hi", questions: [q3]),
                DecisionRequest(questions: [q1, q2], images: [pixels.userImage]),
                DecisionRequest(
                    state: .json(["k": [1, 2]]), questions: [q2],
                    images: [pixels.userImage]),
            ]
            let perRequest = try await requestRows(model, requests)
            let rows = perRequest.flatMap(\.rows)
            let batched = try model.run(rows, tokenBudget: 65536)
            let alone = try perRequest.flatMap { try model.run($0.rows, tokenBudget: 65536) }
            let single = try model.run(rows, tokenBudget: 1)
            #expect(maxAbsDiff(batched, alone) <= 1e-5)
            #expect(maxAbsDiff(batched, single) <= 1e-5)

            // Every question reads the 7 x 10 image prefix; image bools say yes/no.
            let tokenizer = CharTokenizer()
            let tokens = try D1OmniPrompt.Tokens(tokenizer, fallbackBOS: 1)
            let texts = try [q1, q2].map {
                try D1OmniPrompt.encode(
                    tokenizer: tokenizer, tokens: tokens, stateIds: [], question: $0.question,
                    maxLength: 896, noulDefault: D1OmniPrompt.yesNo
                ).ids
            }
            #expect(perRequest[2].rows.map(\.ids) == texts)
            #expect(perRequest[2].rows.map(\.prefixLength) == [70, 70])
            #expect(perRequest[2].inputTokens == 2 * 70 + texts.map(\.count).reduce(0, +))

            // Calibration applies to text rows only.
            model.configuration.temperatures = [:]
            let uncalibrated = try model.run(Array(rows[2 ..< 5]), tokenBudget: 65536)
            #expect(maxAbsDiff([uncalibrated[0]], [batched[2]]) > 1e-3)
            #expect(maxAbsDiff(Array(uncalibrated[1...]), Array(batched[3 ..< 5])) <= 1e-5)
            model.configuration = try tinyConfiguration()

            let container = DecisionModelContainer(model: model, tokenizer: tokenizer)
                .decisionContainer(modelID: "d1-omni-test")
            let questions: KeyValuePairs<String, DecisionQuestion> = [
                "a": q1.question, "b": q2.question, "c": q3.question,
            ]
            let response = try await container.predict(
                state: nil, questions: questions, images: [pixels.userImage])
            let same = try await model.decisionProbabilities(
                DecisionRequest(questions: [q1, q2, q3], images: [pixels.userImage]),
                tokenizer: tokenizer, options: .init())
            #expect(response.answers.map(kind) == [.bool, .choice, .score])
            #expect(response.answers.map(\.probabilities) == same.probabilities)
            #expect(response.inputTokens == same.inputTokens)
            let oneByOne = try await container.predict(
                state: nil, questions: ["a": q1.question, "b": q2.question],
                images: [pixels.userImage], options: DecisionExecutionOptions(tokenBudget: 1))
            #expect(abs((oneByOne["a"]?.probabilities[1] ?? 0) - same.probabilities[0][1]) < 1e-5)
        }
    }

    @Test("The trunk reads the media prefix cast to its own dtype")
    func mediaPrefixIsCastToTheTrunkType() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let half = model.encoder.parameters().flattened().map { ($0.0, $0.1.asType(.float16)) }
            try model.encoder.update(
                parameters: ModuleParameters.unflattened(half), verify: [.all])
            let tokenizer = CharTokenizer()
            let tokens = try D1OmniPrompt.Tokens(tokenizer, fallbackBOS: 1)
            let prefix = try model.vision([
                OrientedImage(randomPixels(width: 64, height: 48, seed: 3))
            ]) { $0 }
            #expect(prefix.dtype == .float32)
            func logits(_ prefix: MLXArray) throws -> MLXArray {
                let rows = try model.rows(
                    state: "s", questions: [named("q", .bool("q"))], prefix: prefix,
                    media: .images, tokenizer: tokenizer, tokens: tokens
                ).rows
                return model.logits(rows)
            }
            let fromFloat32 = try logits(prefix)
            #expect(fromFloat32.dtype == .float32)
            #expect(arrayEqual(fromFloat32, try logits(prefix.asType(.float16))).item(Bool.self))
        }
    }

    @Test("EXIF orientation applies to files only")
    func exifOrientationAppliesToFilesOnly() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            let stored = randomPixels(width: 96, height: 48, seed: 1)
            let turned = stored.oriented(6)
            let png = try pngWithOrientation6(stored)
            let directory = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let file = directory.appending(component: "photo.png")
            try png.write(to: file)
            #expect(try RGB8Image(.url(file)) == turned)

            func answer(_ image: UserInput.Image) async throws -> [[Double]] {
                try await model.decisionProbabilities(
                    DecisionRequest(questions: [named("q", .bool("Upright?"))], images: [image]),
                    tokenizer: CharTokenizer(), options: .init()
                ).probabilities
            }
            // A decoded image is used as given, like a PIL image; a file is turned.
            let opened = try #require(CIImage(data: png))
            let asGiven = try await answer(.ciImage(opened))
            let upright = try await answer(.url(file))
            #expect(asGiven == (try await answer(stored.userImage)))
            #expect(upright == (try await answer(turned.userImage)))
            #expect(asGiven != upright)
        }
    }

    /// Checks that a file stored sideways with EXIF orientation 6 and an EXIF 1 file of the
    /// same upright picture give the same patches.
    private func expectTurnedFileMatchesUprightFile(onGPU: Bool = false) throws {
        let stored = randomPixels(width: 1024, height: 768, seed: 2)
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let turnedFile = directory.appending(component: "turned.tiff")
        let uprightFile = directory.appending(component: "upright.tiff")
        try encoded(stored, as: .tiff, orientation: 6).write(to: turnedFile)
        try encoded(stored.oriented(6), as: .tiff, orientation: 1).write(to: uprightFile)

        let turned = try OrientedImage(.url(turnedFile))
        let upright = try OrientedImage(.url(uprightFile))
        #expect(turned.orientation == 6 && turned.stored == stored)
        #expect(upright.orientation == 1 && upright.stored == stored.oriented(6))
        // Without the GPU the image is made upright once, not once per resize.
        let input = d1OmniResizeInput(turned)
        #expect(input.orientation == (onGPU ? 6 : 1))
        #expect(input.stored == (onGPU ? stored : stored.oriented(6)))
        #if canImport(Metal)
        if onGPU {
            let settings = LFM2VLImageSettings.d1Omni
            let layout = try LFM2VLImageLayout.plan(width: 768, height: 1024, settings: settings)
            let sizes = [
                (layout.gridColumns * settings.tileSize, layout.gridRows * settings.tileSize),
                (layout.thumbnailWidth, layout.thumbnailHeight),
            ]
            for (width, height) in sizes {
                let passes = stored.resizePasses(
                    width: width, height: height, filter: settings.resample, orientation: 6)
                #expect(stored.prefersGPU(passes, orientation: 6))
            }
        }
        #endif
        let patches = try d1OmniPatches(turned)
        let expected = try d1OmniPatches(upright)
        #expect(patches.spatialShapes.count > 1)
        #expect(
            patches.spatialShapes.map { [$0.0, $0.1] }
                == expected.spatialShapes.map { [$0.0, $0.1] })
        #expect(arrayEqual(patches.pixels, expected.pixels).item(Bool.self))
    }

    @Test("A file with EXIF orientation 6 gives the patches of the upright file")
    func turnedFileMatchesUprightFile() throws {
        try Device.withDefaultDevice(.cpu) { () throws in
            try expectTurnedFileMatchesUprightFile()
        }
    }

    #if canImport(Metal)
    @Test(
        "The GPU resize reads a file with EXIF orientation 6 as the upright file",
        .enabled(if: Device.defaultDevice().deviceType == .gpu, "needs a GPU"))
    func turnedFileMatchesUprightFileOnTheGPU() throws {
        try expectTurnedFileMatchesUprightFile(onGPU: true)
    }
    #endif

    @Test("Bad media throws the reference's errors")
    func mediaErrors() async throws {
        let model = try Device.withDefaultDevice(.cpu) { try randomD1Omni() }

        func error(images: [UserInput.Image]) async -> DecisionError? {
            let request = DecisionRequest(questions: [named("q", .bool("q"))], images: images)
            do {
                _ = try await model.decisionProbabilities(
                    request, tokenizer: CharTokenizer(), options: .init())
                return nil
            } catch {
                return error as? DecisionError
            }
        }
        let failure = await error(images: [.array(zeros([0, 8, 3], dtype: .uint8))])
        if case .invalidImage = failure {
        } else {
            Issue.record("empty image: \(String(describing: failure))")
        }
    }

    // MARK: Loading

    @Test("The factory loads a reference-layout checkpoint without its audio tensors")
    func factoryLoadsTinyCheckpoint() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let model = try randomD1Omni()
            var weights = referenceCheckpoint(
                Dictionary(uniqueKeysWithValues: model.parameters().flattened()), headLayers: 2)
            // The released checkpoint also holds the audio encoder.
            weights["audio.encoder.pre_encode.conv.0.weight"] = MLXArray.zeros([4, 1, 3, 3])
            weights["audio.residual.down.weight"] = MLXArray.zeros([8, 64])
            weights["audio.encoder.layers.0.conv.batch_norm.num_batches_tracked"] = MLXArray(
                Int32(3))
            weights["vision.tower.vision_model.embeddings.position_ids"] = MLXArray(
                Int32(0) ..< Int32(16))
            let directory = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            try Data(tinyOmniConfigJSON.utf8).write(
                to: directory.appending(component: "config.json"))
            try save(arrays: weights, url: directory.appending(component: "model.safetensors"))

            let loader = CharTokenizerLoader()
            let container = try await DecisionModelFactory.shared.loadContainer(
                from: directory, using: loader)
            #expect(await loader.directories == [directory])
            #expect(container.modelID == ResolvedModelConfiguration(directory: directory).name)
            #expect(container.capabilities == model.decisionCapabilities)
            #expect(container.capabilities.media == [.images])

            let pixels = randomPixels(width: 64, height: 48, seed: 0)
            let questions: KeyValuePairs<String, DecisionQuestion> = [
                "done": .bool("Finished?"),
                "route": .choice("Route", options: ["A", .init("B", description: "the other")]),
                "quality": .score("Rate", levels: ["low", "mid", "high"]),
            ]
            func request(_ index: Int) -> DecisionRequest {
                index == 0
                    ? DecisionRequest(state: "A B", questions: questions)
                    : DecisionRequest(
                        state: nil, questions: questions, images: [pixels.userImage])
            }
            for index in 0 ..< 2 {
                let expected = try await model.decisionProbabilities(
                    request(index), tokenizer: CharTokenizer(), options: .init())
                let response = try await container.predict(request(index), options: .init())
                #expect(response.inputTokens == expected.inputTokens)
                #expect(
                    maxAbsDiff(response.answers.map(\.probabilities), expected.probabilities)
                        <= 1e-6)
            }
        }
    }

    /// Rows of each request, built the way `decisionProbabilities` builds them.
    private func requestRows(_ model: D1Omni, _ requests: [DecisionRequest]) async throws
        -> [(rows: [D1OmniRow], inputTokens: Int)]
    {
        let tokenizer = CharTokenizer()
        let tokens = try D1OmniPrompt.Tokens(tokenizer, fallbackBOS: 1)
        var result: [(rows: [D1OmniRow], inputTokens: Int)] = []
        for request in requests {
            let (prefix, media) = try await model.mediaPrefix(images: request.images)
            result.append(
                try model.rows(
                    state: request.state, questions: request.questions, prefix: prefix,
                    media: media, tokenizer: tokenizer, tokens: tokens))
        }
        return result
    }
}

// MARK: - Real weights

/// Parity with mlx-vlm on the released checkpoint: text and image requests in float32,
/// float16 and bfloat16. Needs `MLX_D1_OMNI_MODEL` and `MLX_D1_FIXTURES` (see README.md).
/// Python's token ids, keyed by Python's texts, stand in for the tokenizer.
@Suite(
    .serialized,
    .enabled(if: D1Fixtures.omniModel != nil && D1Fixtures.directory != nil))
struct D1OmniParityTests {

    /// Python tokenizer ids from a table. `D1OmniPrompt.encoded(_:limit:tokenizer:)` cuts a
    /// state longer than `4 * limit + 64` characters into pieces the table does not hold, so
    /// fixture states must stay shorter than that.
    private struct TableTokenizer: Tokenizer {
        var table: [String: [Int]]
        var special: [String: Int]

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            guard let ids = table[text] else {
                Issue.record("No token ids for \(text.debugDescription)")
                return []
            }
            return ids
        }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { special[token] }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { "<|startoftext|>" }
        var eosToken: String? { "<|im_end|>" }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// mlx-vlm media cases written by `tools/d1_fixtures.py`.
    private struct MediaFixture: Decodable {
        struct Case: Decodable {
            var tag: String
            var state: JSONValue?
            var questions: [D1Fixtures.Question]
            /// File names in the fixture directory.
            var images: [String]
            /// A file name in the fixture directory.
            var audio: String?
            /// The tensor that holds the audio samples.
            var audioSamples: String?
            var probabilities: [[Double]]
            var inputTokens: Int
            var prefix: String?
            /// Largest probability difference.
            var tolerance: Double
            /// Largest relative L2 distance of the media prefix.
            var prefixTolerance: Double

            enum CodingKeys: String, CodingKey {
                case tag, state, questions, images, audio, probabilities, prefix, tolerance
                case audioSamples = "audio_samples"
                case inputTokens = "input_tokens"
                case prefixTolerance = "prefix_tolerance"
            }

            func request() throws -> DecisionRequest {
                DecisionRequest(
                    state: D1Fixtures.state(state), questions: questions.map(\.named),
                    images: try images.map { .url(try D1Fixtures.url($0)) })
            }
        }

        var special: [String: Int]
        var tokens: [String: [Int]]
        var cases: [Case]

        /// `d1-omni-600M-<label>-media.json` and its tensors.
        static func load(_ label: String) throws -> (Self, tensors: [String: MLXArray]) {
            let stem = "d1-omni-600M-\(label)-media"
            let fixture = try JSONDecoder().decode(
                Self.self, from: Data(contentsOf: D1Fixtures.url("\(stem).json")))
            return (fixture, try loadArrays(url: D1Fixtures.url("\(stem).safetensors")))
        }

        var tokenizer: TableTokenizer { TableTokenizer(table: tokens, special: special) }
    }

    /// The checkpoint is loaded as `DecisionModelFactory` loads it.
    @Test(
        "Text and image requests match mlx-vlm",
        arguments: ["float32", "float16", "bfloat16"])
    func mediaCases(dtype: String) async throws {
        let (fixture, tensors) = try MediaFixture.load(dtype)
        let directory = try #require(D1Fixtures.omniModel)
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let model = try D1Omni(configurationData: data)
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        model.train(false)
        if dtype != "float32" {
            // Cast every floating-point parameter, as Python's `Module.set_dtype`.
            let type: DType = dtype == "float16" ? .float16 : .bfloat16
            let parameters = model.parameters().flattened().map {
                ($0.0, $0.1.dtype.isFloatingPoint ? $0.1.asType(type) : $0.1)
            }
            try model.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        }

        try #require(!fixture.cases.isEmpty)
        // Fixtures from the audio version of tools/d1_fixtures.py also hold audio cases.
        for item in fixture.cases where item.audio == nil && item.audioSamples == nil {
            let request = try item.request()
            let output = try await model.decisionProbabilities(
                request, tokenizer: fixture.tokenizer, options: .init())
            let difference = maxAbsDiff(output.probabilities, item.probabilities)
            #expect(output.inputTokens == item.inputTokens, "\(item.tag)")
            #expect(difference <= item.tolerance, "\(item.tag)")

            var prefixReport = ""
            if let key = item.prefix {
                let prefix = try #require(
                    try await model.mediaPrefix(images: request.images)
                        .prefix
                ).asType(.float32)
                let reference = try #require(tensors[key])
                #expect(prefix.shape == reference.shape, "\(item.tag)")
                let maxAbs = maxAbsDiff(prefix, reference)
                let relative =
                    (sqrt((prefix - reference).square().sum()) / sqrt(reference.square().sum()))
                    .item(Float.self)
                prefixReport = ", prefix \(prefix.shape) max abs \(maxAbs), relative L2 \(relative)"
                #expect(Double(relative) <= item.prefixTolerance, "\(item.tag)")
            }
            print(
                "D1Omni media parity \(dtype) \(item.tag): tokens \(output.inputTokens),"
                    + " probabilities \(difference)\(prefixReport)")
        }
    }
}
