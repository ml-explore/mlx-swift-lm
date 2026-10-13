// Copyright © 2026 Apple Inc.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXLMCommon
import MLXNN
import Testing
import UniformTypeIdentifiers

@testable import MLXDecisions
@testable import MLXVLM

// Ports of `TestD1DecisionModel` in mlx-vlm `tests/test_extraction_models.py`.
// MLX work runs on the CPU, where the tolerance checks hold.

private func tinyD1() throws -> D1 {
    let model = D1(
        LFM2VL(
            try JSONDecoder().decode(LFM2VLConfiguration.self, from: Data(tinyConfigJSON().utf8))))
    eval(model)
    return model
}

/// `<|img_row_R_col_C|>` for every `R * C <= 10`, at ids 260 to 286, above the ASCII ids.
private let tileMarkers: [String: Int] = {
    let names = (1 ... 10).flatMap { row in
        (1 ... 10 / row).map { column in "<|img_row_\(row)_col_\(column)|>" }
    }
    return Dictionary(uniqueKeysWithValues: names.enumerated().map { ($1, 260 + $0) })
}()

/// The tokenizer the tiny model reads: every id is below its vocabulary size of 320.
private let tinyTokenizer = VocabTokenizer(
    vocabulary: ["yes", "Yes", "YES", "no", "No", "NO", "0", "1", "2", "A", "B"],
    specialTokens: [
        "<|startoftext|>": 300, "<|im_start|>": 301, "<|im_end|>": 302, "<image>": 303,
        "<|image_start|>": 304, "<|image_end|>": 305, "<|img_thumbnail|>": 306,
    ].merging(tileMarkers) { first, _ in first },
    characterBase: 100, bos: "<|startoftext|>")

/// A 64x48 test picture (random in mlx-vlm's test, a fixed pattern here).
private func testPicture(width: Int = 64, height: Int = 48) -> RGB8Image {
    let bytes = (0 ..< width * height * 3).map { (i: Int) -> UInt8 in
        UInt8((i * 37 + (i / 7) * 11) % 256)
    }
    return RGB8Image(width: width, height: height, bytes: bytes)
}

/// Writes the picture to a temporary PNG file, or to a TIFF file tagged with an EXIF
/// orientation.
private func imageFile(_ picture: RGB8Image, orientation: Int? = nil) throws -> URL {
    let type = orientation == nil ? UTType.png : UTType.tiff
    let url = FileManager.default.temporaryDirectory.appending(
        component: "d1-\(UUID().uuidString).\(orientation == nil ? "png" : "tiff")")
    let provider = try #require(CGDataProvider(data: Data(picture.bytes) as CFData))
    let image = try #require(
        CGImage(
            width: picture.width, height: picture.height, bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: picture.width * 3, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    let destination = try #require(
        CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil))
    let properties = orientation.map { (value: Int) -> CFDictionary in
        [kCGImagePropertyOrientation: value] as CFDictionary
    }
    CGImageDestinationAddImage(destination, image, properties)
    try #require(CGImageDestinationFinalize(destination))
    return url
}

/// `cap_pixels` of the 2048x1024 pattern against Pillow's bicubic resize to 1448x724
/// (`tools/d1_fixtures.py vectors`).
private func expectCappedPattern(sourceLocation: SourceLocation = #_sourceLocation) {
    let capped = D1Prompt.capPixels(OrientedImage(RGB8Image.testPattern(width: 2048, height: 1024)))
    #expect(capped.width == 1448 && capped.height == 724, sourceLocation: sourceLocation)
    #expect(capped.byteSum == 400_988_986, sourceLocation: sourceLocation)
    #expect(
        Array(capped.bytes.prefix(6)) == [13, 72, 131, 62, 123, 189],
        sourceLocation: sourceLocation)
    #expect(Array(capped.bytes.suffix(3)) == [123, 119, 93], sourceLocation: sourceLocation)
}

/// Checks a Pillow bicubic resize of the pattern against Pillow's values. `resize` gets the
/// image and the output width and height.
private func expectPillowValues(
    _ resize: (RGB8Image, Int, Int) -> RGB8Image,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    // (input, output, byte sum, first 6 bytes, last 3 bytes) from
    // `tools/d1_fixtures.py vectors`. 3x500 takes the height-first path of `Image.resize`.
    let cases: [((Int, Int), (Int, Int), Int, [UInt8], [UInt8])] = [
        ((37, 53), (15, 20), 115_393, [59, 122, 164, 130, 162, 133], [181, 123, 67]),
        ((64, 48), (23, 17), 150_262, [72, 132, 159, 128, 156, 124], [203, 106, 76]),
        ((50, 40), (50, 13), 250_266, [19, 78, 137, 80, 145, 169], [115, 174, 233]),
        ((50, 40), (7, 40), 107_725, [106, 136, 141, 137, 129, 123], [124, 132, 142]),
        ((3, 500), (2, 37), 28_301, [113, 135, 136, 128, 137, 120], [118, 128, 136]),
        ((3, 500), (3, 41), 47_055, [108, 135, 147, 119, 141, 117], [110, 131, 140]),
        ((300, 2), (31, 3), 35_542, [116, 134, 136, 131, 123, 119], [125, 130, 123]),
    ]
    for ((width, height), (outWidth, outHeight), sum, first, last) in cases {
        let out = resize(RGB8Image.testPattern(width: width, height: height), outWidth, outHeight)
        let label = "\(width)x\(height) -> \(outWidth)x\(outHeight)"
        #expect(
            out.width == outWidth && out.height == outHeight, "\(label)",
            sourceLocation: sourceLocation)
        #expect(out.byteSum == sum, "\(label)", sourceLocation: sourceLocation)
        #expect(Array(out.bytes.prefix(6)) == first, "\(label)", sourceLocation: sourceLocation)
        #expect(Array(out.bytes.suffix(3)) == last, "\(label)", sourceLocation: sourceLocation)
    }
}

/// The positions a chunked prefill reports, one after each chunk.
private final class PrefillLog: @unchecked Sendable {
    private let lock = NSLock()
    private var reported = [Int]()

    var positions: [Int] { lock.withLock { reported } }

    func parameters(_ base: PrefillParameters = D1.trunkPrefill) -> PrefillParameters {
        var parameters = base
        parameters.progress = { (processed: Int, _: Int) -> Void in
            self.lock.withLock { self.reported.append(processed) }
        }
        return parameters
    }
}

/// The token run of one 64x48 picture: a single 320x224 view, 20x14 patches, 70 image tokens.
private let singleViewRun = [304] + Array(repeating: 303, count: 70) + [305]

private func encode(_ text: String) -> [Int] {
    tinyTokenizer.encode(text: text, addSpecialTokens: false)
}

/// The LFM2-VL processor's output for `images` with the model's settings, padded to
/// `maxNumPatches` rows as the chat path reads it.
private func paddedVision(_ model: D1, _ images: [RGB8Image]) throws -> Siglip2Patches {
    try LFM2VLImageProcessing(settings: model.imageSettings, tokenizer: tinyTokenizer)
        .process(images).patches
}

/// Log-probabilities after `trunk` through the chat path (`LFM2VL.prepare`).
private func chatLogProbabilities(_ model: D1, trunk: [Int], vision: Siglip2Patches) throws
    -> MLXArray
{
    let tokens = MLXArray(trunk.map { Int32($0) }).reshaped(1, trunk.count)
    let prepared = try model.base.prepare(
        LMInput(
            text: .init(tokens: tokens),
            image: .init(
                pixels: vision.pixels, frames: vision.spatialShapes.map { THW(1, $0.0, $0.1) })),
        cache: try model.base.newCache(parameters: nil), state: nil, prefill: PrefillParameters())
    guard case .logits(let output) = prepared else {
        throw VLMError.processing("prepare returned no logits")
    }
    let last = output.logits[0..., -1, 0...].asType(.float32)
    return last - logSumExp(last, axis: -1, keepDims: true)
}

/// mlx-vlm's `_D1_QUESTIONS`.
private let route = DecisionQuestion.choice("Route", options: ["A", "B"])
private let quality = DecisionQuestion.score("Rate", levels: ["low", "mid", "high"])
private let finished = DecisionQuestion.bool("Finished")

private func maxAbsDifference(_ a: [[Double]], _ b: [[Double]]) -> Double {
    precondition(a.map(\.count) == b.map(\.count))
    return zip(a.joined(), b.joined()).map { abs($0 - $1) }.max() ?? 0
}

private func container(_ model: D1, tokenizer: any Tokenizer) -> DecisionContainer {
    DecisionModelContainer(model: model, tokenizer: tokenizer).decisionContainer(modelID: "test/d1")
}

struct D1Tests {

    // MARK: Prompt

    @Test("Prefix and suffix text match the Python reference")
    func promptMatchesReference() throws {
        let tokenizer = VocabTokenizer(vocabulary: ["A", "B"])
        let head = "<|im_start|>user\n"
        let tail = "<|im_end|>\n<|im_start|>assistant\n"
        #expect(
            D1Prompt.prefixText(state: "Refund me", bos: "<s>")
                == "<s>\(head)Refund me\n\n\nQUESTION:\n")
        #expect(
            D1Prompt.prefixText(
                state: .json(["order": [1, "café"]]))
                == "\(head){\n  \"order\": [\n    1,\n    \"café\"\n  ]\n}\n\n\nQUESTION:\n")
        #expect(
            D1Prompt.prefixText(state: nil, bos: "<s>", images: "<image><image>")
                == "<s>\(head)<image><image>")

        let cases: [(DecisionQuestion, String)] = [
            (
                .choice(
                    "Q?",
                    options: [DecisionOption("billing", description: "Charges"), "tech_support"]),
                "Options:\nA Charges\nB tech support\n\nReply with the option code only."
            ),
            (
                .choice("Q?", options: ["x", "y"]),
                "Options:\nx x\ny y\n\nReply with the option code only."
            ),
            (
                .bool("Q?", whenTrue: "refund", whenFalse: "keep"),
                "Yes: refund\nNo: keep\n\nReply with yes or no only."
            ),
            (.bool("Q?"), "\nReply with yes or no only."),
            (
                .score("Q?", levels: ["low", "high"]),
                "0 low\n1 high\n\nReply with a single digit 0-1 only."
            ),
            // Checked against Python: a missing side prints `None`; an empty description is none.
            (
                .bool("Q?", whenTrue: "refund"),
                "Yes: refund\nNo: None\n\nReply with yes or no only."
            ),
            (.bool("Q?", whenFalse: "keep"), "Yes: None\nNo: keep\n\nReply with yes or no only."),
            (.bool("Q?", whenTrue: ""), "Yes: \nNo: None\n\nReply with yes or no only."),
            (
                .choice("Q?", options: [DecisionOption("a_b", description: ""), "c_d"]),
                "Options:\nA a b\nB c d\n\nReply with the option code only."
            ),
        ]
        for (question, body) in cases {
            let separator = question.kind == .bool ? "\n" : "\n\n"
            let aliases = try D1Prompt.aliases(for: question, tokenizer: tokenizer)
            #expect(
                D1Prompt.suffixText(question, aliases: aliases) == "Q?\(separator)\(body)\(tail)")
        }
    }

    @Test("Option codes and single-token aliases")
    func optionCodesAndAliases() throws {
        #expect(D1Prompt.optionCodes(["x", " Y "]) == ["x", "Y"])
        #expect(D1Prompt.optionCodes(["red", "green"]) == ["A", "B"])
        let numbered = D1Prompt.optionCodes((0 ..< 27).map(String.init))
        #expect([numbered[0], numbered[26]] == ["00", "26"])
        // Python `len(k) == 1 and k.isalpha()` counts code points and letter categories.
        let unicode: [([String], [String])] = [
            (["é", "ß"], ["é", "ß"]), (["e\u{301}", "x"], ["A", "B"]),
            ([" x\u{1C}", "y\u{3000}"], ["x", "y"]), (["ǅ", "ʰ"], ["ǅ", "ʰ"]),
            (["1", "2"], ["A", "B"]), (["x", " "], ["A", "B"]),
        ]
        for (labels, codes) in unicode {
            #expect(D1Prompt.optionCodes(labels) == codes, "\(labels)")
        }

        let letters = (UInt8(ascii: "A") ... UInt8(ascii: "Z")).map { String(UnicodeScalar($0)) }
        let tokenizer = VocabTokenizer(
            vocabulary: letters + (1 ... 26).map { String(format: "%02d", $0) })
        let codes = try D1Prompt.aliases((0 ..< 27).map { "label \($0)" }, tokenizer: tokenizer)
        // "00" is two tokens here, so it takes the first free code of the pool.
        #expect(
            Array(codes.prefix(3)) == [
                .init(code: "A", id: 0), .init(code: "01", id: 26), .init(code: "02", id: 27),
            ])
        #expect(Set(codes.map(\.id)).count == 27)

        // One-character codes are the only single tokens here: 52 of them.
        let many = (0 ..< 53).map { "label \($0)" }
        #expect(
            try D1Prompt.aliases(Array(many.prefix(52)), tokenizer: VocabTokenizer(vocabulary: []))
                .count == 52)
        #expect(throws: DecisionError.noSingleTokenAlias(optionCount: 53)) {
            try D1Prompt.aliases(many, tokenizer: VocabTokenizer(vocabulary: []))
        }
    }

    @Test("Readout max-pools each option's tokens")
    func readoutMaxPoolsEachOption() throws {
        let tokenizer = VocabTokenizer(
            vocabulary: ["A", "B", " A", "yes", "Yes", "no", "0", "1", "2"])
        let choice = DecisionQuestion.choice("q", options: ["red", "blue"])
        let aliases = try D1Prompt.aliases(for: choice, tokenizer: tokenizer)
        #expect(
            try D1Prompt.readoutIDs(choice, aliases: aliases, tokenizer: tokenizer) == [
                [0, 2], [1],
            ])
        #expect(
            try D1Prompt.readoutIDs(.bool("q"), aliases: [], tokenizer: tokenizer) == [
                [5], [3, 4],
            ])
        #expect(
            try D1Prompt.readoutIDs(
                .score("q", levels: ["a", "b", "c"]), aliases: [], tokenizer: tokenizer)
                == [[6], [7], [8]])

        let expected = [1 / (1 + exp(-1.0)), 1 / (1 + exp(1.0))]
        let pooledChoice = D1Prompt.readout(
            [[0, 2], [1]], logProbabilities: [0: -3, 2: -1, 1: -2])
        let pooledBool = D1Prompt.readout(
            [[3, 4], [5]], logProbabilities: [3: -4, 4: -0.5, 5: -1.5])
        #expect(maxAbsDifference([pooledChoice, pooledBool], [expected, expected]) < 1e-12)
    }

    @Test("Missing answer tokens throw, after every alias is found")
    func missingAnswerTokens() throws {
        let tokenizer = VocabTokenizer(vocabulary: [])
        #expect(
            throws: DecisionError.missingToken("The tokenizer has no single-token yes/no")
        ) {
            try D1Prompt.readoutIDs(.bool("q"), aliases: [], tokenizer: tokenizer)
        }
        // Python builds every question's prompt, and so its aliases, before any readout.
        let model = try tinyD1()
        let options = (0 ..< 53).map { DecisionOption("label \($0)") }
        #expect(throws: DecisionError.noSingleTokenAlias(optionCount: 53)) {
            try model.probabilities(
                state: "A B", questions: [.bool("q"), .choice("q", options: options)],
                tokenizer: tokenizer)
        }
        #expect(
            throws: DecisionError.missingToken("The tokenizer has no single-token yes/no")
        ) {
            try model.probabilities(
                state: "A B", questions: [route, .bool("q")], tokenizer: tokenizer)
        }
    }

    @Test("The decision API checks d1's question limits")
    func malformedQuestionsAreRejected() async throws {
        let model = try tinyD1()
        #expect(
            model.decisionCapabilities
                == DecisionCapabilities(
                    questionKinds: [.choice, .score, .bool], media: [.images],
                    maximumScoreLevels: 10))
        let decider = container(model, tokenizer: tinyTokenizer)
        await #expect(throws: DecisionError.tooManyScoreLevels(name: "q", count: 11, maximum: 10)) {
            try await decider.predict(
                state: "A B", questions: ["q": .score("Q", levels: (0 ..< 11).map(String.init))])
        }
        await #expect(throws: DecisionError.tooFewCriteria(name: "q", kind: .choice, minimum: 2)) {
            try await decider.predict(state: "A B", questions: ["q": .choice("Q", options: ["a"])])
        }
        let ten = try await decider.predict(
            state: "A B", questions: ["q": .score("Q", levels: (0 ..< 10).map(String.init))])
        #expect(ten["q"]?.probabilities.count == 10)
    }

    // MARK: Images

    @Test("cap_pixels scales images down to a megapixel")
    func capPixels() throws {
        // RGBA decodes to RGB; a small image keeps its size.
        let rgba = try RGB8Image(.array(MLXArray.zeros([50, 100, 4], type: UInt8.self)))
        #expect(rgba.width == 100 && rgba.height == 50 && rgba.bytes.count == 100 * 50 * 3)
        #expect(D1Prompt.capPixels(OrientedImage(rgba)) == rgba)

        Device.withDefaultDevice(.cpu) { expectCappedPattern() }

        // Python `cap_pixels` sizes: its test image, the d1-3B assets and the megapixel bound.
        let sizes: [((Int, Int), (Int, Int))] = [
            ((2048, 1024), (1448, 724)), ((640, 480), (640, 480)), ((1024, 1024), (1024, 1024)),
            ((1025, 1024), (1024, 1023)), ((1600, 1200), (1182, 886)), ((3024, 4032), (886, 1182)),
            ((720, 1600), (686, 1526)), ((1, 2_000_000), (1, 1_448_154)),
        ]
        for ((width, height), expected) in sizes {
            let size = D1Prompt.cappedSize(width: width, height: height)
            #expect(
                size.width == expected.0 && size.height == expected.1, "\(width)x\(height)")
        }
    }

    @Test("The Pillow bicubic resize matches Pillow")
    func pillowBicubic() {
        expectPillowValues { (image: RGB8Image, width: Int, height: Int) -> RGB8Image in
            image.resizedPILBicubic(width: width, height: height)
        }
        let picture = RGB8Image.testPattern(width: 37, height: 53)
        #expect(picture.resizedPILBicubic(width: 37, height: 53) == picture)
    }

    @Test("The Pillow resize passes follow Image.resize")
    func pillowPassOrder() {
        // (input, output, whether each pass is horizontal). Only a shrink of an image over
        // 100 times taller than wide goes height first.
        let cases: [((Int, Int), (Int, Int), [Bool])] = [
            ((37, 53), (15, 20), [true, false]), ((50, 40), (50, 13), [false]),
            ((50, 40), (7, 40), [true]), ((37, 53), (37, 53), []),
            ((3, 500), (2, 37), [false, true]), ((3, 500), (3, 41), [false]),
            ((3, 500), (2, 600), [true, false]), ((5, 500), (2, 37), [true, false]),
        ]
        for ((width, height), (outWidth, outHeight), horizontal) in cases {
            let image = RGB8Image(
                width: width, height: height,
                bytes: [UInt8](repeating: 0, count: width * height * 3))
            let passes = image.pillowBicubicPasses(width: outWidth, height: outHeight)
            let label = "\(width)x\(height) -> \(outWidth)x\(outHeight)"
            #expect(passes.map(\.horizontal) == horizontal, "\(label)")
            #expect(
                passes.map(\.taps.start.count)
                    == horizontal.map { (isHorizontal: Bool) -> Int in
                        isHorizontal ? outWidth : outHeight
                    }, "\(label)")
            // The upright size decides: 500x3 stored with orientation 6 is 3x500 upright.
            let turned = RGB8Image(width: height, height: width, bytes: image.bytes)
            #expect(
                turned.pillowBicubicPasses(width: outWidth, height: outHeight, orientation: 6)
                    == passes, "\(label)")
        }
    }

    @Test("Pillow's taps keep the resize sums inside int32")
    func pillowTapsFitInt32() {
        // Pillow and the Metal kernel sum in int32; the CPU port sums in Int.
        var sizes = [(Int, Int)]()
        for inSize in 1 ... 40 {
            for outSize in 1 ... 40 {
                sizes.append((inSize, outSize))
            }
        }
        sizes += [
            (8064, 1182), (6048, 886), (4032, 1182), (3024, 886), (1448, 8064), (5, 4000),
            (4000, 5), (65535, 1), (1, 4096), (2048, 1448), (1025, 1024),
        ]
        let bias = 1 << 21
        var (lowest, highest) = (0, 0)
        for (inSize, outSize) in sizes {
            let taps = ResampleTaps(pillowBicubicInSize: inSize, outSize: outSize)
            #expect(taps.shift == 22)
            for i in 0 ..< outSize {
                var (negative, positive) = (0, 0)
                for j in 0 ..< taps.count[i] {
                    let weight = Int(taps.weights[i * taps.kernelSize + j])
                    if weight < 0 {
                        negative += weight
                    } else {
                        positive += weight
                    }
                }
                lowest = min(lowest, bias + 255 * negative)
                highest = max(highest, bias + 255 * positive)
            }
        }
        #expect(lowest >= Int(Int32.min) && highest <= Int(Int32.max), "\(lowest) \(highest)")
    }

    @Test("A turned photo is capped as the upright image")
    func capPixelsReadsTheOrientation() throws {
        try Device.withDefaultDevice(.cpu) {
            let small = RGB8Image.testPattern(width: 260, height: 180)
            for orientation in 1 ... 8 {
                let upright = small.oriented(orientation)
                let capped = D1Prompt.capPixels(
                    OrientedImage(small, orientation: orientation), maxPixels: 20_000)
                let size = D1Prompt.cappedSize(
                    width: upright.width, height: upright.height, maxPixels: 20_000)
                #expect(capped.width == size.width && capped.height == size.height)
                #expect(capped.width < upright.width && capped.height < upright.height)
                let same = capped == D1Prompt.capPixels(OrientedImage(upright), maxPixels: 20_000)
                #expect(same, "orientation \(orientation)")
            }

            // A file over a megapixel keeps its orientation up to the cap.
            let stored = RGB8Image.testPattern(width: 1300, height: 900)
            let file = try imageFile(stored, orientation: 6)
            defer { try? FileManager.default.removeItem(at: file) }
            let upright = stored.oriented(6)
            try #require(try RGB8Image(.url(file)) == upright)
            let model = try tinyD1()
            func inputs(_ image: UserInput.Image) throws -> D1Inputs {
                try model.decisionInputs(
                    state: nil, questions: [finished], images: [image], tokenizer: tinyTokenizer)
            }
            let (turned, expected) = (try inputs(.url(file)), try inputs(upright.userImage))
            #expect(turned.trunk == expected.trunk)
            let (pixels, expectedPixels) = (
                try #require(turned.vision).pixels, try #require(expected.vision).pixels
            )
            #expect(arrayEqual(pixels, expectedPixels).item(Bool.self))
        }
    }

    @Test("Image decisions use the LFM2-VL processor")
    func imageDecisionsUseTheProcessor() throws {
        let picture = testPicture()
        let file = try imageFile(picture)
        defer { try? FileManager.default.removeItem(at: file) }
        try #require(try RGB8Image(.url(file)) == picture)
        // mlx-vlm's test passes a PIL image and a file path.
        let images = [picture.userImage, UserInput.Image.url(file)]

        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            let tokenizer = tinyTokenizer
            let questions = [route, quality, finished]

            let inputs = try model.decisionInputs(
                state: nil, questions: questions, images: images, tokenizer: tokenizer)
            #expect(inputs.trunk == [300, 301] + encode("user\n") + singleViewRun + singleViewRun)
            #expect(inputs.trunk.filter { $0 == 304 }.count == 2)
            #expect(
                inputs.branches
                    == zip(questions, inputs.aliases).map {
                        encode(D1Prompt.suffixText($0, aliases: $1))
                    })
            let vision = try #require(inputs.vision)
            #expect(vision.pixels.shape == [2, 280, 768])
            #expect(vision.spatialShapes.map { [$0.0, $0.1] } == [[14, 20], [14, 20]])

            let (together, read) = try model.probabilities(
                state: nil, questions: questions, images: images, tokenizer: tokenizer)
            #expect(read == inputs.inputTokens)
            #expect(together.map(\.count) == [2, 3, 2])
            let (chunked, _) = try model.probabilities(
                state: nil, questions: questions, images: images, tokenizer: tokenizer,
                tokenBudget: 1)
            #expect(maxAbsDifference(chunked, together) < 1e-5)
            for (index, question) in questions.enumerated() {
                let (alone, aloneRead) = try model.probabilities(
                    state: nil, questions: [question], images: images, tokenizer: tokenizer)
                #expect(aloneRead == inputs.trunk.count + inputs.branches[index].count)
                #expect(maxAbsDifference(alone, [together[index]]) < 1e-5)
            }
        }
    }

    @Test("Image answers ignore the padded patch rows and match the chat path")
    func paddedPatchRowsAreIgnored() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            let picture = testPicture()
            let inputs = try model.decisionInputs(
                state: "A B", questions: [finished], images: [picture.userImage],
                tokenizer: tinyTokenizer)
            let vision = try #require(inputs.vision)
            let trimmed = try model.answerLogProbabilities(
                trunk: inputs.trunk, branches: [], vision: vision)

            let padded = try paddedVision(model, [picture])
            #expect(padded.pixels.dim(1) == 1024)
            let untrimmed = try model.answerLogProbabilities(
                trunk: inputs.trunk, branches: [], vision: padded)
            #expect(allClose(trimmed, untrimmed, atol: 1e-5).item(Bool.self))

            let chat = try chatLogProbabilities(model, trunk: inputs.trunk, vision: padded)
            #expect(allClose(trimmed, chat, atol: 1e-5).item(Bool.self))
        }
    }

    @Test("Views of different sizes keep the patch rows of the longest view")
    func viewsOfDifferentSizes() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            // Python: 64x48 is one 14x20 view (280 patches), 48x96 one 24x12 view (288).
            let pictures = [testPicture(), testPicture(width: 48, height: 96)]
            let inputs = try model.decisionInputs(
                state: "A B", questions: [finished], images: pictures.map(\.userImage),
                tokenizer: tinyTokenizer)
            let vision = try #require(inputs.vision)
            try #require(vision.pixels.shape == [2, 288, 768])
            #expect(vision.spatialShapes.map { [$0.0, $0.1] } == [[14, 20], [24, 12]])
            let secondRun: [Int] = [304] + Array(repeating: 303, count: 72) + [305]
            #expect(
                inputs.trunk
                    == [300, 301] + encode("user\n") + singleViewRun + secondRun
                    + encode("A B\n\n\nQUESTION:\n" + D1Prompt.suffixText(finished, aliases: [])))

            let decision = try model.answerLogProbabilities(
                trunk: inputs.trunk, branches: [], vision: vision)
            let chat = try chatLogProbabilities(
                model, trunk: inputs.trunk, vision: try paddedVision(model, pictures))
            #expect(allClose(decision, chat, atol: 1e-5).item(Bool.self))
        }
    }

    @Test("Images over a megapixel are capped before the LFM2-VL processor")
    func largeImagesAreCapped() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            let inputs = try model.decisionInputs(
                state: nil, questions: [finished],
                images: [testPicture(width: 1100, height: 1100).userImage],
                tokenizer: tinyTokenizer)

            // Python caps 1100x1100 to 1024x1024, which is 2x2 tiles and a thumbnail. Without
            // the cap it is 3x3 tiles and a thumbnail.
            let vision = try #require(inputs.vision)
            #expect(
                vision.spatialShapes.map { [$0.0, $0.1] } == Array(repeating: [32, 32], count: 5))
            func view(_ marker: String) -> [Int] {
                [tinyTokenizer.specialTokens[marker]!] + Array(repeating: 303, count: 256)
            }
            let run: [Int] =
                [304] + view("<|img_row_1_col_1|>") + view("<|img_row_1_col_2|>")
                + view("<|img_row_2_col_1|>") + view("<|img_row_2_col_2|>")
                + view("<|img_thumbnail|>") + [305]
            #expect(
                inputs.trunk
                    == [300, 301] + encode("user\n") + run
                    + encode(D1Prompt.suffixText(finished, aliases: [])))

            // Every view is full, so the chat path reads the same pixels.
            let decision = try model.answerLogProbabilities(
                trunk: inputs.trunk, branches: [], vision: vision)
            let chat = try chatLogProbabilities(model, trunk: inputs.trunk, vision: vision)
            #expect(allClose(decision, chat, atol: 1e-5).item(Bool.self))
        }
    }

    @Test("Decisions lay out images with the model's image settings")
    func decisionsUseTheModelImageSettings() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            model.imageSettings.minImageTokens = 32
            let picture = testPicture()
            let inputs = try model.decisionInputs(
                state: nil, questions: [finished], images: [picture.userImage],
                tokenizer: tinyTokenizer)

            // Python with `min_image_tokens` 32: one 10x14 view and 35 image tokens, not 14x20.
            let vision = try #require(inputs.vision)
            #expect(vision.spatialShapes.map { [$0.0, $0.1] } == [[10, 14]])
            #expect(vision.pixels.shape == [1, 140, 768])
            let run: [Int] = [304] + Array(repeating: 303, count: 35) + [305]
            #expect(
                inputs.trunk
                    == [300, 301] + encode("user\n") + run
                    + encode(D1Prompt.suffixText(finished, aliases: [])))

            let decision = try model.answerLogProbabilities(
                trunk: inputs.trunk, branches: [], vision: vision)
            let chat = try chatLogProbabilities(
                model, trunk: inputs.trunk, vision: try paddedVision(model, [picture]))
            #expect(allClose(decision, chat, atol: 1e-5).item(Bool.self))
        }
    }

    @Test("Requests with only images, or images then a state")
    func imagesOnlyRequests() async throws {
        let picture = testPicture()
        try await Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            let decider = container(model, tokenizer: tinyTokenizer)
            let response = try await decider.predict(
                state: nil, questions: ["route": route, "quality": quality, "finished": finished],
                images: [picture.userImage])
            #expect(response.answers.map(\.name) == ["route", "quality", "finished"])
            let (raw, read) = try model.probabilities(
                state: nil, questions: [route, quality, finished], images: [picture.userImage],
                tokenizer: tinyTokenizer)
            #expect(response.inputTokens == read)
            #expect(response["route"]?.probabilities == raw[0])
            #expect(response["finished"]?.probabilities == raw[2])

            let single = try await decider.predict(
                state: nil, questions: ["finished": finished], images: [picture.userImage])
            let (rawSingle, readSingle) = try model.probabilities(
                state: nil, questions: [finished], images: [picture.userImage],
                tokenizer: tinyTokenizer)
            #expect(single["finished"]?.probabilities == rawSingle[0])
            #expect(single.inputTokens == readSingle)

            let suffix = D1Prompt.suffixText(finished, aliases: [])
            let withState = try model.decisionInputs(
                state: "A B", questions: [finished], images: [picture.userImage],
                tokenizer: tinyTokenizer)
            #expect(
                withState.trunk
                    == [300, 301] + tinyTokenizer.encode(text: "user\n", addSpecialTokens: false)
                    + singleViewRun
                    + tinyTokenizer.encode(
                        text: "A B\n\n\nQUESTION:\n" + suffix, addSpecialTokens: false))
        }
    }

    @Test("Images that cannot be read throw invalidImage")
    func unreadableImagesThrow() async throws {
        let model = try tinyD1()
        let image = UserInput.Image.url(URL(fileURLWithPath: "/nonexistent/cat.png"))
        let error = await #expect(throws: DecisionError.self) {
            try await model.decisionProbabilities(
                DecisionRequest(questions: ["route": route], images: [image]),
                tokenizer: tinyTokenizer, options: .init())
        }
        guard case .invalidImage = error else {
            Issue.record("Expected invalidImage, got \(String(describing: error))")
            return
        }
    }

    @Test("Image placeholders in the prompt and missing image tokens throw decision errors")
    func imageProcessingErrorsAreDecisionErrors() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let image = testPicture().userImage
            let model = try tinyD1()
            let decider = container(model, tokenizer: tinyTokenizer)
            // Python's processor raises a ValueError when the trunk has more `<image>` than images.
            let mismatch = DecisionError.imagePlaceholderInText("<image>")
            await #expect(throws: mismatch) {
                try await decider.predict(
                    state: "HTML has <image> tags", questions: ["done": finished], images: [image])
            }
            await #expect(throws: mismatch) {
                try await decider.predict(
                    state: nil, questions: ["done": .bool("Is <image> a tag?")], images: [image])
            }
            // A branch, or a request without images, reads the text as a plain token.
            _ = try await decider.predict(
                state: nil, questions: ["done": .bool("Is <image> a tag?"), "route": route],
                images: [image])
            _ = try await decider.predict(
                state: "HTML has <image> tags", questions: ["done": finished])

            var noMarkers = tinyTokenizer
            noMarkers.specialTokens["<|image_start|>"] = nil
            await #expect(
                throws: DecisionError.missingToken(
                    "The tokenizer has no <|image_start|> token")
            ) {
                try await container(model, tokenizer: noMarkers).predict(
                    state: "A B", questions: ["done": finished], images: [image])
            }
        }
    }

    // MARK: Batching

    @Test("Branch chunks match mlx-vlm#2500's _chunks")
    func chunksMatchPython() {
        // A 981-token choice at index 30 among 63 bool questions of 20 tokens.
        let bools = Array(repeating: 20, count: 30) + [981] + Array(repeating: 20, count: 33)
        let cases: [([Int], Int, Int, [[Int]])] = [
            ([20, 30, 35], 15, 65536, [[0, 1, 2]]),
            ([20, 30, 35], 15, 1, [[2], [1], [0]]),
            ([20, 30, 35], 15, 90, [[2], [0, 1]]),
            ([20, 30, 35], 15, 150, [[0, 1, 2]]),
            ([5, 40, 5, 5], 10, 100, [[1], [0, 2, 3]]),
            ([10, 12, 11, 12], 4, 32, [[1, 3], [0, 2]]),
            ([10, 5, 4], 0, 1000, [[0, 1], [2]]),
            (bools, 199, 16384, [[30], Array(0 ..< 30) + Array(31 ..< 64)]),
            (
                bools, 199, 2000,
                [
                    [30], Array(0 ..< 9), Array(9 ..< 18), Array(18 ..< 27),
                    [27, 28, 29] + Array(31 ..< 37), Array(37 ..< 46), Array(46 ..< 55),
                    Array(55 ..< 64),
                ]
            ),
            ([], 3, 10, []),
        ]
        for (lengths, trunk, budget, expected) in cases {
            #expect(D1Prompt.chunks(lengths: lengths, trunk: trunk, budget: budget) == expected)
        }
    }

    @Test("Tiled caches copy the trunk state and keep their types")
    func tileKeepsCacheTypes() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            let cache = try model.base.newCache(parameters: nil)
            _ = model.base.hiddenStates(
                tokens: MLXArray((0 ..< 7).map { Int32($0) }).reshaped(1, 7), embeddings: nil,
                cache: cache)
            eval(cache)
            let tiled = D1.tile(cache, count: 3)
            let conv = try #require(tiled[0] as? MambaCache)
            let attention = try #require(tiled[1] as? KVCacheSimple)
            #expect(conv[0]?.shape == [3, 2, 16])
            #expect(attention.offset == 7)
            #expect(attention.state.map(\.shape) == [[3, 1, 7, 8], [3, 1, 7, 8]])
            #expect(allClose(attention.state[0][2 ..< 3], cache[1].state[0]).item(Bool.self))
            #expect(allClose(conv[0]![1 ..< 2], (cache[0] as! MambaCache)[0]!).item(Bool.self))
        }
    }

    @Test("Questions continue the shared state as single passes do")
    func questionsContinueTheSharedState() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            let tokenizer = tinyTokenizer
            let questions = [route, quality, finished]
            let state = DecisionState.json(["ticket": "A B"])
            func suffix(_ question: DecisionQuestion) throws -> String {
                D1Prompt.suffixText(
                    question, aliases: try D1Prompt.aliases(for: question, tokenizer: tokenizer))
            }

            let prefix = D1Prompt.prefixText(state: state, bos: "<|startoftext|>")
            let inputs = try model.decisionInputs(
                state: state, questions: questions, tokenizer: tokenizer)
            #expect(inputs.trunk == encode(prefix))
            #expect(inputs.trunk.prefix(2) == [300, 301])
            let branches = try questions.map { encode(try suffix($0)) }
            #expect(inputs.branches == branches)
            #expect(inputs.vision == nil)

            let (together, read) = try model.probabilities(
                state: state, questions: questions, tokenizer: tokenizer)
            #expect(read == inputs.trunk.count + inputs.branches.reduce(0) { $0 + $1.count })
            #expect(together.map(\.count) == [2, 3, 2])
            let (chunked, chunkedRead) = try model.probabilities(
                state: state, questions: questions, tokenizer: tokenizer, tokenBudget: 1)
            #expect(chunkedRead == read)
            #expect(maxAbsDifference(chunked, together) < 1e-5)

            for (question, probabilities) in zip(questions, together) {
                let (alone, aloneRead) = try model.probabilities(
                    state: state, questions: [question], tokenizer: tokenizer)
                let single = encode(prefix + (try suffix(question)))
                #expect(aloneRead == single.count)
                #expect(maxAbsDifference(alone, [probabilities]) < 1e-5)
            }
        }
    }

    @Test("Answers keep request order when a long question runs first")
    func answersKeepRequestOrder() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            let tokenizer = tinyTokenizer
            let words = String(repeating: "long words ", count: 10)
            let long = DecisionQuestion.choice(
                "Route", options: [.init("A", description: words), .init("B", description: words)])
            let questions = [finished, long, route, quality]
            let state = DecisionState.text("A B")
            let inputs = try model.decisionInputs(
                state: state, questions: questions, tokenizer: tokenizer)
            let chunks = D1Prompt.chunks(
                lengths: inputs.branches.map(\.count), trunk: inputs.trunk.count, budget: 16_384)
            try #require(chunks == [[1], [0, 2, 3]])

            let (together, _) = try model.probabilities(
                state: state, questions: questions, tokenizer: tokenizer)
            #expect(together.map(\.count) == [2, 2, 2, 3])
            let (reversed, _) = try model.probabilities(
                state: state, questions: questions.reversed(), tokenizer: tokenizer)
            #expect(maxAbsDifference(reversed.reversed(), together) < 1e-5)
            for (question, probabilities) in zip(questions, together) {
                let (alone, _) = try model.probabilities(
                    state: state, questions: [question], tokenizer: tokenizer)
                #expect(maxAbsDifference(alone, [probabilities]) < 1e-5)
            }
        }
    }

    // MARK: Long trunks

    /// Checks a chunked prefill's reports: chunks of at most `step` tokens up to the last
    /// position of a `count`-token trunk, in as few chunks as that takes.
    private func expectChunks(
        _ positions: [Int], count: Int, step: Int,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(positions.count == (count - 1 + step - 1) / step, sourceLocation: sourceLocation)
        #expect(positions.last == count - 1, sourceLocation: sourceLocation)
        let sizes = zip([0] + positions, positions).map { (start: Int, end: Int) -> Int in
            end - start
        }
        #expect(
            sizes.allSatisfy { (size: Int) -> Bool in size >= 1 && size <= step },
            "\(positions)", sourceLocation: sourceLocation)
    }

    @Test("A long trunk is prefilled in chunks and answers as a single pass does")
    func longTrunksArePrefilledInChunks() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            // One question and several, with image features in the trunk.
            for questions in [[finished], [route, quality, finished]] {
                let inputs = try model.decisionInputs(
                    state: "A B", questions: questions, images: [testPicture().userImage],
                    tokenizer: tinyTokenizer)
                let count = inputs.trunk.count
                try #require(count > 90)
                func answer(limit: Int, _ prefill: PrefillParameters) throws -> MLXArray {
                    try model.answerLogProbabilities(
                        trunk: inputs.trunk, branches: inputs.branches, vision: inputs.vision,
                        singlePassLimit: limit, prefill: prefill)
                }
                let single = try model.answerLogProbabilities(
                    trunk: inputs.trunk, branches: inputs.branches, vision: inputs.vision)
                #expect(single.shape == [questions.count, 320])

                for step in [7, 32, 512] {
                    let log = PrefillLog()
                    let chunked = try answer(
                        limit: 16, log.parameters(PrefillParameters(stepSize: step)))
                    expectChunks(log.positions, count: count, step: step)
                    let difference = abs(chunked - single).max().item(Float.self)
                    #expect(difference < 2e-5, "\(questions.count) questions, step \(step)")
                }

                // The limit itself is still one pass, with the same bits.
                let log = PrefillLog()
                let atLimit = try answer(
                    limit: count, log.parameters(PrefillParameters(stepSize: 7)))
                #expect(log.positions.isEmpty)
                #expect(arrayEqual(atLimit, single).item(Bool.self))
                _ = try answer(limit: count - 1, log.parameters(PrefillParameters(stepSize: 7)))
                expectChunks(log.positions, count: count, step: 7)
            }
        }
    }

    @Test("Trunks of up to 2,048 tokens keep the single pass")
    func shortTrunksKeepTheSinglePass() throws {
        #expect(D1.singlePassLimit == 2048)
        let step = try #require(D1.trunkPrefill.stepSize)
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            for questions in [[finished], [route, quality]] {
                func inputs(_ state: String) throws -> D1Inputs {
                    try model.decisionInputs(
                        state: .text(state), questions: questions, tokenizer: tinyTokenizer)
                }
                // Every character of the state is one token.
                let overhead = try inputs("A").trunk.count - 1
                func answer(
                    tokens: Int, limit: Int = D1.singlePassLimit
                ) throws -> (logz: MLXArray, chunks: [Int]) {
                    let state = String(repeating: "A B, ", count: 500).prefix(tokens - overhead)
                    let inputs = try inputs(String(state))
                    try #require(inputs.trunk.count == tokens)
                    let log = PrefillLog()
                    let logz = try model.answerLogProbabilities(
                        trunk: inputs.trunk, branches: inputs.branches, singlePassLimit: limit,
                        prefill: log.parameters())
                    return (logz, log.positions)
                }

                let atLimit = try answer(tokens: 2048)
                #expect(atLimit.chunks.isEmpty)
                let forced = try answer(tokens: 2048, limit: .max)
                #expect(arrayEqual(atLimit.logz, forced.logz).item(Bool.self))

                let chunked = try answer(tokens: 2049)
                expectChunks(chunked.chunks, count: 2049, step: step)
                let single = try answer(tokens: 2049, limit: .max)
                #expect(single.chunks.isEmpty)
                let difference = abs(chunked.logz - single.logz).max().item(Float.self)
                #expect(difference < 2e-5, "\(questions.count) questions")
            }
        }
    }

    @Test("A lower token budget bounds the trunk's forward passes")
    func tokenBudgetBoundsTheTrunk() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            for questions in [[finished], [route, quality, finished]] {
                let inputs = try model.decisionInputs(
                    state: "A B", questions: questions, images: [testPicture().userImage],
                    tokenizer: tinyTokenizer)
                let count = inputs.trunk.count
                try #require(count > 90)
                let single = try model.answerLogProbabilities(
                    trunk: inputs.trunk, branches: inputs.branches, vision: inputs.vision)
                func answer(
                    budget: Int, _ prefill: PrefillParameters = D1.trunkPrefill
                ) throws -> (logz: MLXArray, chunks: [Int]) {
                    let log = PrefillLog()
                    let logz = try model.answerLogProbabilities(
                        trunk: inputs.trunk, branches: inputs.branches, vision: inputs.vision,
                        tokenBudget: budget, prefill: log.parameters(prefill))
                    return (logz, log.positions)
                }

                // A budget that holds the trunk keeps the single pass.
                #expect(try answer(budget: count).chunks.isEmpty)
                // Under it, chunks are at most the budget, or the prefill step if smaller.
                let cases: [(budget: Int, step: Int?, chunk: Int)] = [
                    (count - 1, nil, count - 1), (40, nil, 40), (40, 7, 7),
                ]
                for (budget, step, chunk) in cases {
                    let prefill = step.map { (size: Int) -> PrefillParameters in
                        PrefillParameters(stepSize: size)
                    }
                    let chunked = try answer(budget: budget, prefill ?? D1.trunkPrefill)
                    expectChunks(chunked.chunks, count: count, step: chunk)
                    let difference = abs(chunked.logz - single).max().item(Float.self)
                    #expect(
                        difference < 2e-5, "\(questions.count) questions, budget \(budget)")
                }
            }
        }
    }

    @Test("A single question reads the chat model's next-token log-probabilities")
    func singlePassMatchesChatLogits() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try tinyD1()
            let ids = tinyTokenizer.encode(
                text: D1Prompt.prefixText(state: "A B", bos: "<|startoftext|>")
                    + D1Prompt.suffixText(finished, aliases: []),
                addSpecialTokens: false)
            let logits = model.base(
                MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count), cache: nil)
            let last = logits[0..., -1, 0...].asType(.float32)
            let chat = last - logSumExp(last, axis: -1, keepDims: true)
            let decision = try model.answerLogProbabilities(trunk: ids, branches: [])
            #expect(decision.shape == [1, 320])
            #expect(allClose(decision, chat, atol: 1e-6).item(Bool.self))
        }
    }

    // MARK: Loading

    @Test("lfm2_vl checkpoints with d1's auto_map load as plain LFM2VL")
    func lfm2VLCheckpointsWithD1AutoMapLoadAsLFM2VL() async throws {
        let cases: [(extra: String, modelType: String)] = [
            (#""auto_map": {"AutoModel": "modeling_d1.D1Model"},"#, "lfm2_vl"),
            (#""auto_map": {"AutoModel": "modeling_d1.D1Model"},"#, "lfm2-vl"),
            (#""auto_map": {"AutoModel": "LiquidAI/d1-3B--modeling_d1.D1Model"},"#, "lfm2_vl"),
            ("", "lfm2_vl"),
            (#""auto_map": {"AutoModel": "modeling_lfm2_vl.Lfm2VlModel"},"#, "lfm2_vl"),
        ]
        for (extra, modelType) in cases {
            let model = try await VLMTypeRegistry.shared.createModel(
                configuration: Data(tinyConfigJSON(extra).utf8), modelType: modelType)
            #expect(type(of: model) == LFM2VL.self, "\(extra) \(modelType)")
        }
    }

    /// A tiny d1 checkpoint whose processor settings differ from the defaults.
    private func tinyCheckpoint() throws -> (directory: URL, settings: LFM2VLImageSettings) {
        let directory = FileManager.default.temporaryDirectory.appending(
            component: "d1-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(tinyConfigJSON(#""auto_map": {"AutoModel": "modeling_d1.D1Model"},"#).utf8)
            .write(to: directory.appending(component: "config.json"))
        try save(
            arrays: Dictionary(uniqueKeysWithValues: try tinyD1().base.parameters().flattened()),
            url: directory.appending(component: "model.safetensors"))
        try Data(
            #"""
            {"image_processor": {"max_image_tokens": 128, "min_image_tokens": 32, "resample": 3,
                                 "max_pixels_tolerance": 1.5},
             "processor_class": "Lfm2VlProcessor"}
            """#.utf8
        ).write(to: directory.appending(component: "processor_config.json"))
        var settings = LFM2VLImageSettings()
        settings.maxImageTokens = 128
        settings.minImageTokens = 32
        settings.resample = .bicubic
        settings.maxPixelsTolerance = 1.5
        return (directory, settings)
    }

    private struct Loader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { tinyTokenizer }
    }

    @Test("D1 gets the checkpoint's processor settings, also with a message generator")
    func loadD1PassesProcessorSettings() async throws {
        let (directory, expected) = try tinyCheckpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(try tinyD1().imageSettings == LFM2VLImageSettings())
        for generator in [nil, NoSystemMessageGenerator()] as [(any MessageGenerator)?] {
            var resolved = ResolvedModelConfiguration(directory: directory)
            resolved.messageGenerator = generator
            let (model, _) = try await DecisionModelFactory.shared.loadD1(
                resolved: resolved, modelID: "test/d1", tokenizerLoader: Loader())
            #expect(model.imageSettings == expected)
            #expect(type(of: model.base) == LFM2VL.self)
        }
    }
}

// MARK: - GPU resize

#if canImport(Metal)
/// The Pillow resize in the Metal kernel against the CPU port, which `D1Tests` checks against
/// Pillow.
@Suite(.enabled(if: Device.defaultDevice().deviceType == .gpu, "needs a GPU"))
struct D1PillowGPUResizeTests {
    /// Random bytes, a fifth of them 0 or 255, so bicubic overshoots clamp at both ends.
    private func randomImage(
        width: Int, height: Int, using rng: inout SeededGenerator
    ) -> RGB8Image {
        let bytes = (0 ..< width * height * 3).map { (_: Int) -> UInt8 in
            switch Int.random(in: 0 ..< 10, using: &rng) {
            case 0: 0
            case 1: 255
            default: UInt8.random(in: 0 ... 255, using: &rng)
            }
        }
        return RGB8Image(width: width, height: height, bytes: bytes)
    }

    /// Compares the GPU and CPU resize of `image` for every orientation. Each target is
    /// `size(uprightWidth, uprightHeight)`.
    private func expectSameBytes(
        _ image: RGB8Image, _ size: (Int, Int) -> (Int, Int),
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        for orientation in 1 ... 8 {
            let upright =
                orientation >= 5 ? (image.height, image.width) : (image.width, image.height)
            let (width, height) = size(upright.0, upright.1)
            let passes = image.pillowBicubicPasses(
                width: width, height: height, orientation: orientation)
            let gpu = image.gpuResampled(passes, orientation: orientation)
            let same = gpu == image.cpuResampled(passes, orientation: orientation)
            #expect(
                same,
                "\(image.width)x\(image.height) orientation \(orientation) -> \(width)x\(height)",
                sourceLocation: sourceLocation)
            #expect(
                gpu.width == width && gpu.height == height, sourceLocation: sourceLocation)
        }
    }

    @Test func gpuMatchesPillow() {
        expectPillowValues { (image: RGB8Image, width: Int, height: Int) -> RGB8Image in
            image.gpuResampled(image.pillowBicubicPasses(width: width, height: height))
        }
    }

    @Test func gpuMatchesCPUOnRandomSizes() {
        var rng = SeededGenerator(seed: 1)
        for _ in 0 ..< 30 {
            let image = randomImage(
                width: Int.random(in: 1 ... 257, using: &rng),
                height: Int.random(in: 1 ... 257, using: &rng), using: &rng)
            let target = (
                Int.random(in: 1 ... 257, using: &rng), Int.random(in: 1 ... 257, using: &rng)
            )
            expectSameBytes(image) { (_: Int, _: Int) -> (Int, Int) in target }
        }
    }

    @Test func gpuMatchesCPUOnExtremeRatios() {
        var rng = SeededGenerator(seed: 2)
        // Upright images over 100 times taller than wide shrink height first.
        let cases: [(Int, Int, Int, Int)] = [
            (1, 1, 300, 7), (2, 2, 1, 1), (2000, 3, 9, 1), (3, 2000, 1, 11), (1000, 1, 1, 1000),
            (3, 500, 2, 37), (4, 900, 9, 450), (900, 700, 23, 31), (13, 17, 600, 450),
        ]
        for (width, height, newWidth, newHeight) in cases {
            let image = randomImage(width: width, height: height, using: &rng)
            expectSameBytes(image) { (_: Int, _: Int) -> (Int, Int) in (newWidth, newHeight) }
        }
    }

    @Test func gpuSkipsPassesLikeCPU() {
        var rng = SeededGenerator(seed: 3)
        let image = randomImage(width: 61, height: 37, using: &rng)
        expectSameBytes(image) { (width: Int, _: Int) -> (Int, Int) in (width, 90) }
        expectSameBytes(image) { (_: Int, height: Int) -> (Int, Int) in (20, height) }
        expectSameBytes(image) { (width: Int, height: Int) -> (Int, Int) in (width, height) }
    }

    @Test("Photos over a megapixel are capped on the GPU")
    func photosAreCappedOnTheGPU() {
        let pattern = RGB8Image.testPattern(width: 2048, height: 1024)
        #expect(
            pattern.prefersGPU(
                pattern.pillowBicubicPasses(width: 1448, height: 724), orientation: 1))
        expectCappedPattern()

        let stored = RGB8Image.testPattern(width: 1300, height: 900)
        for orientation in [1, 6, 8] {
            let image = OrientedImage(stored, orientation: orientation)
            let size = D1Prompt.cappedSize(width: image.width, height: image.height)
            let passes = stored.pillowBicubicPasses(
                width: size.width, height: size.height, orientation: orientation)
            #expect(stored.prefersGPU(passes, orientation: orientation))
            let same =
                D1Prompt.capPixels(image) == stored.cpuResampled(passes, orientation: orientation)
            #expect(same, "orientation \(orientation)")
        }

        // An image under the cap is only turned, and a small resize stays on the CPU.
        #expect(!stored.prefersGPU([], orientation: 6))
        let small = RGB8Image.testPattern(width: 64, height: 48)
        #expect(!small.prefersGPU(small.pillowBicubicPasses(width: 23, height: 17), orientation: 1))
    }
}
#endif

// MARK: - Real weights

/// Parity with the PyTorch d1 reference (the checkpoint's `modeling_d1.py` with transformers'
/// `Lfm2VlProcessor`) on the released checkpoint in float32, on the text and image cases of
/// `d1-3B-float32.json`. Needs `MLX_D1_3B_MODEL` and `MLX_D1_FIXTURES` (see README.md).
/// Python's token ids, keyed by Python's prompt texts, stand in for the tokenizer, so a prompt
/// that differs from Python's by one character fails the id check.
@Suite(
    .serialized,
    .enabled(if: D1Fixtures.d1Model != nil && D1Fixtures.directory != nil))
struct D1ParityTests {

    private struct TableTokenizer: Tokenizer {
        var table: [String: [Int]]
        /// The added tokens of `tokenizer.json`, for the image markers.
        var addedTokens: [String: Int] = [:]

        /// Text missing from the table counts as two tokens, so it is never an alias.
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { table[text] ?? [-1, -1] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { addedTokens[token] }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { "<|startoftext|>" }
        var eosToken: String? { "<|im_end|>" }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// Python ids of the codes, digits and yes/no forms the readout encodes.
    private static let readoutTokens: [String: [Int]] = [
        "A": [41], "B": [42], "C": [43], " A": [334], " B": [378], " C": [340], "0": [24],
        "1": [25], "2": [26], "yes": [11683], "Yes": [12447], "YES": [65, 2020], "no": [2243],
        "No": [4547], "NO": [19598],
    ]

    /// One request of `d1-3B-float32.json` and the reference's results for it.
    private struct Case: Decodable {
        var tag: String
        /// File names in the fixture directory.
        var images: [String]
        var cappedSizes: [[Int]]
        var state: JSONValue?
        var questions: [D1Fixtures.Question]
        /// Python's prompt before image expansion: the whole prompt for one question, else
        /// the trunk.
        var promptText: String
        var suffixTexts: [String]
        var trunkIds: [Int]
        var branchIds: [[Int]]
        var visionShapes: [String: [Int]]
        var spatialShapes: [[Int]]
        var maskSums: [Int]
        /// Python's order, bool as `[yes, no]`.
        var probabilities: [[Double]]
        var inputTokens: Int
        var tolerance: Double

        enum CodingKeys: String, CodingKey {
            case tag, images, state, questions, probabilities, tolerance
            case cappedSizes = "capped_sizes"
            case promptText = "prompt_text"
            case suffixTexts = "suffix_texts"
            case trunkIds = "trunk_ids"
            case branchIds = "branch_ids"
            case visionShapes = "vision_shapes"
            case spatialShapes = "spatial_shapes"
            case maskSums = "mask_sums"
            case inputTokens = "input_tokens"
        }
    }

    /// Text and PNG cases match to 1e-5. Core Image decodes JPEG a few levels away from
    /// libjpeg-turbo, so the JPEG cases (`cats.jpg`) only agree to about 0.02.
    @Test("Text and image cases match the PyTorch reference")
    func cases() async throws {
        struct Fixture: Decodable { var cases: [Case] }
        let directory = try #require(D1Fixtures.d1Model)
        let configData = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = LFM2VL(
            try JSONDecoder.json5().decode(LFM2VLConfiguration.self, from: configData))
        // Load the vision tower now, so the float32 cast below reaches it.
        try await loadWeights(modelDirectory: directory, model: base, componentLoading: .immediate)
        let float32 = base.parameters().flattened().map { ($0.0, $0.1.asType(.float32)) }
        base.update(parameters: ModuleParameters.unflattened(float32))
        let settings = try #require(
            try LFM2VLImageSettings.resolvedConfiguration(directory: directory))
        let model = D1(
            base,
            imageSettings: try JSONDecoder.json5().decode(LFM2VLImageSettings.self, from: settings))
        model.train(false)
        eval(model)

        let cases = try JSONDecoder().decode(
            Fixture.self, from: Data(contentsOf: D1Fixtures.url("d1-3B-float32.json"))
        ).cases
        let tokenizerJSON = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appending(component: "tokenizer.json")))
                as? [String: Any])
        let addedTokens = try #require(tokenizerJSON["added_tokens"] as? [[String: Any]])
        let markers = Dictionary(
            addedTokens.compactMap { token -> (String, Int)? in
                guard let content = token["content"] as? String, let id = token["id"] as? Int
                else { return nil }
                return (content, id)
            }, uniquingKeysWith: { first, _ in first })
        let (imageStart, imageEnd, image) = (
            try #require(markers["<|image_start|>"]), try #require(markers["<|image_end|>"]),
            try #require(markers["<image>"])
        )

        try #require(cases.contains { $0.images.isEmpty } && cases.contains { !$0.images.isEmpty })
        for c in cases {
            let named = c.questions.map(\.named)
            let questions = named.map(\.question)
            let state = D1Fixtures.state(c.state)
            let images = try c.images.map { UserInput.Image.url(try D1Fixtures.url($0)) }

            let capped = try images.map { try RGB8Image($0) }.map {
                D1Prompt.cappedSize(width: $0.width, height: $0.height)
            }
            #expect(capped.map { [$0.width, $0.height] } == c.cappedSizes, "\(c.tag)")

            // Python's ids with each image run folded back into one `<image>`. A Swift prompt
            // that differs from Python's misses the table and has no `<image>` to expand.
            var folded = [Int]()
            var inRun = false
            for id in c.trunkIds {
                if id == imageStart {
                    inRun = true
                    folded.append(image)
                } else if id == imageEnd {
                    inRun = false
                } else if !inRun {
                    folded.append(id)
                }
            }
            var tokenizer = TableTokenizer(table: Self.readoutTokens, addedTokens: markers)
            tokenizer.table[c.promptText] = folded
            for (text, ids) in zip(c.suffixTexts, c.branchIds) {
                tokenizer.table[text] = ids
            }

            let inputs = try model.decisionInputs(
                state: state, questions: questions, images: images, tokenizer: tokenizer)
            try #require(inputs.trunk == c.trunkIds, "\(c.tag)")
            try #require(inputs.branches == c.branchIds, "\(c.tag)")
            #expect((inputs.vision == nil) == c.images.isEmpty, "\(c.tag)")
            if let vision = inputs.vision {
                try #require(vision.pixels.shape == c.visionShapes["pixel_values"], "\(c.tag)")
                #expect(vision.spatialShapes.map { [$0.0, $0.1] } == c.spatialShapes, "\(c.tag)")
                #expect(
                    Siglip2NaFlex.patchMask(vision.spatialShapes, length: vision.pixels.dim(1))
                        .sum(axis: 1).asArray(Int32.self).map(Int.init) == c.maskSums,
                    "\(c.tag)")
            }

            let output = try await model.decisionProbabilities(
                DecisionRequest(state: state, questions: named, images: images),
                tokenizer: tokenizer, options: .init())
            let expected = zip(questions, c.probabilities).map { question, p in
                question.kind == .bool ? Array(p.reversed()) : p
            }
            let difference = maxAbsDifference(output.probabilities, expected)
            print("d1-3B \(c.tag): max |dp| \(difference)")
            #expect(difference <= c.tolerance, "\(c.tag)")
            #expect(output.inputTokens == c.inputTokens, "\(c.tag)")
        }
    }
}
