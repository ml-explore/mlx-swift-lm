// Copyright © 2026 Apple Inc.
// Port of mlx-vlm's d1 model (Blaizzy/mlx-vlm#2467), which follows LiquidAI's reference code.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

// MARK: - Prompt

/// The d1 System One prompt and answer-slot readout (`d1/prompt.py`): no system turn,
/// JSON states, option descriptions, no calibration.
enum D1Prompt {
    static let imStart = "<|im_start|>"
    static let imEnd = "<|im_end|>"
    static let yesForms = ["yes", "Yes", "YES"]
    static let noForms = ["no", "No", "NO"]
    /// The most pixels an image keeps before the LFM2-VL processor sees it.
    static let visionMaxPixels = 1024 * 1024

    /// An option's code and its single token.
    struct Alias: Equatable, Sendable {
        var code: String
        var id: Int
    }

    /// Codes tried in order when an option's own code is not a free single token.
    static let fallbackPool: [String] = {
        let upper = (UInt8(ascii: "A") ... UInt8(ascii: "Z")).map { String(UnicodeScalar($0)) }
        let lower = (UInt8(ascii: "a") ... UInt8(ascii: "z")).map { String(UnicodeScalar($0)) }
        return upper + (0 ..< 100).map { String(format: "%02d", $0) } + lower
            + (0 ..< 200).map { "#\($0)" } + upper.flatMap { a in upper.map { a + $0 } }
    }()

    /// Labels that are all single letters are their own codes; otherwise `A`…`Z`, or
    /// `00`, `01`, … past 26 options.
    static func optionCodes(_ labels: [String]) -> [String] {
        let stripped = labels.map(pythonStrip)
        if !stripped.isEmpty, stripped.allSatisfy(isSingleLetter) {
            return stripped
        }
        if labels.count <= 26 {
            return (0 ..< labels.count).map { String(UnicodeScalar(UInt8(65 + $0))) }
        }
        return (0 ..< labels.count).map { String(format: "%02d", $0) }
    }

    /// A distinct single-token code for every label.
    static func aliases(_ labels: [String], tokenizer: any Tokenizer) throws -> [Alias] {
        var used = Set<Int>()
        var out = [Alias]()
        func take(_ code: String) -> Bool {
            let ids = tokenizer.encode(text: code, addSpecialTokens: false)
            guard ids.count == 1, !used.contains(ids[0]) else { return false }
            out.append(Alias(code: code, id: ids[0]))
            used.insert(ids[0])
            return true
        }
        for code in optionCodes(labels) {
            if !take(code), !fallbackPool.contains(where: take) {
                throw DecisionError.noSingleTokenAlias(optionCount: labels.count)
            }
        }
        return out
    }

    /// The aliases of a choice question's options; none for other questions.
    static func aliases(for question: DecisionQuestion, tokenizer: any Tokenizer) throws -> [Alias]
    {
        guard case .choice(let options) = question.criteria else { return [] }
        return try aliases(options.map(\.label), tokenizer: tokenizer)
    }

    /// The distinct ids of the texts that encode to a single token, in order.
    static func singleTokenIDs(_ texts: [String], tokenizer: any Tokenizer) -> [Int] {
        var out = [Int]()
        for text in texts {
            let ids = tokenizer.encode(text: text, addSpecialTokens: false)
            if ids.count == 1, !out.contains(ids[0]) {
                out.append(ids[0])
            }
        }
        return out
    }

    /// The token ids scored for each option: `[no, yes]` for bool, a digit per score level,
    /// the code and the code after a space for choice.
    static func readoutIDs(
        _ question: DecisionQuestion, aliases: [Alias], tokenizer: any Tokenizer
    ) throws -> [[Int]] {
        switch question.criteria {
        case .bool:
            let yes = singleTokenIDs(yesForms, tokenizer: tokenizer)
            let no = singleTokenIDs(noForms, tokenizer: tokenizer)
            guard !yes.isEmpty, !no.isEmpty else {
                throw DecisionError.missingToken("The tokenizer has no single-token yes/no")
            }
            return [no, yes]
        case .score(let levels):
            let groups = levels.indices.map { singleTokenIDs([String($0)], tokenizer: tokenizer) }
            if groups.contains(where: \.isEmpty) {
                throw DecisionError.missingToken(
                    "A score with \(levels.count) levels needs single-token digits")
            }
            return groups
        case .choice:
            return aliases.map { alias in
                [alias.id]
                    + singleTokenIDs([" " + alias.code], tokenizer: tokenizer).filter {
                        $0 != alias.id
                    }
            }
        }
    }

    /// Option probabilities: each option's best log-probability, softmaxed over the options.
    static func readout(_ groups: [[Int]], logProbabilities: [Int: Double]) -> [Double] {
        let scores = groups.map { group in group.map { logProbabilities[$0]! }.max()! }
        let top = scores.max()!
        let exps = scores.map { Foundation.exp($0 - top) }
        let total = exps.reduce(0, +)
        return exps.map { $0 / total }
    }

    static func stateBlock(_ state: DecisionState) -> String {
        state.serialized(indent: 2) + "\n\n"
    }

    /// `codes` are the option codes of a choice question; other questions ignore them.
    static func questionBlock(_ question: DecisionQuestion, codes: [String]) -> String {
        let instructions = question.instructions
        switch question.criteria {
        case .choice(let options):
            let lines = zip(codes, options).map { code, option in
                let text =
                    option.description.flatMap { $0.isEmpty ? nil : $0 }
                    ?? option.label.replacingOccurrences(of: "_", with: " ")
                return "\(code) \(text)"
            }
            return "\(instructions)\n\nOptions:\n\(lines.joined(separator: "\n"))\n\n"
                + "Reply with the option code only."
        case .bool(let whenTrue, let whenFalse):
            // Python prints criteria only for a non-empty dict, and `None` for a missing side.
            var extra = ""
            if whenTrue != nil || whenFalse != nil {
                extra = "\nYes: \(whenTrue ?? "None")\nNo: \(whenFalse ?? "None")"
            }
            return "\(instructions)\(extra)\n\nReply with yes or no only."
        case .score(let levels):
            let legend = levels.enumerated().map { "\($0.offset) \($0.element)" }
            return "\(instructions)\n\n\(legend.joined(separator: "\n"))\n\n"
                + "Reply with a single digit 0-\(levels.count - 1) only."
        }
    }

    /// Everything before the question, shared by all questions on one state.
    static func prefixText(state: DecisionState?, bos: String = "", images: String = "") -> String {
        let body = state.map { "\(stateBlock($0))\nQUESTION:\n" } ?? ""
        return "\(bos)\(imStart)user\n\(images)\(body)"
    }

    /// The question and the assistant header, up to the answer slot.
    static func suffixText(_ question: DecisionQuestion, aliases: [Alias]) -> String {
        "\(questionBlock(question, codes: aliases.map(\.code)))\(imEnd)\n\(imStart)assistant\n"
    }

    /// What the LFM2-VL chat template writes for `count` images at the head of a user turn:
    /// `<image>` per image, with no separator. Python renders the template; d1-3B's matches.
    static func imageMarkup(count: Int) -> String {
        String(repeating: "<image>", count: count)
    }

    /// The size `capPixels` gives an image: at most `maxPixels`, aspect kept, sides truncated.
    static func cappedSize(width: Int, height: Int, maxPixels: Int = visionMaxPixels) -> (
        width: Int, height: Int
    ) {
        guard width * height > maxPixels else { return (width, height) }
        let scale = (Double(maxPixels) / Double(width * height)).squareRoot()
        return (max(1, Int(Double(width) * scale)), max(1, Int(Double(height) * scale)))
    }

    /// The upright image scaled down to at most `maxPixels` with Pillow's bicubic resize, as the
    /// d1 reference's `cap_pixels` does before the LFM2-VL processor.
    static func capPixels(_ image: OrientedImage, maxPixels: Int = visionMaxPixels) -> RGB8Image {
        let size = cappedSize(width: image.width, height: image.height, maxPixels: maxPixels)
        return image.stored.resizedPILBicubic(
            width: size.width, height: size.height, orientation: image.orientation)
    }

    /// Branch batches, longest first, each in request order: at most `budget` padded tokens,
    /// trunk included, and no branch padded past twice its length.
    static func chunks(lengths: [Int], trunk: Int, budget: Int) -> [[Int]] {
        var order = lengths.indices.sorted {
            lengths[$0] != lengths[$1] ? lengths[$0] > lengths[$1] : $0 < $1
        }[...]
        var out = [[Int]]()
        while let first = order.first {
            let longest = lengths[first]
            var size = 1
            while size < order.count, (size + 1) * (trunk + longest) <= budget,
                2 * lengths[order[order.startIndex + size]] >= longest
            {
                size += 1
            }
            out.append(order.prefix(size).sorted())
            order = order.dropFirst(size)
        }
        return out
    }

    /// Python `str.strip()`: Unicode white space plus the separators U+001C to U+001F.
    private static func pythonStrip(_ text: String) -> String {
        func isSpace(_ scalar: Unicode.Scalar) -> Bool {
            scalar.properties.isWhitespace || (0x1C ... 0x1F).contains(scalar.value)
        }
        let scalars = Array(text.unicodeScalars)
        guard let first = scalars.firstIndex(where: { !isSpace($0) }),
            let last = scalars.lastIndex(where: { !isSpace($0) })
        else { return "" }
        return String(String.UnicodeScalarView(scalars[first ... last]))
    }

    /// Python `len(k) == 1 and k.isalpha()`: one code point in a letter category.
    private static func isSingleLetter(_ text: String) -> Bool {
        guard text.unicodeScalars.count == 1, let scalar = text.unicodeScalars.first else {
            return false
        }
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter:
            return true
        default:
            return false
        }
    }
}

// MARK: - Model

/// Token ids and vision input of one decision request.
struct D1Inputs {
    /// The shared prefix, or the whole prompt when there is one question. Image runs are
    /// expanded.
    var trunk: [Int]
    /// One continuation of the trunk per question; empty when there is one question.
    var branches: [[Int]]
    /// The patches the trunk's image tokens read, cut to the longest view; nil without images.
    var vision: Siglip2Patches?
    /// Each question's option aliases; empty for score and bool questions.
    var aliases: [[D1Prompt.Alias]]

    /// The trunk counts once, then each branch.
    var inputTokens: Int { trunk.count + branches.reduce(0) { $0 + $1.count } }
}

/// LiquidAI d1-3B: LFM2-VL weights read as a decision model at the answer slot.
///
/// ``DecisionModelFactory`` wraps the `LFM2VL` that `VLMModelFactory` loads for `lfm2_vl`
/// checkpoints whose `auto_map.AutoModel` ends in `D1Model`.
final class D1: Module, DecisionModel {
    /// The LFM2-VL model, a child module so that `train(false)` reaches it.
    let base: LFM2VL

    /// How images become views and tokens: the checkpoint's processor settings, which the chat
    /// processor also uses.
    var imageSettings: LFM2VLImageSettings

    init(_ base: LFM2VL, imageSettings: LFM2VLImageSettings = .init()) {
        self.base = base
        self.imageSettings = imageSettings
        super.init()
    }

    var decisionCapabilities: DecisionCapabilities {
        DecisionCapabilities(
            questionKinds: [.choice, .score, .bool], media: [.images], maximumScoreLevels: 10)
    }

    func decisionProbabilities(
        _ request: consuming DecisionRequest, tokenizer: any Tokenizer,
        options: DecisionExecutionOptions
    ) async throws -> DecisionProbabilities {
        let (probabilities, inputTokens) = try probabilities(
            state: request.state, questions: request.questions.map(\.question),
            images: request.images, tokenizer: tokenizer, tokenBudget: options.tokenBudget)
        return DecisionProbabilities(probabilities: probabilities, inputTokens: inputTokens)
    }

    /// Each question's option probabilities and the tokens read.
    func probabilities(
        state: DecisionState?, questions: [DecisionQuestion], images: [UserInput.Image] = [],
        tokenizer: any Tokenizer, tokenBudget: Int = DecisionExecutionOptions().tokenBudget
    ) throws -> (probabilities: [[Double]], inputTokens: Int) {
        let inputs = try decisionInputs(
            state: state, questions: questions, images: images, tokenizer: tokenizer)
        let groups = try zip(questions, inputs.aliases).map {
            try D1Prompt.readoutIDs($0, aliases: $1, tokenizer: tokenizer)
        }
        let ids = Array(Set(groups.joined().joined())).sorted()
        let logz = try answerLogProbabilities(
            trunk: inputs.trunk, branches: inputs.branches, vision: inputs.vision, tokens: ids,
            tokenBudget: tokenBudget
        ).asArray(Float.self)
        let probabilities = groups.enumerated().map { index, groups in
            let row = logz[index * ids.count ..< (index + 1) * ids.count]
            return D1Prompt.readout(
                groups,
                logProbabilities: Dictionary(
                    uniqueKeysWithValues: zip(ids, row.map { Double($0) })))
        }
        return (probabilities, inputs.inputTokens)
    }

    /// Trunk ids, branch ids and vision input. One question is a single pass with no
    /// branches; several share the state as the trunk, one branch each. Images go first in
    /// the trunk, each capped at `D1Prompt.visionMaxPixels` and laid out by the LFM2-VL
    /// processor.
    func decisionInputs(
        state: DecisionState?, questions: [DecisionQuestion], images: [UserInput.Image] = [],
        tokenizer: any Tokenizer
    ) throws -> D1Inputs {
        let pixels = try decisionImageErrors {
            try images.map { D1Prompt.capPixels(try OrientedImage($0)) }
        }
        let aliases = try questions.map { try D1Prompt.aliases(for: $0, tokenizer: tokenizer) }
        var prefix = D1Prompt.prefixText(
            state: state, bos: tokenizer.bosToken ?? "",
            images: D1Prompt.imageMarkup(count: pixels.count))
        var branches = zip(questions, aliases).map { D1Prompt.suffixText($0, aliases: $1) }
        if branches.count == 1 {
            prefix += branches[0]
            branches = []
        }
        func encode(_ text: String) -> [Int] {
            tokenizer.encode(text: text, addSpecialTokens: false)
        }
        var inputs = D1Inputs(
            trunk: encode(prefix), branches: branches.map(encode), vision: nil, aliases: aliases)
        if !pixels.isEmpty {
            let processing: LFM2VLImageProcessing
            do {
                processing = try LFM2VLImageProcessing(
                    settings: imageSettings, tokenizer: tokenizer)
            } catch VLMError.processing(let message) {
                // The tokenizer lacks an image marker token.
                throw DecisionError.missingToken(message)
            }
            let processed = try decisionImageErrors { try processing.process(pixels) }
            // The state, or a single question, can contain the text `<image>`.
            guard inputs.trunk.filter({ $0 == processing.imageTokenId }).count == pixels.count
            else {
                throw DecisionError.imagePlaceholderInText("<image>")
            }
            inputs.trunk = try processing.expand(inputs.trunk, tokenRuns: processed.tokenRuns)
            // Python keeps the patch rows up to the longest view.
            var vision = processed.patches
            let longest = vision.spatialShapes.map { $0.0 * $0.1 }.max() ?? 0
            vision.pixels = vision.pixels[0..., ..<longest]
            inputs.vision = vision
        }
        return inputs
    }

    /// The longest trunk read in one pass, unless the token budget is lower. A longer one is
    /// prefilled through the cache in chunks, which bounds the memory a long state needs.
    static let singlePassLimit = 2048

    /// The chunks of a trunk over `singlePassLimit` tokens. On a 16k-token state on an M3 Ultra,
    /// they cost 1-2% time against 11-13% for 512 tokens, and keep most of the memory saving.
    static let trunkPrefill = PrefillParameters(stepSize: 2048)

    /// Answer-slot log-probabilities in float32: of the trunk alone when there are no branches,
    /// else of each branch continuing the trunk, which is read once. The shape is
    /// `[questions, vocabulary]`, or `[questions, tokens.count]` for the given token ids.
    /// A trunk over `min(singlePassLimit, tokenBudget)` tokens is read in chunks as `prefill`
    /// asks, each at most `tokenBudget` tokens.
    func answerLogProbabilities(
        trunk: [Int], branches: [[Int]], vision: Siglip2Patches? = nil, tokens: [Int]? = nil,
        tokenBudget: Int = DecisionExecutionOptions().tokenBudget,
        singlePassLimit: Int = D1.singlePassLimit, prefill: PrefillParameters = D1.trunkPrefill
    ) throws -> MLXArray {
        let columns = tokens.map { MLXArray($0.map { Int32($0) }) }
        // Float32 log-softmax of the vocabulary logits of hidden rows `[n, D]`.
        func logz(_ hidden: MLXArray) -> MLXArray {
            let logits = base.vocabularyLogits(hidden).asType(.float32)
            let z = logits - logSumExp(logits, axis: -1, keepDims: true)
            return columns.map { z.take($0, axis: -1) } ?? z
        }
        let ids = MLXArray(trunk.map { Int32($0) }).reshaped(1, trunk.count)
        let embeddings = try vision.map { vision in
            try base.inputEmbeddings(
                tokens: ids,
                imageFeatures: base.imageFeatures(
                    pixels: vision.pixels, spatialShapes: vision.spatialShapes))
        }
        let chunked = trunk.count > min(singlePassLimit, tokenBudget)
        // Hidden states of the trunk, or of its last position after a chunked prefill.
        func trunkStates(cache: [KVCache]?) throws -> MLXArray {
            guard chunked, let cache else {
                return base.hiddenStates(tokens: ids, embeddings: embeddings, cache: cache)
            }
            let done = try prefill.forEachChunk(
                total: trunk.count, maximumStepSize: tokenBudget
            ) { range in
                _ = base.hiddenStates(
                    tokens: ids[0..., range], embeddings: embeddings?[0..., range, 0...],
                    cache: cache)
                asyncEval(cache)
            }
            return base.hiddenStates(
                tokens: ids[0..., done...], embeddings: embeddings?[0..., done..., 0...],
                cache: cache)
        }
        guard !branches.isEmpty else {
            let hidden = try trunkStates(cache: chunked ? base.newCache(parameters: nil) : nil)
            return logz(hidden[0..., -1, 0...])
        }
        let cache = try base.newCache(parameters: nil)
        _ = try trunkStates(cache: cache)
        eval(cache)
        var out = [MLXArray]()
        let chunks = D1Prompt.chunks(
            lengths: branches.map(\.count), trunk: trunk.count, budget: tokenBudget)
        for chunk in chunks {
            try Task.checkCancellation()
            let rows = chunk.map { branches[$0] }
            let width = rows.map(\.count).max() ?? 0
            let padded = MLXArray(
                rows.flatMap { row in
                    row.map { Int32($0) } + Array(repeating: 0, count: width - row.count)
                }, [rows.count, width])
            let hidden = base.hiddenStates(
                tokens: padded, embeddings: nil, cache: Self.tile(cache, count: rows.count))
            let last = MLXArray(rows.map { Int32($0.count - 1) })
            let z = logz(hidden[MLXArray(Int32(0) ..< Int32(rows.count)), last])
            eval(z)
            out.append(z)
        }
        // Chunks follow length, so put the rows back in branch order.
        var rowOfBranch = [Int32](repeating: 0, count: branches.count)
        for (row, branch) in chunks.joined().enumerated() {
            rowOfBranch[branch] = Int32(row)
        }
        return concatenated(out, axis: 0).take(MLXArray(rowOfBranch), axis: 0)
    }

    /// Fresh caches holding `count` copies of the trunk's state. Conv layers stay
    /// `MambaCache`, which the LFM2 conv layer requires.
    static func tile(_ cache: [KVCache], count: Int) -> [KVCache] {
        func repeated(_ x: MLXArray) -> MLXArray {
            broadcast(x, to: [count] + x.shape.dropFirst())
        }
        return cache.map { source -> KVCache in
            if let conv = source as? MambaCache {
                let copy = MambaCache()
                copy[0] = conv[0].map(repeated)
                return copy
            }
            let copy = KVCacheSimple()
            copy.state = source.state.map(repeated)
            return copy
        }
    }
}

/// Run image decoding or layout code for a decision model, reporting its failures as
/// `DecisionError.invalidImage`.
func decisionImageErrors<T>(_ body: () throws -> T) throws -> T {
    do {
        return try body()
    } catch VLMError.imageProcessingFailure(let message) {
        throw DecisionError.invalidImage(message)
    }
}
