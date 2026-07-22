//
//  DeepseekOCR.swift
//  mlx-swift-lm
//
//  Adapted from https://github.com/mzbac/deepseek-ocr.swift (MIT)
//  and refactored into mlx-swift-lm's MLXVLM model interfaces.
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct DeepseekOCRConfiguration: Decodable, Sendable {

    public struct VisionConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let outputChannels: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numChannels: Int
        public let imageSize: Int
        public let patchSize: Int
        public let layerNormEps: Float
        public let attentionDropout: Float
        public let qkvBias: Bool
        public let useAbsPos: Bool
        public let useRelPos: Bool
        public let windowSize: Int
        public let globalAttentionIndexes: [Int]
        public let mlpDim: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case outputChannels = "output_channels"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numChannels = "num_channels"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case layerNormEps = "layer_norm_eps"
            case attentionDropout = "attention_dropout"
            case qkvBias = "qkv_bias"
            case useAbsPos = "use_abs_pos"
            case useRelPos = "use_rel_pos"
            case windowSize = "window_size"
            case globalAttentionIndexes = "global_attn_indexes"
            case mlpDim = "mlp_dim"
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
            self.outputChannels =
                try container.decodeIfPresent(Int.self, forKey: .outputChannels) ?? 256
            self.numHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
            self.numAttentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
            self.numChannels = try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
            self.imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize) ?? 1024
            self.patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            self.layerNormEps =
                try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            self.attentionDropout =
                try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
            self.qkvBias = try container.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? true
            self.useAbsPos = try container.decodeIfPresent(Bool.self, forKey: .useAbsPos) ?? true
            self.useRelPos = try container.decodeIfPresent(Bool.self, forKey: .useRelPos) ?? true
            self.windowSize = try container.decodeIfPresent(Int.self, forKey: .windowSize) ?? 14
            self.globalAttentionIndexes =
                try container.decodeIfPresent([Int].self, forKey: .globalAttentionIndexes)
                ?? [2, 5, 8, 11]
            self.mlpDim = try container.decodeIfPresent(Int.self, forKey: .mlpDim) ?? 3072
        }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public let vocabSize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let moeIntermediateSize: Int?
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let maxPositionEmbeddings: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let tieWordEmbeddings: Bool
        public let nRoutedExperts: Int?
        public let nSharedExperts: Int?
        public let numExpertsPerTok: Int?
        public let moeLayerFreq: Int?
        public let firstKDenseReplace: Int?
        public let normTopkProb: Bool?
        public let nGroup: Int?
        public let topkGroup: Int?
        public let routedScalingFactor: Float?
        /// Python `TextConfig.scoring_func` default is `"softmax"` (Unlimited / DeepSeek-OCR).
        public let scoringFunc: String?

        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case maxPositionEmbeddings = "max_position_embeddings"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case tieWordEmbeddings = "tie_word_embeddings"
            case lmHead = "lm_head"
            case nRoutedExperts = "n_routed_experts"
            case nSharedExperts = "n_shared_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case moeLayerFreq = "moe_layer_freq"
            case firstKDenseReplace = "first_k_dense_replace"
            case normTopkProb = "norm_topk_prob"
            case nGroup = "n_group"
            case topkGroup = "topk_group"
            case routedScalingFactor = "routed_scaling_factor"
            case scoringFunc = "scoring_func"
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 129_280
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6848
            self.moeIntermediateSize = try container.decodeIfPresent(
                Int.self, forKey: .moeIntermediateSize)
            self.numHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
            self.numAttentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 10
            self.numKeyValueHeads =
                try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 10
            self.maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32_768
            self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
            // Unlimited packs omit tie_word_embeddings but set lm_head=true when untied.
            if let tie = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) {
                self.tieWordEmbeddings = tie
            } else if let hasLmHead = try container.decodeIfPresent(Bool.self, forKey: .lmHead) {
                self.tieWordEmbeddings = !hasLmHead
            } else {
                self.tieWordEmbeddings = true
            }
            self.nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
            self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
            self.numExpertsPerTok = try container.decodeIfPresent(
                Int.self, forKey: .numExpertsPerTok)
            self.moeLayerFreq = try container.decodeIfPresent(Int.self, forKey: .moeLayerFreq)
            self.firstKDenseReplace = try container.decodeIfPresent(
                Int.self, forKey: .firstKDenseReplace)
            self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
            self.nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup)
            self.topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup)
            self.routedScalingFactor =
                try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor)
            self.scoringFunc = try container.decodeIfPresent(String.self, forKey: .scoringFunc)
        }
    }

    public struct BaseConfiguration: Decodable, Sendable {
        public let modelType: String
        private let _imageTokenId: Int?
        public var imageTokenId: Int { _imageTokenId ?? 128_815 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _imageTokenId = "image_token_id"
        }
    }

    public let visionConfiguration: VisionConfiguration
    public let textConfiguration: TextConfiguration
    public let baseConfiguration: BaseConfiguration
    private let _imageSeqLength: Int?
    public var imageSeqLength: Int { _imageSeqLength ?? 576 }

    enum CodingKeys: String, CodingKey {
        case visionConfiguration = "vision_config"
        case textConfiguration = "text_config"
        case languageConfiguration = "language_config"
        case _imageSeqLength = "image_seq_length"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)
        if let text = try container.decodeIfPresent(
            TextConfiguration.self, forKey: .textConfiguration)
        {
            self.textConfiguration = text
        } else {
            self.textConfiguration = try container.decode(
                TextConfiguration.self, forKey: .languageConfiguration)
        }
        self.baseConfiguration = try BaseConfiguration(from: decoder)
        self._imageSeqLength = try container.decodeIfPresent(Int.self, forKey: ._imageSeqLength)
    }
}

public struct DeepseekOCRProcessorConfiguration: Decodable, Sendable {
    public struct CandidateResolution: Decodable, Sendable {
        public let width: Int
        public let height: Int

        public init(width: Int, height: Int) {
            self.width = width
            self.height = height
        }

        public init(from decoder: any Decoder) throws {
            var container = try decoder.unkeyedContainer()
            self.width = try container.decode(Int.self)
            self.height = try container.decode(Int.self)
        }
    }

    public struct Size: Decodable, Sendable {
        private let _shortestEdge: Int?
        private let _longestEdge: Int?

        public var shortestEdge: Int { _shortestEdge ?? 1024 }
        public var longestEdge: Int { _longestEdge ?? 1024 }

        enum CodingKeys: String, CodingKey {
            case _shortestEdge = "shortest_edge"
            case _longestEdge = "longest_edge"
        }

        public init() {
            self._shortestEdge = nil
            self._longestEdge = nil
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let candidateResolutions: [CandidateResolution]
    public let patchSize: Int
    public let downsampleRatio: Int
    public let imageToken: String
    public let size: Size
    private let _imageSeqLength: Int?

    public var imageSeqLength: Int { _imageSeqLength ?? 576 }
    public var baseSize: Int { candidateResolutions.first?.width ?? size.longestEdge }
    public var localImageSize: Int { min(size.shortestEdge, 640) }
    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case candidateResolutions = "candidate_resolutions"
        case patchSize = "patch_size"
        case downsampleRatio = "downsample_ratio"
        case imageToken = "image_token"
        case size
        case _imageSeqLength = "image_seq_length"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.imageMean =
            try container.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        self.imageStd =
            try container.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        self.candidateResolutions =
            try container.decodeIfPresent([CandidateResolution].self, forKey: .candidateResolutions)
            ?? [.init(width: 1024, height: 1024)]
        self.patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        self.downsampleRatio =
            try container.decodeIfPresent(Int.self, forKey: .downsampleRatio) ?? 4
        self.imageToken =
            try container.decodeIfPresent(String.self, forKey: .imageToken) ?? "<image>"
        self.size = try container.decodeIfPresent(Size.self, forKey: .size) ?? Size()
        self._imageSeqLength = try container.decodeIfPresent(Int.self, forKey: ._imageSeqLength)
    }

    public init() {
        self.imageMean = [0.5, 0.5, 0.5]
        self.imageStd = [0.5, 0.5, 0.5]
        self.candidateResolutions = [.init(width: 1024, height: 1024)]
        self.patchSize = 16
        self.downsampleRatio = 4
        self.imageToken = "<image>"
        self.size = Size()
        self._imageSeqLength = nil
    }
}

/// DeepSeek-OCR grounding / localization special tokens (mlx-vlm + HF packs).
///
/// These strings are registered in the Hub tokenizer (`added_tokens_decoder`);
/// resolve IDs via ``id(of:tokenizer:)`` rather than hard-coding in hot paths.
/// Prompt helpers match Python / mlx-vlm DeepSeek-OCR README usage.
///
/// **Layout decode deferral:** this surface exposes tokens + prompt builders and a
/// minimal `<|det|>[[x1,y1,x2,y2]]` extractor. Building a full structured layout
/// tree from interleaved `<|ref|>…<|/ref|><|det|>…` markdown is intentionally
/// deferred — callers should keep `skipSpecialTokens: false` on decode and parse
/// what they need (or consume raw grounding markdown).
public enum DeepseekOCRSpecialTokens: String, Sendable, CaseIterable {
    case grounding = "<|grounding|>"
    case refOpen = "<|ref|>"
    case refClose = "<|/ref|>"
    case detOpen = "<|det|>"
    case detClose = "<|/det|>"

    /// Fallback IDs from deepseek-ai / mlx-community / majentik tokenizer packs
    /// (`<image>` is 128815; grounding family follows contiguously).
    public var defaultId: Int {
        switch self {
        case .refOpen: return 128_816
        case .refClose: return 128_817
        case .detOpen: return 128_818
        case .detClose: return 128_819
        case .grounding: return 128_820
        }
    }

    /// Prefer tokenizer lookup; fall back to ``defaultId`` for known HF packs.
    public static func id(of token: DeepseekOCRSpecialTokens, tokenizer: any Tokenizer) -> Int {
        tokenizer.convertTokenToId(token.rawValue) ?? token.defaultId
    }

    /// All grounding-family strings (openers + closers), in stable CaseIterable order.
    public static var allStrings: [String] { allCases.map(\.rawValue) }

    /// Resolve every grounding-family token through the tokenizer (with defaults).
    public static func resolveIds(tokenizer: any Tokenizer) -> [DeepseekOCRSpecialTokens: Int] {
        Dictionary(uniqueKeysWithValues: allCases.map { ($0, id(of: $0, tokenizer: tokenizer)) })
    }

    // MARK: Prompt builders (match mlx-vlm DeepSeek-OCR README)

    /// Structured OCR / layout prompt, e.g. `"<|grounding|>OCR this image."`.
    public static func groundingPrompt(_ instruction: String = "OCR this image.") -> String {
        let trimmed = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
        let body = trimmed.hasSuffix(".") ? trimmed : trimmed + "."
        return "\(Self.grounding.rawValue)\(body)"
    }

    /// Document → markdown with layout tags.
    public static func groundingMarkdownPrompt(
        _ instruction: String = "Convert the document to markdown."
    ) -> String {
        groundingPrompt(instruction)
    }

    /// Text localization: `"Locate <|ref|>…<|/ref|> in the image."`.
    public static func locatePrompt(_ text: String) -> String {
        "Locate \(Self.refOpen.rawValue)\(text)\(Self.refClose.rawValue) in the image."
    }

    /// Normalized 0–1000 axis-aligned box from a `<|det|>[[x1, y1, x2, y2]]` span.
    public struct Detection: Equatable, Sendable {
        public let x1: Int
        public let y1: Int
        public let x2: Int
        public let y2: Int

        public init(x1: Int, y1: Int, x2: Int, y2: Int) {
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
        }
    }

    /// Extract `[[x1,y1,x2,y2]]` boxes from model output that still contains special tokens.
    /// Does **not** rebuild a layout tree — see type-level deferral note.
    public static func parseDetections(from text: String) -> [Detection] {
        // Match [[a, b, c, d]] (as emitted inside <|det|>…<|/det|>).
        let pattern =
            #"\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        return regex.matches(in: text, range: range).compactMap { match in
            guard match.numberOfRanges == 5 else { return nil }
            let ints = (1 ... 4).compactMap { i -> Int? in
                Int(ns.substring(with: match.range(at: i)))
            }
            guard ints.count == 4 else { return nil }
            return Detection(x1: ints[0], y1: ints[1], x2: ints[2], y2: ints[3])
        }
    }
}

public struct DeepseekOCRProcessor: UserInputProcessor {
    /// Image crop modes matching Python `DeepseekOCRProcessor.tokenize_with_images`.
    ///
    /// Select via ``modeContext(_:)`` on `ChatSession` / `UserInput.additionalContext`,
    /// or the `MODE` env var in `DeepseekOCRSmoke` / `scripts/swift_smoke.sh`.
    public enum Mode: String, Sendable, CaseIterable {
        /// Multi-resolution OCR: 1024² global view + 640² local tiles when the page
        /// exceeds 640×640 (Python `cropping=True`, `base_size=1024`). Default.
        case gundam
        /// Single 640² padded view, no local tiles (Python `cropping=False`).
        /// Cheaper / shorter context; required for multipage fusion prompts.
        case base
    }

    @_spi(Testing)
    public struct PreparedImageInputs: @unchecked Sendable {
        public let inputIds: MLXArray
        public let pixelValues: MLXArray
        public let localCrops: MLXArray
        public let imagesSeqMask: MLXArray
        public let imagesSpatialCrop: MLXArray
        public let mode: Mode
    }

    /// `UserInput.additionalContext` / `ChatSession.additionalContext` key for ``Mode``.
    /// Value must be the mode's `rawValue` (`"gundam"` or `"base"`).
    public static let modeContextKey = "deepseekocr_mode"

    /// Context dictionary selecting ``Mode`` for `ChatSession` or `UserInput`.
    public static func modeContext(_ mode: Mode) -> [String: any Sendable] {
        [modeContextKey: mode.rawValue]
    }

    /// Resolves ``Mode`` from additional context; unknown or missing → ``Mode/gundam``.
    public static func mode(from additionalContext: [String: any Sendable]?) -> Mode {
        guard let raw = additionalContext?[modeContextKey] as? String,
            let mode = Mode(rawValue: raw)
        else {
            return .gundam
        }
        return mode
    }

    private let config: DeepseekOCRProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: DeepseekOCRProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func preprocess(image: CIImage, side: Int) -> MLXArray {
        // Match PIL ImageOps.pad(image, (side, side)): contain-resize then pad.
        // Match PIL ImageOps.pad — no extra tone-curve convert (PIL keeps sRGB code values).
        let srgb = image
        let extent = srgb.extent.integral
        let width = extent.width
        let height = extent.height
        let targetSide = CGFloat(side)
        let contained: CGSize
        let imRatio = width / height
        if imRatio > 1 {
            let newHeight = (height / width * targetSide).rounded()
            contained = CGSize(width: targetSide, height: newHeight)
        } else if imRatio < 1 {
            let newWidth = (width / height * targetSide).rounded()
            contained = CGSize(width: newWidth, height: targetSide)
        } else {
            contained = CGSize(width: targetSide, height: targetSide)
        }

        // Clamp before scale so bicubic edge taps don't sample empty (→ washed borders).
        let resized = resampleClamped(srgb, to: contained)
        // PIL ImageOps.pad uses color=tuple(int(mean*255) for mean in ...)
        // i.e. 127/255 for mean 0.5 — not exact 0.5 (normalize residual ~-0.00392).
        func pilMeanByte(_ mean: CGFloat) -> CGFloat {
            CGFloat(Int(mean * 255)) / 255.0
        }
        // Core Image's default render emits linear light while Python/PIL keep
        // sRGB code values. Compensate mid-gray: specify sRGB-encoded value so
        // linear output equals the PIL byte-quantized mean.
        func srgbEncode(_ linear: CGFloat) -> CGFloat {
            if linear <= 0.0031308 { return linear * 12.92 }
            return 1.055 * pow(linear, 1.0 / 2.4) - 0.055
        }
        let padLinear = (
            pilMeanByte(config.imageMean[0]),
            pilMeanByte(config.imageMean[1]),
            pilMeanByte(config.imageMean[2])
        )
        let padColor = CIColor(
            red: srgbEncode(padLinear.0),
            green: srgbEncode(padLinear.1),
            blue: srgbEncode(padLinear.2))
        let canvas = CIImage(color: padColor).cropped(
            to: CGRect(x: 0, y: 0, width: targetSide, height: targetSide))
        let dx = ((targetSide - contained.width) * 0.5).rounded() - resized.extent.origin.x
        let dy = ((targetSide - contained.height) * 0.5).rounded() - resized.extent.origin.y
        let placed = resized.transformed(by: CGAffineTransform(translationX: dx, y: dy))

        // Normalize with the same byte-quantized mean PIL effectively uses in the
        // pad region; content uses config mean (identical for 0.5 → still 0.5 after
        // int path only affects pads). Match Python: mean/std from processor config.
        return placed.composited(over: canvas)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
            .asMLXArray()
            .asType(.bfloat16)
    }

    /// Bicubic resize with edge clamp — CI's default samples empty outside the
    /// extent, which washes tile borders (local crop corners were ~0.70 not 1.0).
    private func resampleClamped(_ image: CIImage, to size: CGSize) -> CIImage {
        let extent = image.extent
        let yScale = size.height / extent.height
        let xScale = size.width / extent.width
        let filter = CIFilter.bicubicScaleTransform()
        filter.inputImage = image.clampedToExtent()
        filter.scale = Float(yScale)
        filter.aspectRatio = Float(xScale / yScale)
        let scaled = filter.outputImage!
        let target = CGRect(
            x: extent.origin.x * xScale,
            y: extent.origin.y * yScale,
            width: size.width,
            height: size.height)
        return
            scaled
            .cropped(to: target)
            .transformed(by: CGAffineTransform(translationX: -target.minX, y: -target.minY))
    }

    private func imageTokenId() -> Int {
        tokenizer.convertTokenToId(config.imageToken) ?? DeepseekOCR.defaultImageTokenId
    }

    private func promptMode(from input: UserInput) -> Mode {
        Self.mode(from: input.additionalContext)
    }

    private func numQueries(for side: Int) -> Int {
        Int(ceil((Double(side) / Double(config.patchSize)) / Double(config.downsampleRatio)))
    }

    private func makeImageTokenGrid(width: Int, height: Int, includeSeparator: Bool = true)
        -> [Int]
    {
        let token = imageTokenId()
        var tokens = [Int]()
        for _ in 0 ..< height {
            tokens.append(contentsOf: Array(repeating: token, count: width))
            tokens.append(token)
        }
        if includeSeparator {
            tokens.append(token)
        }
        return tokens
    }

    private func makeImagePromptTokens(mode: Mode, cropWidth: Int, cropHeight: Int) -> [Int] {
        switch mode {
        case .gundam:
            // Match Python unlimited_ocr tokenize_with_images placeholder order:
            // global grid, one separator, then local crops (if any).
            // Feature merge stays local→global→view_separator (see getImageFeatures);
            // that intentional token/feature order mismatch is how the model was trained.
            let baseQueries = numQueries(for: config.baseSize)
            var tokens = [Int]()
            tokens.append(
                contentsOf: makeImageTokenGrid(
                    width: baseQueries, height: baseQueries, includeSeparator: false))
            tokens.append(imageTokenId())
            if cropWidth > 1 || cropHeight > 1 {
                let localQueries = numQueries(for: config.localImageSize)
                tokens.append(
                    contentsOf: makeImageTokenGrid(
                        width: localQueries * cropWidth,
                        height: localQueries * cropHeight,
                        includeSeparator: false))
            }
            return tokens
        case .base:
            let queries = numQueries(for: config.localImageSize)
            return makeImageTokenGrid(width: queries, height: queries)
        }
    }

    private func closestAspectRatio(
        for aspectRatio: CGFloat,
        candidateRatios: [(Int, Int)],
        imageWidth: CGFloat,
        imageHeight: CGFloat,
        tileSize: Int
    ) -> (Int, Int) {
        var bestRatio = (1, 1)
        var bestDifference = CGFloat.greatestFiniteMagnitude
        let area = imageWidth * imageHeight

        for ratio in candidateRatios {
            let targetAspectRatio = CGFloat(ratio.0) / CGFloat(ratio.1)
            let difference = abs(aspectRatio - targetAspectRatio)
            if difference < bestDifference {
                bestDifference = difference
                bestRatio = ratio
            } else if difference == bestDifference {
                let threshold = 0.5 * CGFloat(tileSize * tileSize * ratio.0 * ratio.1)
                if area > threshold {
                    bestRatio = ratio
                }
            }
        }

        return bestRatio
    }

    private func dynamicPreprocess(image: CIImage) -> ([MLXArray], Int, Int) {
        let tileSize = config.localImageSize
        let width = image.extent.width
        let height = image.extent.height
        let aspectRatio = width / height

        var targetRatios = [(Int, Int)]()
        for n in 2 ... 9 {
            for i in 1 ... n {
                for j in 1 ... n where i * j <= 9 && i * j >= 2 {
                    let candidate = (i, j)
                    if !targetRatios.contains(where: { $0.0 == candidate.0 && $0.1 == candidate.1 })
                    {
                        targetRatios.append(candidate)
                    }
                }
            }
        }

        let sortedRatios = targetRatios.sorted { lhs, rhs in
            (lhs.0 * lhs.1, lhs.0, lhs.1) < (rhs.0 * rhs.1, rhs.0, rhs.1)
        }
        let (tilesWide, tilesHigh) = closestAspectRatio(
            for: aspectRatio,
            candidateRatios: sortedRatios,
            imageWidth: width,
            imageHeight: height,
            tileSize: tileSize)

        let targetWidth = tileSize * tilesWide
        let targetHeight = tileSize * tilesHigh
        let resized = resampleClamped(
            image, to: .init(width: targetWidth, height: targetHeight))

        var processed = [MLXArray]()
        for block in 0 ..< (tilesWide * tilesHigh) {
            let x = (block % tilesWide) * tileSize
            let y = (block / tilesWide) * tileSize
            let cropRect = CGRect(x: x, y: y, width: tileSize, height: tileSize)
            let crop =
                resized
                .cropped(to: cropRect)
                .transformed(by: .init(translationX: -cropRect.minX, y: -cropRect.minY))
            processed.append(
                crop
                    .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
                    .asMLXArray()
                    .asType(.bfloat16))
        }

        return (processed, tilesWide, tilesHigh)
    }

    private func userPromptText(from input: UserInput) -> String {
        switch input.prompt {
        case .text(let text):
            return text
        case .chat(let messages):
            return messages.last(where: { $0.role == .user })?.content
                ?? messages.last?.content
                ?? ""
        case .messages(let messages):
            for message in messages.reversed() {
                if let role = message["role"] as? String, role != "user" { continue }
                if let content = message["content"] as? String {
                    return content
                }
                if let parts = message["content"] as? [[String: any Sendable]] {
                    let text = parts.compactMap { $0["text"] as? String }.joined()
                    if !text.isEmpty { return text }
                }
            }
            return ""
        }
    }

    private func emptyLocalCrops() -> MLXArray {
        zeros([1, 3, config.baseSize, config.baseSize], type: Float.self).asType(.bfloat16)
    }

    @_spi(Testing)
    public func prepareForTesting(input: UserInput) async throws -> PreparedImageInputs {
        guard !input.images.isEmpty else {
            let promptTokens = tokenizer.encode(
                text: userPromptText(from: input), addSpecialTokens: true)
            return .init(
                inputIds: MLXArray(promptTokens.map(Int32.init)).reshaped(1, promptTokens.count),
                pixelValues: zeros([1, 3, config.baseSize, config.baseSize], type: Float.self),
                localCrops: emptyLocalCrops(),
                imagesSeqMask: zeros([1, promptTokens.count], type: Bool.self),
                imagesSpatialCrop: zeros([1, 2], type: Int32.self),
                mode: promptMode(from: input))
        }

        guard input.images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        let mode = promptMode(from: input)
        let image = MediaProcessing.apply(
            try input.images[0].asCIImage(), processing: input.processing)

        let pixelValues: MLXArray
        let localCrops: MLXArray
        let cropWidth: Int
        let cropHeight: Int

        switch mode {
        case .gundam:
            pixelValues = preprocess(image: image, side: config.baseSize)
            if image.extent.width <= CGFloat(config.localImageSize)
                && image.extent.height <= CGFloat(config.localImageSize)
            {
                cropWidth = 1
                cropHeight = 1
                localCrops = emptyLocalCrops()
            } else {
                let (crops, widthTiles, heightTiles) = dynamicPreprocess(image: image)
                cropWidth = widthTiles
                cropHeight = heightTiles
                localCrops = concatenated(crops, axis: 0).asType(.bfloat16)
            }
        case .base:
            pixelValues = preprocess(image: image, side: config.localImageSize)
            cropWidth = 1
            cropHeight = 1
            localCrops = emptyLocalCrops()
        }

        // Match mlx-vlm unlimited_ocr tokenize_with_images:
        //   [bos] + image lattice + raw user text
        // (no chat-template wrappers — prefixes before <image> break OCR).
        let bosId =
            tokenizer.bosToken.flatMap { tokenizer.convertTokenToId($0) } ?? 0
        let imagePromptTokens = makeImagePromptTokens(
            mode: mode, cropWidth: cropWidth, cropHeight: cropHeight)
        let textTokens = tokenizer.encode(
            text: userPromptText(from: input), addSpecialTokens: false)
        let fullPromptTokens = [bosId] + imagePromptTokens + textTokens
        let sequenceMask =
            [false]
            + Array(repeating: true, count: imagePromptTokens.count)
            + Array(repeating: false, count: textTokens.count)

        return .init(
            inputIds: MLXArray(fullPromptTokens.map(Int32.init)).reshaped(
                1, fullPromptTokens.count),
            pixelValues: pixelValues,
            localCrops: localCrops,
            imagesSeqMask: MLXArray(sequenceMask).reshaped(1, sequenceMask.count),
            imagesSpatialCrop: MLXArray([Int32(cropWidth), Int32(cropHeight)]).reshaped(1, 2),
            mode: mode)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let prepared = try await prepareForTesting(input: input)
        let mask = ones(like: prepared.inputIds).asType(.int8)
        let cropWidth = prepared.imagesSpatialCrop[0, 0].item(Int.self)
        let cropHeight = prepared.imagesSpatialCrop[0, 1].item(Int.self)
        let hasLocalCrops = cropWidth > 1 || cropHeight > 1

        // Pack gundam locals through `video` and spatial crop through `positionIds`
        // so DeepseekOCR.prepare can assemble features without extending LMInput.
        return LMInput(
            text: .init(tokens: prepared.inputIds, mask: mask),
            image: .init(
                pixels: prepared.pixelValues,
                positionIds: prepared.imagesSpatialCrop,
                frames: [
                    THW(
                        1,
                        prepared.pixelValues.dim(2),
                        prepared.pixelValues.dim(3))
                ]),
            video: hasLocalCrops
                ? .init(
                    pixels: prepared.localCrops,
                    frames: [
                        THW(prepared.localCrops.dim(0), cropWidth, cropHeight)
                    ])
                : nil)
    }
}

public struct DeepseekOCRMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var dictionary: MLXLMCommon.Message = [
            "role": message.role.rawValue,
            "content": [["type": "text", "text": message.content]]
                + message.images.map { _ in ["type": "image"] },
        ]
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

public final class DeepseekOCR: Module, VLMModel, KVCacheDimensionProvider {
    static let defaultImageTokenId = 128_815

    @ModuleInfo(key: "sam_model") private var samModel: VisionEncoder
    @ModuleInfo(key: "vision_model") private var clipModel: ClipVisionModel
    @ModuleInfo(key: "projector") private var projector: MultiModalProjector
    @ModuleInfo(key: "model") private var languageModel: TextModel
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    public let config: DeepseekOCRConfiguration
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.layers }
    public var vocabularySize: Int { config.textConfiguration.vocabSize }

    private var imageNewline: MLXArray
    private var viewSeparator: MLXArray

    public init(_ config: DeepseekOCRConfiguration) {
        self.config = config
        self._samModel.wrappedValue = VisionEncoder(config.visionConfiguration)
        self._clipModel.wrappedValue = ClipVisionModel()
        self._projector.wrappedValue = MultiModalProjector(config)
        self._languageModel.wrappedValue = TextModel(config.textConfiguration)
        if !config.textConfiguration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.textConfiguration.hiddenSize,
                config.textConfiguration.vocabSize,
                bias: false)
        }

        let std = Float(1.0 / sqrt(Double(config.textConfiguration.hiddenSize)))
        self.imageNewline = MLXRandom.normal([config.textConfiguration.hiddenSize]) * std
        self.viewSeparator = MLXRandom.normal([config.textConfiguration.hiddenSize]) * std
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        state: LMOutput.State?,
        windowSize _: Int?
    ) throws -> PrepareResult {
        let pixels = input.image?.pixels.asType(samModel.patchEmbed.proj.weight.dtype)
        let embeddings: MLXArray
        if let pixels {
            // Processor / MediaProcessing emit NCHW [B,C,H,W]; MLX Conv2d wants NHWC.
            let globalPixels = nchwToNhwc(pixels)
            let localPixels = input.video.map { nchwToNhwc($0.pixels.asType(pixels.dtype)) }
            let spatial = input.image?.positionIds
            let cropWidth = spatial.map { $0[0, 0].item(Int.self) } ?? 1
            let cropHeight = spatial.map { $0[0, 1].item(Int.self) } ?? 1
            let imageFeatures = getImageFeatures(
                globalPixels: globalPixels,
                localPixels: localPixels,
                cropWidth: cropWidth,
                cropHeight: cropHeight)
            embeddings = mergeInputIdsWithImageFeatures(
                inputIds: input.text.tokens, imageFeatures: imageFeatures)
        } else {
            embeddings = languageModel.embedTokens(input.text.tokens)
        }
        let logits = computeLogits(languageModel.forward(embeddings, cache: cache))
        return .logits(LMOutput(logits: logits, state: state ?? .init()))
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [any KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        let logits = computeLogits(languageModel(input.tokens, cache: cache))
        return LMOutput(logits: logits, state: state ?? .init())
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Normalize Unlimited-OCR pack keys onto the DeepseekOCR module tree:
        // language_model.model.* → model.*, root sam/vision/projector → model.*,
        // view_separator / image_newline → camelCase Module parameters.
        var newWeights = [String: MLXArray]()
        for (key, value) in weights {
            var normalized = key
            if normalized.hasPrefix("language_model.model.") {
                normalized =
                    "model." + String(normalized.dropFirst("language_model.model.".count))
            } else if normalized.hasPrefix("language_model.lm_head.") {
                normalized =
                    "lm_head." + String(normalized.dropFirst("language_model.lm_head.".count))
            } else if normalized == "view_separator" || normalized == "view_seperator"
                || normalized.hasSuffix(".view_separator")
                || normalized.hasSuffix(".view_seperator")
            {
                normalized = "viewSeparator"
            } else if normalized == "image_newline" || normalized.hasSuffix(".image_newline") {
                normalized = "imageNewline"
            } else if normalized.hasPrefix("sam_model.") || normalized.hasPrefix("vision_model.")
                || normalized.hasPrefix("projector.")
            {
                normalized = "model." + normalized
            }
            newWeights[normalized] = value
        }

        for layer in 0 ..< config.textConfiguration.numHiddenLayers {
            let prefix = "model.layers.\(layer)"
            for proj in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases", "bias"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(proj).\(key)"
                    guard newWeights[firstKey] != nil else { continue }
                    let expertCount = config.textConfiguration.nRoutedExperts ?? 1
                    let stackedExperts = (0 ..< expertCount).compactMap {
                        newWeights["\(prefix).mlp.experts.\($0).\(proj).\(key)"]
                    }
                    guard !stackedExperts.isEmpty else { continue }
                    newWeights["\(prefix).mlp.switch_mlp.\(proj).\(key)"] = stacked(stackedExperts)
                }
            }

            // Unlimited / some DeepSeek packs omit MoE gate correction bias; zeros are a no-op.
            let gateWeightKey = "\(prefix).mlp.gate.weight"
            let gateBiasKey = "\(prefix).mlp.gate.e_score_correction_bias"
            if newWeights[gateBiasKey] == nil, let gateWeight = newWeights[gateWeightKey] {
                newWeights[gateBiasKey] = zeros([gateWeight.dim(0)])
            }
        }

        var result = [String: MLXArray]()
        for (key, value) in newWeights {
            var newKey: String?
            var adjusted = value

            if key == "viewSeparator" || key == "imageNewline" {
                newKey = key
            } else if key == "model.projector.layers.weight" || key == "projector.layers.weight" {
                newKey = "projector.layers.weight"
            } else if key == "model.projector.layers.bias" || key == "projector.layers.bias" {
                newKey = "projector.layers.bias"
            } else if key.hasPrefix("model.projector.") {
                // Quantized projector extras (scales/biases) under Unlimited packs.
                newKey = "projector." + String(key.dropFirst("model.projector.".count))
            } else if key.hasPrefix("lm_head.") {
                newKey = key
            } else if key.hasPrefix("model.sam_model.") {
                var suffix = String(key.dropFirst("model.sam_model.".count))
                if suffix.hasPrefix("neck.0.") {
                    suffix = suffix.replacingOccurrences(of: "neck.0.", with: "neck.conv1.")
                } else if suffix.hasPrefix("neck.1.") {
                    suffix = suffix.replacingOccurrences(of: "neck.1.", with: "neck.layer_norm1.")
                } else if suffix.hasPrefix("neck.2.") {
                    suffix = suffix.replacingOccurrences(of: "neck.2.", with: "neck.conv2.")
                } else if suffix.hasPrefix("neck.3.") {
                    suffix = suffix.replacingOccurrences(of: "neck.3.", with: "neck.layer_norm2.")
                }
                suffix = suffix.replacingOccurrences(of: "blocks.", with: "layers.")
                newKey = "sam_model.\(suffix)"
            } else if key.hasPrefix("model.vision_model.") {
                // ModuleInfo key is vision_model (property is clipModel).
                newKey = "vision_model.\(key.dropFirst("model.vision_model.".count))"
            } else if key.hasPrefix("model.") {
                guard !key.contains("projector") else { continue }
                guard !key.contains("sam_model") else { continue }
                guard !key.contains("vision_model") else { continue }
                newKey = String(key)
            }

            guard let finalKey = newKey, !finalKey.contains("rotary_emb.inv_freq") else { continue }
            if finalKey.contains("proj.weight") || finalKey.contains("conv")
                || finalKey.contains("patch_embed") || finalKey.contains("patch_embedding")
            {
                // PyTorch Conv2d is [O,I,H,W]; MLX is [O,H,W,I]. Unlimited MLX packs
                // already ship OHWI — only transpose when the layout still looks OIHW.
                if value.ndim == 4 {
                    let channelOrSpatial = value.dim(1)
                    let spatialA = value.dim(2)
                    let spatialB = value.dim(3)
                    let looksLikePyTorchOIHW =
                        spatialA == spatialB && channelOrSpatial != spatialA
                    if looksLikePyTorchOIHW {
                        adjusted = value.transposed(0, 2, 3, 1)
                    }
                }
            }

            // Bare MLXArray Module properties use camelCase paths (no @ModuleInfo key).
            var resolvedKey = finalKey
            let leafRemaps = [
                "pos_embed": "posEmbed",
                "rel_pos_h": "relPosH",
                "rel_pos_w": "relPosW",
                "class_embedding": "classEmbedding",
            ]
            for (snake, camel) in leafRemaps {
                if resolvedKey.hasSuffix(".\(snake)") {
                    resolvedKey =
                        String(resolvedKey.dropLast(snake.count)) + camel
                } else if resolvedKey == snake {
                    resolvedKey = camel
                }
            }
            // HF packs store CLIP position embeddings as Embedding.weight [N, D];
            // ClipVisionEmbeddings.positionEmbedding is [1, N, D].
            if resolvedKey.hasSuffix(".position_embedding.weight") {
                resolvedKey =
                    String(resolvedKey.dropLast(".position_embedding.weight".count))
                    + ".positionEmbedding"
                if adjusted.ndim == 2 {
                    adjusted = adjusted.reshaped(1, adjusted.dim(0), adjusted.dim(1))
                }
            }

            result[resolvedKey] = adjusted
        }

        if config.textConfiguration.tieWordEmbeddings {
            result["lm_head.weight"] = nil
        }

        return result
    }

    private func nchwToNhwc(_ pixels: MLXArray) -> MLXArray {
        if pixels.ndim == 4 && pixels.dim(1) <= 4 && pixels.dim(1) < pixels.dim(2) {
            return pixels.transposed(0, 2, 3, 1)
        }
        return pixels
    }

    private func computeLogits(_ hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return languageModel.embedTokens.asLinear(hiddenStates)
    }

    /// Append per-row `imageNewline` tokens to a [H, W, D] feature map and flatten.
    private func appendImageNewlines(_ featuresHW: MLXArray) -> MLXArray {
        let height = featuresHW.dim(0)
        let hiddenSize = featuresHW.dim(2)
        let newline = imageNewline[.newAxis, .newAxis, 0...]
        let newlineBroadcast = broadcast(newline, to: [height, 1, hiddenSize])
        return concatenated([featuresHW, newlineBroadcast], axis: 1).reshaped(-1, hiddenSize)
    }

    private func getImageFeatures(
        globalPixels: MLXArray,
        localPixels: MLXArray?,
        cropWidth: Int,
        cropHeight: Int
    ) -> MLXArray {
        let hasLocal =
            (cropWidth > 1 || cropHeight > 1)
            && localPixels != nil
            && (localPixels?.dim(0) ?? 0) > 0

        let globalProjected = projectedImageFeatures(globalPixels)[0]
        let globalHW = globalProjected.dim(0)
        let hiddenSize = globalProjected.dim(1)
        let globalSide = Int(sqrt(Double(globalHW)))
        let globalFeatures = appendImageNewlines(
            globalProjected.reshaped(globalSide, globalSide, hiddenSize))

        if hasLocal, let localPixels {
            let localProjected = projectedImageFeatures(localPixels)
            // [N, hw, D] → tile grid [cropH, cropW, h, w, D] → [cropH*h, cropW*w, D]
            let tileTokens = localProjected.dim(1)
            let tileSide = Int(sqrt(Double(tileTokens)))
            let tiled = localProjected.reshaped(
                cropHeight, cropWidth, tileSide, tileSide, hiddenSize
            )
            .transposed(0, 2, 1, 3, 4)
            .reshaped(cropHeight * tileSide, cropWidth * tileSide, hiddenSize)
            let localFeatures = appendImageNewlines(tiled)
            let separator = viewSeparator[.newAxis, 0...]
            let merged = concatenated([localFeatures, globalFeatures, separator], axis: 0)
            return merged.reshaped(1, merged.dim(0), hiddenSize)
        }

        let separator = viewSeparator[.newAxis, 0...]
        let merged = concatenated([globalFeatures, separator], axis: 0)
        return merged.reshaped(1, merged.dim(0), hiddenSize)
    }

    private func fusedVisionFeatures(_ pixelValues: MLXArray) -> MLXArray {
        let samFeatures = samModel(pixelValues)
        let batchSize = samFeatures.dim(0)
        let samH = samFeatures.dim(1)
        let samW = samFeatures.dim(2)
        let samChannels = samFeatures.dim(3)
        let samFlat = samFeatures.reshaped(batchSize, samH * samW, samChannels)

        let clipFeatures = clipModel(pixelValues, patchEmbeds: samFeatures)
        let clipWithoutCls = clipFeatures[0..., 1..., 0...]
        return concatenated([clipWithoutCls, samFlat], axis: -1)
    }

    private func projectedImageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        projector(fusedVisionFeatures(pixelValues))
    }

    private func mergeInputIdsWithImageFeatures(inputIds: MLXArray, imageFeatures: MLXArray)
        -> MLXArray
    {
        let textEmbeddings = languageModel.embedTokens(inputIds)
        let imageMask = inputIds .== config.baseConfiguration.imageTokenId
        let seqLen = textEmbeddings.dim(1)
        let imageTokenCount = imageFeatures.dim(1)
        let firstImageIndex = argMax(imageMask.asType(.int32), axis: 1)[0].item(Int.self)
        let prePad = firstImageIndex
        let postPad = max(0, seqLen - firstImageIndex - imageTokenCount)

        var paddedImageFeatures = imageFeatures
        if prePad > 0 || postPad > 0 {
            paddedImageFeatures = padded(imageFeatures, widths: [0, .init((prePad, postPad)), 0])
        }
        if paddedImageFeatures.dim(1) > seqLen {
            paddedImageFeatures = paddedImageFeatures[0..., ..<seqLen, 0...]
        }

        return which(imageMask[.ellipsis, .newAxis], paddedImageFeatures, textEmbeddings)
    }

    @_spi(Testing)
    public func samFeaturesForTesting(_ pixelValues: MLXArray) -> MLXArray {
        samModel(pixelValues)
    }

    @_spi(Testing)
    public func fusedVisionFeaturesForTesting(_ pixelValues: MLXArray) -> MLXArray {
        fusedVisionFeatures(pixelValues)
    }

    @_spi(Testing)
    public func projectedImageFeaturesForTesting(_ pixelValues: MLXArray) -> MLXArray {
        projectedImageFeatures(pixelValues)
    }

    @_spi(Testing)
    public func routeLayerForTesting(_ hiddenStates: MLXArray, layerIndex: Int) -> (
        MLXArray, MLXArray
    )? {
        guard layerIndex >= 0, layerIndex < languageModel.layers.count else { return nil }
        guard let sparseMoE = languageModel.layers[layerIndex].mlp as? SparseMoE else { return nil }
        return sparseMoE.gate(hiddenStates)
    }
}

private final class TextAttention: Module {
    let config: DeepseekOCRConfiguration.TextConfiguration
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        self.config = config
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)
        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: config.ropeTheta, scale: 1.0)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = qProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return oProj(output)
    }
}

private final class TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private final class MoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int?
    let scoringFunc: String

    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        let expertCount = config.nRoutedExperts ?? 1
        self.topK = config.numExpertsPerTok ?? 1
        self.normTopkProb = config.normTopkProb ?? false
        self.routedScalingFactor = config.routedScalingFactor ?? 1.0
        self.nGroup = config.nGroup ?? 1
        self.topkGroup = config.topkGroup
        // Match Python TextConfig default (`softmax`) used by Unlimited-OCR.
        self.scoringFunc = config.scoringFunc ?? "softmax"
        self._weight.wrappedValue = zeros([expertCount, config.hiddenSize])
        self._eScoreCorrectionBias.wrappedValue = zeros([expertCount])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let (batchSize, sequenceLength, _) = (x.dim(0), x.dim(1), x.dim(2))
        let gates = matmul(x.asType(.float32), weight.T.asType(.float32))
        var scores: MLXArray
        switch scoringFunc {
        case "sigmoid":
            scores = sigmoid(gates)
        default:
            // "softmax" (Python default) and any unrecognized value.
            scores = MLX.softmax(gates, axis: -1, precise: true)
        }
        let scoresForChoice = scores + eScoreCorrectionBias

        if nGroup > 1 {
            let expertCount = weight.dim(0)
            let expertsPerGroup = expertCount / nGroup
            var groupScores = scoresForChoice.reshaped(
                batchSize, sequenceLength, nGroup, expertsPerGroup)
            let topPerGroup = top(groupScores, k: min(2, expertsPerGroup), axis: -1)
                .sum(axis: -1, keepDims: true)
            let droppedGroups = max(0, nGroup - (topkGroup ?? 1))
            if droppedGroups > 0 {
                var maskedGroupIndices = argPartition(
                    topPerGroup, kth: droppedGroups - 1, axis: -2)[
                        .ellipsis, ..<droppedGroups, 0...
                    ]
                maskedGroupIndices = broadcast(
                    maskedGroupIndices,
                    to: [batchSize, sequenceLength, droppedGroups, expertsPerGroup])
                groupScores = putAlong(
                    groupScores, stopGradient(maskedGroupIndices), values: MLXArray(0.0), axis: -2)
                scores = flattened(groupScores, start: -2, end: -1)
            }
        }

        let indices = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var selected = takeAlong(scores, indices, axis: -1)
        if normTopkProb {
            selected = selected / (selected.sum(axis: -1, keepDims: true) + MLXArray(1e-20))
        }
        // Python MoEGate always scales selected scores by routed_scaling_factor.
        selected = selected * routedScalingFactor
        return (indices, selected.asType(x.dtype))
    }
}

private final class SparseMoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: TextMLP?

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        let expertCount = config.nRoutedExperts ?? 1
        let hiddenSize = config.hiddenSize
        let intermediate = config.moeIntermediateSize ?? config.intermediateSize
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: hiddenSize,
            hiddenDims: intermediate,
            numExperts: expertCount)
        // Default SwitchGLU uses silu — matches Python mlx_vlm SwitchGLU/SwiGLU.
        self._gate.wrappedValue = MoEGate(config)
        if let shared = config.nSharedExperts {
            self._sharedExperts.wrappedValue = TextMLP(
                hiddenSize: hiddenSize,
                intermediateSize: intermediate * shared)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = weightedExpertSum(y, scores)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

private final class TextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: TextAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let mlp: UnaryLayer

    init(_ config: DeepseekOCRConfiguration.TextConfiguration, layerIndex: Int) {
        self._selfAttention.wrappedValue = TextAttention(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps)

        let useMoE =
            (config.nRoutedExperts ?? 0) > 0
            && layerIndex >= (config.firstKDenseReplace ?? 0)
            && layerIndex % max(config.moeLayerFreq ?? 1, 1) == 0
        if useMoE {
            self.mlp = SparseMoE(config)
        } else {
            self.mlp = TextMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attentionOut = selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
        let hidden = x + attentionOut
        return hidden + mlp(postAttentionLayerNorm(hidden))
    }
}

private final class TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [TextDecoderLayer]
    let norm: RMSNorm
    let kvHeads: [Int]

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            TextDecoderLayer(config, layerIndex: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
    }

    func callAsFunction(_ inputIds: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(embedTokens(inputIds), cache: cache)
    }

    func forward(_ hidden: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = hidden
        let mask = createAttentionMask(h: hidden, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }
        return norm(hidden)
    }
}

private final class VisionMLP: Module {
    @ModuleInfo(key: "lin1") var lin1: Linear
    @ModuleInfo(key: "lin2") var lin2: Linear

    init(hiddenSize: Int, mlpDim: Int) {
        self._lin1.wrappedValue = Linear(hiddenSize, mlpDim)
        self._lin2.wrappedValue = Linear(mlpDim, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lin2(gelu(lin1(x)))
    }
}

private final class VisionAttention: Module {
    let numHeads: Int
    let scale: Float
    let windowSize: Int
    let useRelPos: Bool

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    var relPosH: MLXArray?
    var relPosW: MLXArray?

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration, windowSize: Int) {
        self.numHeads = config.numAttentionHeads
        self.scale = pow(Float(config.hiddenSize / config.numAttentionHeads), -0.5)
        self.windowSize = windowSize
        self.useRelPos = config.useRelPos
        self._qkv.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize * 3,
            bias: config.qkvBias)
        self._proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        if useRelPos {
            let inputSize = windowSize > 0 ? windowSize : (config.imageSize / config.patchSize)
            let headDim = config.hiddenSize / config.numAttentionHeads
            self.relPosH = zeros([2 * inputSize - 1, headDim])
            self.relPosW = zeros([2 * inputSize - 1, headDim])
        }
    }

    private func getRelPos(qSize: Int, kSize: Int, relPos: MLXArray) -> MLXArray {
        let maxRelDist = 2 * max(qSize, kSize) - 1
        var relPosResized = relPos
        if relPos.dim(0) != maxRelDist {
            let scale = Float(maxRelDist) / Float(relPos.dim(0))
            relPosResized = Upsample(scaleFactor: [scale], mode: .linear(alignCorners: false))(
                relPos.expandedDimensions(axis: 0)
            ).squeezed(axis: 0)
        }
        let qCoords = MLXArray(
            (0 ..< qSize).map { Float($0) * max(Float(kSize) / Float(qSize), 1.0) })
        let kCoords = MLXArray(
            (0 ..< kSize).map { Float($0) * max(Float(qSize) / Float(kSize), 1.0) })
        let offset = Float(kSize - 1) * max(Float(qSize) / Float(kSize), 1.0)
        return relPosResized[
            (qCoords.reshaped(qSize, 1) - kCoords.reshaped(1, kSize) + offset).asType(.int32)]
    }

    private func relativeBias(query: MLXArray, height: Int, width: Int) -> MLXArray? {
        guard useRelPos, let relPosH, let relPosW else { return nil }
        let batchHeads = query.dim(0)
        let dim = query.dim(-1)
        let q = query.reshaped(batchHeads, height, width, dim)
        let relH = getRelPos(qSize: height, kSize: height, relPos: relPosH)
        let relW = getRelPos(qSize: width, kSize: width, relPos: relPosW)
        let relHBias = matmul(q, relH.swappedAxes(-2, -1)).reshaped(
            batchHeads, height * width, height, 1)
        let relWBias = matmul(
            q.transposed(0, 2, 1, 3),
            relW.swappedAxes(-2, -1)
        ).transposed(0, 2, 1, 3).reshaped(batchHeads, height * width, 1, width)
        return (relHBias + relWBias).reshaped(-1, height * width, height * width)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let (batch, height, width, _) = (
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            hiddenStates.dim(2),
            hiddenStates.dim(3)
        )
        let headDim = hiddenStates.dim(3) / numHeads
        let qkvOut = qkv(hiddenStates)
            .reshaped(batch, height * width, 3, numHeads, headDim)
            .transposed(2, 0, 3, 1, 4)
        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let bias = relativeBias(
            query: q.reshaped(batch * numHeads, height * width, headDim), height: height,
            width: width)
        {
            mask = .array(bias.reshaped(batch, numHeads, height * width, height * width))
        } else {
            mask = .none
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )
        return proj(
            output.reshaped(batch, numHeads, height, width, headDim).transposed(0, 2, 3, 1, 4)
                .reshaped(batch, height, width, -1))
    }
}

private final class VisionLayer: Module {
    @ModuleInfo(key: "norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "attn") var attention: VisionAttention
    @ModuleInfo(key: "norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: VisionMLP

    let windowSize: Int

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration, windowSize: Int) {
        self.windowSize = windowSize
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._attention.wrappedValue = VisionAttention(config, windowSize: windowSize)
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = VisionMLP(hiddenSize: config.hiddenSize, mlpDim: config.mlpDim)
    }

    private func partition(_ hiddenStates: MLXArray) -> (MLXArray, (Int, Int), Int) {
        let (batch, height, width, channels) = (
            hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2), hiddenStates.dim(3)
        )
        let padH = (windowSize - height % windowSize) % windowSize
        let padW = (windowSize - width % windowSize) % windowSize
        var paddedStates = hiddenStates
        if padH > 0 || padW > 0 {
            paddedStates = padded(
                hiddenStates,
                widths: [.init((0, 0)), .init((0, padH)), .init((0, padW)), .init((0, 0))])
        }
        let paddedHeight = height + padH
        let paddedWidth = width + padW
        let windows =
            paddedStates
            .reshaped(
                batch, paddedHeight / windowSize, windowSize, paddedWidth / windowSize, windowSize,
                channels
            )
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(-1, windowSize, windowSize, channels)
        return (windows, (paddedHeight, paddedWidth), batch)
    }

    private func unpartition(
        _ windows: MLXArray,
        padding: (Int, Int),
        original: (Int, Int),
        batchSize: Int
    ) -> MLXArray {
        let channels = windows.dim(-1)
        var merged =
            windows
            .reshaped(
                batchSize, padding.0 / windowSize, padding.1 / windowSize, windowSize, windowSize,
                channels
            )
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batchSize, padding.0, padding.1, channels)
        if padding != original {
            merged = merged[0..., ..<original.0, ..<original.1, 0...]
        }
        return merged
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let residual = hiddenStates
        var hidden = layerNorm1(hiddenStates)
        let original = (hidden.dim(1), hidden.dim(2))
        var padding: (Int, Int)?
        var batchSize: Int?
        if windowSize > 0 {
            let partitioned = partition(hidden)
            hidden = partitioned.0
            padding = partitioned.1
            batchSize = partitioned.2
        }
        hidden = attention(hidden)
        if let padding, let batchSize {
            hidden = unpartition(hidden, padding: padding, original: original, batchSize: batchSize)
        }
        hidden = residual + hidden
        return hidden + mlp(layerNorm2(hidden))
    }
}

private final class PatchEmbeddings: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self._proj.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize))
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        proj(pixelValues)
    }
}

private final class VisionNeck: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self._conv1.wrappedValue = Conv2d(
            inputChannels: config.hiddenSize,
            outputChannels: config.outputChannels,
            kernelSize: IntOrPair(1),
            bias: false)
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.outputChannels, eps: config.layerNormEps)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: config.outputChannels,
            outputChannels: config.outputChannels,
            kernelSize: IntOrPair(3),
            padding: IntOrPair(1),
            bias: false)
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.outputChannels, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layerNorm2(conv2(layerNorm1(conv1(x))))
    }
}

private final class VisionEncoder: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbeddings
    @ModuleInfo(key: "layers") var layers: [VisionLayer]
    @ModuleInfo(key: "neck") var neck: VisionNeck
    @ModuleInfo(key: "net_2") var net2: Conv2d
    @ModuleInfo(key: "net_3") var net3: Conv2d

    var posEmbed: MLXArray?
    let patchCount: Int

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self.patchCount = config.imageSize / config.patchSize
        self._patchEmbed.wrappedValue = PatchEmbeddings(config)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { index in
            VisionLayer(
                config,
                windowSize: config.globalAttentionIndexes.contains(index) ? 0 : config.windowSize)
        }
        self._neck.wrappedValue = VisionNeck(config)
        self._net2.wrappedValue = Conv2d(
            inputChannels: config.outputChannels,
            outputChannels: 512,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1),
            bias: false)
        self._net3.wrappedValue = Conv2d(
            inputChannels: 512,
            outputChannels: 1024,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1),
            bias: false)
        if config.useAbsPos {
            self.posEmbed = zeros([1, patchCount, patchCount, config.hiddenSize])
        }
    }

    private func interpolatedPositionalEmbedding(_ posEmbed: MLXArray, height: Int, width: Int)
        -> MLXArray
    {
        if posEmbed.dim(1) == height && posEmbed.dim(2) == width {
            return posEmbed
        }
        let scaleH = Float(height) / Float(posEmbed.dim(1))
        let scaleW = Float(width) / Float(posEmbed.dim(2))
        return Upsample(scaleFactor: [scaleH, scaleW], mode: .cubic(alignCorners: false))(posEmbed)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var hidden = patchEmbed(pixelValues)
        if let posEmbed {
            hidden =
                hidden
                + interpolatedPositionalEmbedding(
                    posEmbed, height: hidden.dim(1), width: hidden.dim(2))
        }
        for layer in layers {
            hidden = layer(hidden)
        }
        hidden = neck(hidden)
        hidden = net2(hidden)
        hidden = net3(hidden)
        return hidden
    }
}

private struct ClipVisionConfig {
    let hiddenSize = 1024
    let numLayers = 24
    let numAttentionHeads = 16
    let ffnHiddenSize = 4096
    let imageSize = 224
    let patchSize = 14
    let layerNormEps: Float = 1e-5
}

private func quickGelu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(1.702 * x)
}

private func clippedSilu(_ x: MLXArray) -> MLXArray {
    clip(x * sigmoid(x), min: -100, max: 100)
}

private final class ClipVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d

    var classEmbedding: MLXArray
    var positionEmbedding: MLXArray
    let config = ClipVisionConfig()

    override init() {
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: 3,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize),
            bias: false)
        self.classEmbedding = MLXRandom.normal([config.hiddenSize])
        self.positionEmbedding = MLXRandom.normal([
            1,
            (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize) + 1,
            config.hiddenSize,
        ])
    }

    private func resizedPositionEmbedding(targetSize: Int) -> MLXArray {
        if positionEmbedding.dim(1) == targetSize {
            return positionEmbedding
        }
        let clsToken = positionEmbedding[0..., ..<1, 0...]
        let patchEmbedding = positionEmbedding[0..., 1..., 0...]
        let sourceSize = Int(sqrt(Double(positionEmbedding.dim(1) - 1)))
        let destinationSize = Int(sqrt(Double(targetSize - 1)))
        if sourceSize == destinationSize {
            return positionEmbedding
        }
        let resized = Upsample(
            scaleFactor: [
                Float(destinationSize) / Float(sourceSize),
                Float(destinationSize) / Float(sourceSize),
            ],
            mode: .cubic(alignCorners: false)
        )(patchEmbedding.reshaped(1, sourceSize, sourceSize, config.hiddenSize))
        return concatenated(
            [clsToken, resized.reshaped(1, destinationSize * destinationSize, config.hiddenSize)],
            axis: 1)
    }

    func callAsFunction(_ pixelValues: MLXArray, patchEmbeds: MLXArray?) -> MLXArray {
        let batchSize = pixelValues.dim(0)
        let patches: MLXArray
        if let patchEmbeds {
            patches = patchEmbeds.reshaped(
                batchSize, patchEmbeds.dim(1) * patchEmbeds.dim(2), patchEmbeds.dim(3))
        } else {
            let projected = patchEmbedding(pixelValues)
            patches = projected.reshaped(
                batchSize, projected.dim(1) * projected.dim(2), projected.dim(3))
        }
        let cls = broadcast(
            classEmbedding[.newAxis, .newAxis, 0...], to: [batchSize, 1, config.hiddenSize])
        let embeddings = concatenated([cls, patches], axis: 1)
        return embeddings + resizedPositionEmbedding(targetSize: embeddings.dim(1))
    }
}

private final class ClipFeedForward: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    override init() {
        let config = ClipVisionConfig()
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.ffnHiddenSize)
        self._fc2.wrappedValue = Linear(config.ffnHiddenSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // mlx-vlm deepseekocr Vision MLP uses nn.GELU (not OpenAI QuickGELU).
        fc2(gelu(fc1(x)))
    }
}

private final class ClipAttention: Module {
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let config = ClipVisionConfig()
    let scale: Float

    override init() {
        self.scale = pow(Float(config.hiddenSize / config.numAttentionHeads), -0.5)
        self._qkvProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3)
        self._outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let headDim = config.hiddenSize / config.numAttentionHeads
        let qkv = qkvProj(x).reshaped(batch, length, 3, config.numAttentionHeads, headDim)
        let q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        let k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none)
        return outProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

private final class ClipTransformerBlock: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: ClipAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: ClipFeedForward

    override init() {
        let config = ClipVisionConfig()
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._selfAttention.wrappedValue = ClipAttention()
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = ClipFeedForward()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = x + selfAttention(layerNorm1(x))
        return hidden + mlp(layerNorm2(hidden))
    }
}

private final class ClipVisionTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [ClipTransformerBlock]

    override init() {
        self._layers.wrappedValue = (0 ..< ClipVisionConfig().numLayers).map { _ in
            ClipTransformerBlock()
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers.reduce(x) { partialResult, layer in layer(partialResult) }
    }
}

private final class ClipVisionModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: ClipVisionEmbeddings
    @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm
    @ModuleInfo(key: "transformer") var transformer: ClipVisionTransformer

    override init() {
        let config = ClipVisionConfig()
        self._embeddings.wrappedValue = ClipVisionEmbeddings()
        self._preLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._transformer.wrappedValue = ClipVisionTransformer()
    }

    func callAsFunction(_ pixelValues: MLXArray, patchEmbeds: MLXArray?) -> MLXArray {
        transformer(preLayerNorm(embeddings(pixelValues, patchEmbeds: patchEmbeds)))
    }
}

private final class MultiModalProjector: Module {
    @ModuleInfo(key: "layers") var layers: Linear

    init(_ config: DeepseekOCRConfiguration) {
        self._layers.wrappedValue = Linear(2048, config.textConfiguration.hiddenSize)
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        layers(features)
    }
}
