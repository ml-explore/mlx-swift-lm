// Copyright © 2025 Apple Inc.
//
// Based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4/processing_gemma4.py

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Processor Configuration

public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let patchSize: Int
    public let maxSoftTokens: Int
    public let poolingKernelSize: Int
    public let doNormalize: Bool
    public let doRescale: Bool
    public let doResize: Bool

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case patchSize = "patch_size"
        case maxSoftTokens = "max_soft_tokens"
        case poolingKernelSize = "pooling_kernel_size"
        case doNormalize = "do_normalize"
        case doRescale = "do_rescale"
        case doResize = "do_resize"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass) ?? "Gemma4Processor"
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        maxSoftTokens = try c.decodeIfPresent(Int.self, forKey: .maxSoftTokens) ?? 280
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        doNormalize = try c.decodeIfPresent(Bool.self, forKey: .doNormalize) ?? false
        doRescale = try c.decodeIfPresent(Bool.self, forKey: .doRescale) ?? true
        doResize = try c.decodeIfPresent(Bool.self, forKey: .doResize) ?? true
    }

    /// Max patches allowed (= maxSoftTokens × poolingKernelSize²).
    var maxPatches: Int { maxSoftTokens * poolingKernelSize * poolingKernelSize }

    /// Side multiplier (must divide target H and W).
    var sideMult: Int { poolingKernelSize * patchSize }
}

// MARK: - Gemma4 Processor

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    // MARK: Image Preprocessing

    /// Compute target dimensions for aspect-ratio-preserving resize.
    /// Target H and W are multiples of sideMult, with at most maxPatches total patches.
    private func targetSize(for image: CIImage) -> CGSize {
        let extent = image.extent
        let h = Float(extent.height)
        let w = Float(extent.width)

        let targetPx = Float(config.maxPatches) * Float(config.patchSize * config.patchSize)
        let factor = (targetPx / (h * w)).squareRoot()
        let sideMult = Float(config.sideMult)

        var targetH = Int(floor(factor * h / sideMult)) * config.sideMult
        var targetW = Int(floor(factor * w / sideMult)) * config.sideMult

        // Clamp zero dimensions
        if targetH == 0 && targetW == 0 {
            targetH = config.sideMult
            targetW = config.sideMult
        } else if targetH == 0 {
            targetH = config.sideMult
            let maxSide =
                (config.maxSoftTokens / (config.poolingKernelSize * config.poolingKernelSize))
                * config.sideMult
            targetW = min(Int(floor(w / h)) * config.sideMult, maxSide)
        } else if targetW == 0 {
            targetW = config.sideMult
            let maxSide =
                (config.maxSoftTokens / (config.poolingKernelSize * config.poolingKernelSize))
                * config.sideMult
            targetH = min(Int(floor(h / w)) * config.sideMult, maxSide)
        }

        return CGSize(width: targetW, height: targetH)
    }

    /// Compute number of soft tokens produced by an image of given pixel dimensions.
    func numSoftTokens(targetH: Int, targetW: Int) -> Int {
        let numPatches = (targetH / config.patchSize) * (targetW / config.patchSize)
        return numPatches / (config.poolingKernelSize * config.poolingKernelSize)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        guard let image = images.first else {
            throw VLMError.imageProcessingFailure("No images provided")
        }

        // Bring into sRGB linear space
        let srgb = MediaProcessing.inSRGBToneCurveSpace(image)

        // Aspect-ratio preserving resize
        let target = targetSize(for: srgb)
        let resized = MediaProcessing.resampleBicubic(srgb, to: target)

        // MediaProcessing.asMLXArray already returns [1, 3, H, W] (channel-first, RGB, batch dim).
        // Do NOT transpose — it's already in the correct format for the vision encoder.
        let bchw = MediaProcessing.asMLXArray(resized)  // [1, 3, H, W]

        let tH = Int(target.height)
        let tW = Int(target.width)
        // Sanity check: actual tensor dims must match target (shapes are evaluated lazily)
        assert(bchw.dim(0) == 1, "Expected batch=1, got \(bchw.dim(0))")
        assert(bchw.dim(1) == 3, "Expected 3 RGB channels, got \(bchw.dim(1))")
        assert(bchw.dim(2) == tH, "Expected H=\(tH), got \(bchw.dim(2))")
        assert(bchw.dim(3) == tW, "Expected W=\(tW), got \(bchw.dim(3))")
        return (bchw, THW(1, tH, tW))
    }

    // MARK: - Chat Formatting

    /// Gemma 4 escape token for strings in tool definitions.
    private static let escapeToken = "<|\"|>"

    /// Format a single tool into Gemma4 declaration format:
    /// `<|tool>declaration:name{description:...parameters:{...}}<tool|>`
    private func formatTool(_ tool: [String: any Sendable]) -> String {
        guard let function = tool["function"] as? [String: any Sendable],
            let name = function["name"] as? String
        else {
            return ""
        }

        var result = "<|tool>"

        // Declaration and description
        result += "declaration:\(name){"
        if let description = function["description"] as? String {
            result += "description:\(Self.escapeToken)\(description)\(Self.escapeToken)"
        }

        // Parameters
        if let parameters = function["parameters"] as? [String: any Sendable] {
            result += ",parameters:{"

            // Properties
            if let properties = parameters["properties"] as? [String: any Sendable] {
                result += "properties:{"
                var firstProp = true
                for (propName, propValue) in properties {
                    if !firstProp { result += "," }
                    firstProp = false
                    result += "\(propName):{"
                    if let propDict = propValue as? [String: any Sendable] {
                        if let type = propDict["type"] as? String {
                            result +=
                                "type:\(Self.escapeToken)\(type.uppercased())\(Self.escapeToken)"
                        }
                    }
                    result += "}"
                }
                result += "},"
            }

            // Required
            if let required = parameters["required"] as? [String] {
                result += "required:["
                for (i, req) in required.enumerated() {
                    if i > 0 { result += "," }
                    result += "\(Self.escapeToken)\(req)\(Self.escapeToken)"
                }
                result += "],"
            }

            result += "}"
        }

        result += "}<tool|>"
        return result
    }

    /// Format tools into the Gemma4 system block format.
    private func formatTools(_ tools: [[String: any Sendable]]?) -> String {
        guard let tools = tools, !tools.isEmpty else { return "" }
        var result = ""
        for tool in tools {
            result += formatTool(tool)
        }
        return result
    }

    /// Format plain-text messages (no images) into the Gemma4 turn-based format:
    /// `<bos><|turn>role\ncontent<turn|>\n...<|turn>model\n`
    ///
    /// This avoids the Jinja chat template, which uses Jinja2 features
    /// (namespace, dictsort) that the Swift Jinja library does not support.
    private func formatPrompt(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?)
        -> String
    {
        var result = "<bos>"

        // System/developer block with tools (if provided)
        let hasSystemMessage = messages.contains { $0["role"] as? String == "system" }
        let toolsBlock = formatTools(tools)

        if hasSystemMessage || !toolsBlock.isEmpty {
            result += "<|turn>system\n"
            if let systemMsg = messages.first(where: { $0["role"] as? String == "system" }),
                let content = systemMsg["content"] as? String
            {
                result += content
            }
            if !toolsBlock.isEmpty {
                result += toolsBlock
            }
            result += "<turn|>\n"
        }

        // Regular messages
        for message in messages {
            guard let role = message["role"] as? String,
                let content = message["content"] as? String
            else { continue }
            if role == "system" { continue }  // Already handled above
            // Map "assistant" → "model" per Gemma4 convention
            let gemmaRole = role == "assistant" ? "model" : role
            result += "<|turn>\(gemmaRole)\n\(content)<turn|>\n"
        }

        // Append the start-of-model-turn prompt
        result += "<|turn>model\n"
        return result
    }

    // MARK: Prepare

    public func prepare(input: UserInput) async throws -> LMInput {
        // Build token array. For chat prompts with images we inject boi_token_id (255999)
        // directly, because the tokenizer vocabulary uses "<|image>" (not "<start_of_image>")
        // and encoding "<start_of_image>" as text splits it into 7 subword tokens.
        var promptTokens: [Int]
        if case .chat(let chatMessages) = input.prompt, !input.images.isEmpty {
            // Look up the actual token string for boi_token_id so we encode it correctly.
            let boiTokenId = 255999
            let boiStr = tokenizer.convertIdToToken(boiTokenId) ?? "<start_of_image>"

            // Build the formatted string inserting N boi tokens before each message's content.
            var result = "<bos>"

            // System block with tools (if provided)
            let toolsBlock = formatTools(input.tools)
            if !toolsBlock.isEmpty {
                result += "<|turn>system\n\(toolsBlock)<turn|>\n"
            }

            for message in chatMessages {
                let gemmaRole = message.role == .assistant ? "model" : message.role.rawValue
                let imagePrefixes = String(repeating: boiStr, count: message.images.count)
                result += "<|turn>\(gemmaRole)\n\(imagePrefixes)\(message.content)<turn|>\n"
            }
            result += "<|turn>model\n"
            promptTokens = tokenizer.encode(text: result, addSpecialTokens: false)
        } else {
            let messages = DefaultMessageGenerator().generate(from: input)
            let formatted = formatPrompt(messages: messages, tools: input.tools)
            promptTokens = tokenizer.encode(text: formatted, addSpecialTokens: false)
        }

        var processedImage: LMInput.ProcessedImage?

        if !input.images.isEmpty {
            let boiTokenId = 255999
            let imageTokenId = 258880

            // Process each image and expand boi tokens
            var expandedTokens: [Int] = []
            var allPixels: [MLXArray] = []
            var allFrames: [THW] = []

            // Process all images first to get their soft token counts
            var imageInfos: [(pixels: MLXArray, nSoftTokens: Int, frame: THW)] = []
            for userImage in input.images {
                let ciImage = try userImage.asCIImage()
                let (pixels, frame) = try preprocess(
                    images: [ciImage], processing: input.processing)
                let tH = Int(pixels.shape[2])
                let tW = Int(pixels.shape[3])
                let n = numSoftTokens(targetH: tH, targetW: tW)
                imageInfos.append((pixels: pixels, nSoftTokens: n, frame: frame))
            }

            // Expand tokens: replace each boi_token_id with n × image_token_id
            var imageIdx = 0
            for token in promptTokens {
                if token == boiTokenId && imageIdx < imageInfos.count {
                    let n = imageInfos[imageIdx].nSoftTokens
                    expandedTokens.append(contentsOf: Array(repeating: imageTokenId, count: n))
                    allPixels.append(imageInfos[imageIdx].pixels)
                    allFrames.append(imageInfos[imageIdx].frame)
                    imageIdx += 1
                } else {
                    expandedTokens.append(token)
                }
            }

            promptTokens = expandedTokens
            if !allPixels.isEmpty {
                let concatenatedPixels = concatenated(allPixels, axis: 0)
                processedImage = LMInput.ProcessedImage(
                    pixels: concatenatedPixels, frames: allFrames)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage
        )
    }
}
