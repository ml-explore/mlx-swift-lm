// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@_spi(Testing) @testable import MLXVLM

final class DeepseekOCRProcessorTests: XCTestCase {

    func testMessageGeneratorUsesCheckpointCompatibleStringContent() {
        let input = UserInput(
            chat: [
                .system("You are an OCR assistant."),
                .user(
                    "Parse both pages.",
                    images: [
                        .ciImage(makeSolidImage(width: 8, height: 8, color: .red)),
                        .ciImage(makeSolidImage(width: 8, height: 8, color: .blue)),
                    ]),
            ])

        let messages = DeepseekOCRMessageGenerator().generate(from: input)

        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[0]["content"] as? String, "You are an OCR assistant.")
        XCTAssertEqual(messages[1]["content"] as? String, "<image><image>Parse both pages.")
        XCTAssertNil(messages[1]["content"] as? [[String: any Sendable]])
    }

    func testMessageGeneratorPreservesExplicitImagePlaceholders() {
        let input = UserInput(
            prompt: .text("before<image>after"),
            images: [.ciImage(makeSolidImage(width: 8, height: 8, color: .red))])

        let messages = DeepseekOCRMessageGenerator().generate(from: input)

        XCTAssertEqual(messages[0]["content"] as? String, "before<image>after")
    }

    func testGundamModeMatchesExpectedCropMetadataAndTokenMask() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "Describe this page.",
            images: [.ciImage(makeSolidImage(width: 800, height: 400, color: .red))])

        let prepared = try await processor.internalPrepare(input: input)

        XCTAssertEqual(prepared.mode, .gundam)
        XCTAssertEqual(prepared.pixelValues.shape, [1, 3, 1024, 1024])
        XCTAssertEqual(prepared.localCrops.shape, [2, 3, 640, 640])
        XCTAssertEqual(prepared.imagesSpatialCrop.map { [$0.w, $0.h] }, [[2, 1]])
        XCTAssertEqual(prepared.imagesSeqMask.asType(.int32).sum().item(Int.self), 483)
        // Chat-template tokens with its image placeholder expanded to the lattice.
        XCTAssertEqual(prepared.inputIds.shape[0], 1)
        XCTAssertEqual(prepared.inputIds[0, 0].item(Int.self), 0)
        XCTAssertEqual(prepared.imagesSeqMask[0, 0].item(Bool.self), false)
        let firstImageTokenIndex = try XCTUnwrap(
            (0 ..< prepared.inputIds.shape[1]).first {
                prepared.imagesSeqMask[0, $0].item(Bool.self)
            })
        XCTAssertEqual(prepared.inputIds[0, firstImageTokenIndex].item(Int.self), 999)
        XCTAssertEqual(firstImageTokenIndex, 1)
        let lastImageTokenIndex = try XCTUnwrap(
            (0 ..< prepared.inputIds.shape[1]).last {
                prepared.imagesSeqMask[0, $0].item(Bool.self)
            })
        XCTAssertNotEqual(prepared.inputIds[0, lastImageTokenIndex + 1].item(Int.self), 999)
        XCTAssertEqual(prepared.imagesSeqMask[0, lastImageTokenIndex + 1].item(Bool.self), false)
        XCTAssertGreaterThanOrEqual(prepared.inputIds.shape[1], 484)

        let globalPixels = prepared.pixelValues.asType(.float32)
        let localPixels = prepared.localCrops.asType(.float32)
        XCTAssertLessThan(globalPixels.sum().item(Float.self), -100_000)
        XCTAssertLessThan(localPixels.sum().item(Float.self), -500_000)
    }

    func testBaseModeIsSelectableAndUsesSingleViewTokenGrid() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "Describe this page.",
            images: [.ciImage(makeSolidImage(width: 800, height: 400, color: .red))],
            additionalContext: DeepseekOCRProcessor.modeContext(.base))

        let prepared = try await processor.internalPrepare(input: input)

        XCTAssertEqual(prepared.mode, .base)
        XCTAssertEqual(prepared.pixelValues.shape, [1, 3, 640, 640])
        XCTAssertEqual(prepared.imagesSpatialCrop.map { [$0.w, $0.h] }, [[1, 1]])
        XCTAssertEqual(prepared.localCrops.shape, [1, 3, 1024, 1024])
        XCTAssertEqual(prepared.imagesSeqMask.asType(.int32).sum().item(Int.self), 111)
        XCTAssertEqual(prepared.inputIds.shape[0], 1)
        XCTAssertEqual(prepared.inputIds[0, 0].item(Int.self), 0)
        XCTAssertGreaterThanOrEqual(prepared.inputIds.shape[1], 112)
    }

    func testPrepareBaseModeOmitsLocalCropsFromLMInput() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "document parsing. ",
            images: [.ciImage(makeSolidImage(width: 800, height: 400, color: .red))],
            additionalContext: DeepseekOCRProcessor.modeContext(.base))

        let lmInput = try await processor.prepare(input: input)

        XCTAssertEqual(lmInput.image?.pixels.shape, [1, 3, 640, 640])
        XCTAssertEqual(lmInput.image?.frames?.map { [$0.t, $0.h, $0.w] }, [[1, 1, 1]])
        XCTAssertNil(lmInput.image?.positionIds)
        XCTAssertNil(lmInput.video, "base mode must not pack gundam local crops into video")
    }

    func testPrepareGundamModePacksLocalCropsIntoVideo() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "document parsing. ",
            images: [.ciImage(makeSolidImage(width: 800, height: 400, color: .red))])

        let lmInput = try await processor.prepare(input: input)

        XCTAssertEqual(lmInput.image?.pixels.shape, [1, 3, 1024, 1024])
        // One tile grid per page: (tiles, tilesHigh, tilesWide) for the 800×400 page.
        XCTAssertEqual(lmInput.image?.frames?.map { [$0.t, $0.h, $0.w] }, [[2, 1, 2]])
        XCTAssertNil(lmInput.image?.positionIds)
        XCTAssertEqual(lmInput.video?.pixels.shape, [2, 3, 640, 640])
    }

    func testPrepareCarriesOneTileGridPerPageInImageFrames() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "Multi page parsing.",
            images: [
                .ciImage(makeSolidImage(width: 800, height: 400, color: .red)),
                .ciImage(makeSolidImage(width: 400, height: 800, color: .blue)),
            ])

        let lmInput = try await processor.prepare(input: input)

        XCTAssertEqual(lmInput.image?.pixels.shape, [2, 3, 1024, 1024])
        XCTAssertEqual(
            lmInput.image?.frames?.map { [$0.t, $0.h, $0.w] }, [[2, 1, 2], [2, 2, 1]],
            "landscape page tiles 2 wide, portrait page tiles 2 high")
        XCTAssertEqual(
            lmInput.video?.pixels.shape, [4, 3, 640, 640],
            "local tiles of both pages are concatenated in page order")
    }

    /// The model reads each page's tile grid from `image.frames`: a hand-built LMInput
    /// with a 2×1 grid must send the two local tiles through the projector as one batch,
    /// after the global view. The projector probe records shapes without evaluating.
    func testPrefillDerivesLocalTileBatchFromImageFrames() throws {
        let model = try makeModel()
        let probe = ProbeLinear(2048, 32)
        model.update(modules: ModuleChildren.unflattened([("projector.layers", probe as Module)]))

        let tokens = MLXArray([Int32(0), Int32(20), Int32(21)]).reshaped(1, 3)
        let lmInput = LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
            image: .init(pixels: zeros([1, 3, 1024, 1024]), frames: [THW(2, 1, 2)]),
            video: .init(pixels: zeros([2, 3, 640, 640])))

        let result = try model.prepare(
            lmInput, cache: try model.newCache(parameters: nil), state: nil, prefill: .init())
        guard case .logits = result else {
            return XCTFail("expected prefill logits, got \(result)")
        }

        // 1024² global view → 256 fused tokens; the 640² tiles → 100 tokens each.
        XCTAssertEqual(probe.inputShapes, [[1, 256, 2048], [2, 100, 2048]])
    }

    /// Python `get_input_embeddings` assigns `input_embeds[idx, image_indices] = features`:
    /// the k-th image token takes feature k even when text separates the page lattices
    /// (one `<image>` placeholder per page), which a first-index pad would misalign.
    func testMergeAssignsImageFeaturesToImageTokenPositionsInOrder() throws {
        let model = try makeModel(configJSON: Self.mergeModelConfigJSON)
        let image = 999
        let ids = [5, image, image, image, 6, image, image, 7]
        let inputIds = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
        let hiddenSize = 32
        let features = stacked((0 ..< 5).map { MLXArray.ones([hiddenSize]) * Float($0 + 1) })[
            .newAxis]

        let merged = model.mergeInputIdsWithImageFeatures(
            inputIds: inputIds, imageFeatures: features)

        let embedding = try XCTUnwrap(
            model.parameters().flattened().first { $0.0 == "model.embed_tokens.weight" }?.1)
        XCTAssertEqual(merged.shape, [1, ids.count, hiddenSize])
        var featureIndex = 0
        for (position, token) in ids.enumerated() {
            let expected: MLXArray
            if token == image {
                expected = features[0, featureIndex]
                featureIndex += 1
            } else {
                expected = embedding[token]
            }
            XCTAssertTrue(
                allClose(merged[0, position], expected).item(Bool.self),
                "position \(position) (token \(token)) must carry feature/embedding")
        }
    }

    func testPrepareTextOnlyReturnsChatTemplateTokensWithoutAnImage() async throws {
        let processor = try makeProcessor()
        let input = UserInput(prompt: "document parsing. ")

        let lmInput = try await processor.prepare(input: input)

        XCTAssertNil(lmInput.image, "a text-only prompt must not fabricate an image")
        XCTAssertNil(lmInput.video)
        // DeterministicTokenizer: BOS, then one synthetic id per whitespace-delimited word.
        XCTAssertEqual(lmInput.text.tokens.shape, [1, 3])
        XCTAssertEqual(lmInput.text.tokens.asArray(Int32.self), [0, 20, 21])
        XCTAssertEqual(lmInput.text.mask?.asArray(Int8.self), [1, 1, 1])
    }

    func testInternalPrepareRequiresAnImage() async throws {
        let processor = try makeProcessor()

        do {
            _ = try await processor.internalPrepare(input: UserInput(prompt: "document parsing. "))
            XCTFail("expected VLMError.imageRequired for a text-only prompt")
        } catch VLMError.imageRequired {
            // expected
        }
    }

    /// End to end: a text-only prompt prefilled through the model (R-SWA ring caches, as on
    /// Unlimited-OCR packs) must never reach the projector, and its logits must be the plain
    /// text forward. The probe replaces the projector's `Linear` and counts calls.
    func testTextOnlyPrefillSkipsVisionTower() async throws {
        let processor = try makeProcessor()
        let model = try makeModel()
        let probe = ProbeLinear(2048, 32)
        model.update(modules: ModuleChildren.unflattened([("projector.layers", probe as Module)]))

        let lmInput = try await processor.prepare(input: UserInput(prompt: "document parsing. "))
        let cache = try model.newCache(parameters: nil)
        XCTAssertTrue(cache.allSatisfy { $0 is RingSlidingKVCache })

        let result = try model.prepare(lmInput, cache: cache, state: nil, prefill: .init())
        guard case .logits(let output) = result else {
            return XCTFail("expected prefill logits, got \(result)")
        }
        eval(output.logits)

        XCTAssertEqual(probe.callCount, 0, "text-only prefill must not run the vision tower")
        XCTAssertEqual(output.logits.shape, [1, 3, 32])
        XCTAssertEqual(cache.first?.offset, 3)

        let direct = model(
            LMInput.Text(tokens: lmInput.text.tokens),
            cache: try model.newCache(parameters: nil), state: nil)
        XCTAssertTrue(allClose(output.logits, direct.logits).item(Bool.self))
    }

    func testModeContextHelpers() {
        XCTAssertEqual(
            DeepseekOCRProcessor.mode(from: DeepseekOCRProcessor.modeContext(.base)), .base)
        XCTAssertEqual(
            DeepseekOCRProcessor.mode(from: DeepseekOCRProcessor.modeContext(.gundam)), .gundam)
        XCTAssertEqual(DeepseekOCRProcessor.mode(from: nil), .gundam)
        XCTAssertEqual(
            DeepseekOCRProcessor.mode(from: [DeepseekOCRProcessor.modeContextKey: "nope"]),
            .gundam)
        XCTAssertEqual(
            Set(DeepseekOCRProcessor.Mode.allCases.map(\.rawValue)),
            Set(["gundam", "base"]))
        XCTAssertEqual(DeepseekOCRProcessor.maxNumTiles(from: nil), 9)
        XCTAssertEqual(
            DeepseekOCRProcessor.maxNumTiles(from: [
                DeepseekOCRProcessor.maxNumTilesContextKey: 32
            ]), 32)
        XCTAssertEqual(
            DeepseekOCRProcessor.mode(from: DeepseekOCRProcessor.unlimitedContext(.base)), .base)
        XCTAssertEqual(
            DeepseekOCRProcessor.maxNumTiles(from: DeepseekOCRProcessor.unlimitedContext(.base)),
            32)
    }

    func testMultipageFusedBasePrepareStacksPagesAndLattices() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "Multi page parsing.",
            images: [
                .ciImage(makeSolidImage(width: 800, height: 400, color: .red)),
                .ciImage(makeSolidImage(width: 640, height: 480, color: .blue)),
            ],
            additionalContext: DeepseekOCRProcessor.modeContext(.base))

        let prepared = try await processor.internalPrepare(input: input)

        XCTAssertEqual(prepared.mode, .base)
        // Unlimited multipage base = 1024² (273 image tokens / page), not single-page 640.
        XCTAssertEqual(prepared.pixelValues.shape, [2, 3, 1024, 1024])
        XCTAssertEqual(prepared.imagesSpatialCrop.map { [$0.w, $0.h] }, [[1, 1], [1, 1]])
        XCTAssertEqual(prepared.imagesSeqMask.asType(.int32).sum().item(Int.self), 546)
        XCTAssertEqual(prepared.inputIds[0, 0].item(Int.self), 0)
        XCTAssertEqual(prepared.imagesSeqMask[0, 0].item(Bool.self), false)
        XCTAssertGreaterThanOrEqual(prepared.inputIds.shape[1], 547)

        let lmInput = try await processor.prepare(input: input)
        XCTAssertEqual(lmInput.image?.pixels.shape, [2, 3, 1024, 1024])
        XCTAssertEqual(
            lmInput.image?.frames?.map { [$0.t, $0.h, $0.w] }, [[1, 1, 1], [1, 1, 1]])
        XCTAssertNil(lmInput.video, "multipage base must not pack local crops")
    }

    func testMultipageRejectsMismatchedImageTokenCount() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "<image>page a<image>page b<image>extra",
            images: [
                .ciImage(makeSolidImage(width: 200, height: 200, color: .red)),
                .ciImage(makeSolidImage(width: 200, height: 200, color: .blue)),
            ],
            additionalContext: DeepseekOCRProcessor.modeContext(.base))

        do {
            _ = try await processor.internalPrepare(input: input)
            XCTFail("expected VLMError.singleImageAllowed for mismatched <image> count")
        } catch VLMError.singleImageAllowed {
            // expected
        }
    }

    func testGroundingSpecialTokenStringsAndDefaultIds() {
        XCTAssertEqual(
            Set(DeepseekOCRSpecialTokens.allStrings),
            Set([
                "<|grounding|>", "<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>",
            ]))
        XCTAssertEqual(DeepseekOCRSpecialTokens.refOpen.defaultId, 128_816)
        XCTAssertEqual(DeepseekOCRSpecialTokens.refClose.defaultId, 128_817)
        XCTAssertEqual(DeepseekOCRSpecialTokens.detOpen.defaultId, 128_818)
        XCTAssertEqual(DeepseekOCRSpecialTokens.detClose.defaultId, 128_819)
        XCTAssertEqual(DeepseekOCRSpecialTokens.grounding.defaultId, 128_820)
    }

    func testGroundingTokensResolveViaTokenizerPath() {
        let tokenizer = DeterministicTokenizer()
        let ids = DeepseekOCRSpecialTokens.resolveIds(tokenizer: tokenizer)
        XCTAssertEqual(ids[.grounding], 128_820)
        XCTAssertEqual(ids[.refOpen], 128_816)
        XCTAssertEqual(ids[.refClose], 128_817)
        XCTAssertEqual(ids[.detOpen], 128_818)
        XCTAssertEqual(ids[.detClose], 128_819)
        // Tokenizer lookup wins over defaults when present.
        XCTAssertEqual(
            DeepseekOCRSpecialTokens.id(of: .grounding, tokenizer: tokenizer), 128_820)
    }

    func testGroundingPromptBuildersMatchPythonShapes() {
        XCTAssertEqual(
            DeepseekOCRSpecialTokens.groundingPrompt(),
            "<|grounding|>OCR this image.")
        XCTAssertEqual(
            DeepseekOCRSpecialTokens.groundingPrompt("OCR this image"),
            "<|grounding|>OCR this image.")
        XCTAssertEqual(
            DeepseekOCRSpecialTokens.groundingMarkdownPrompt(),
            "<|grounding|>Convert the document to markdown.")
        XCTAssertEqual(
            DeepseekOCRSpecialTokens.locatePrompt("Total assets"),
            "Locate <|ref|>Total assets<|/ref|> in the image.")
    }

    func testPreparePreservesGroundingPromptTokensInInputIds() async throws {
        let processor = try makeProcessor(tokenizer: DeterministicTokenizer())
        let prompt = DeepseekOCRSpecialTokens.groundingPrompt()
        let input = UserInput(
            prompt: prompt,
            images: [.ciImage(makeSolidImage(width: 200, height: 200, color: .blue))],
            additionalContext: DeepseekOCRProcessor.modeContext(.base))

        let prepared = try await processor.internalPrepare(input: input)
        let ids = prepared.inputIds.asArray(Int32.self).map(Int.init)
        // DeterministicTokenizer encodes known special tokens as single IDs.
        XCTAssertTrue(
            ids.contains(DeepseekOCRSpecialTokens.grounding.defaultId),
            "grounding token must survive prepare() encode path; ids=\(ids)")
    }

    func testParseDetectionsExtractsNormalizedBoxes() {
        let sample =
            #"<|/ref|><|det|>[[330, 198, 558, 230]]<|/det|>"#
            + "\n"
            + #"<|ref|>body<|/ref|><|det|>[[10,20,30,40]]<|/det|>"#
        let boxes = DeepseekOCRSpecialTokens.parseDetections(from: sample)
        XCTAssertEqual(
            boxes,
            [
                .init(x1: 330, y1: 198, x2: 558, y2: 230),
                .init(x1: 10, y1: 20, x2: 30, y2: 40),
            ])
    }

    private func makeProcessor(tokenizer: any Tokenizer = DeterministicTokenizer()) throws
        -> DeepseekOCRProcessor
    {
        let config = try JSONDecoder().decode(
            DeepseekOCRProcessorConfiguration.self,
            from: Data(Self.processorConfigJSON.utf8))
        return DeepseekOCRProcessor(config, tokenizer: tokenizer)
    }

    private func makeSolidImage(width: CGFloat, height: CGFloat, color: CIColor) -> CIImage {
        CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    private func makeModel(configJSON: String = DeepseekOCRProcessorTests.modelConfigJSON)
        throws -> DeepseekOCR
    {
        let config = try JSONDecoder().decode(
            DeepseekOCRConfiguration.self,
            from: Data(configJSON.utf8))
        return DeepseekOCR(config)
    }

    /// SAM-base vision geometry (so the tower accepts the processor's 1024² global view)
    /// over a one-layer language model, with an R-SWA window as on Unlimited-OCR packs.
    private static let modelConfigJSON = #"""
        {
         "model_type": "deepseekocr",
         "sliding_window_size": 8,
         "vision_config": {
          "hidden_size": 768,
          "output_channels": 256,
          "num_hidden_layers": 12,
          "num_attention_heads": 12,
          "image_size": 1024,
          "patch_size": 16,
          "global_attn_indexes": [2, 5, 8, 11],
          "mlp_dim": 3072
         },
         "language_config": {
          "vocab_size": 32,
          "hidden_size": 32,
          "intermediate_size": 64,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "max_position_embeddings": 32
         }
        }
        """#

    /// Same geometry with a vocabulary that contains the test tokenizer's `<image>` id.
    private static let mergeModelConfigJSON = #"""
        {
         "model_type": "deepseekocr",
         "image_token_id": 999,
         "vision_config": {
          "hidden_size": 768,
          "output_channels": 256,
          "num_hidden_layers": 12,
          "num_attention_heads": 12,
          "image_size": 1024,
          "patch_size": 16,
          "global_attn_indexes": [2, 5, 8, 11],
          "mlp_dim": 3072
         },
         "language_config": {
          "vocab_size": 1000,
          "hidden_size": 32,
          "intermediate_size": 64,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "max_position_embeddings": 32
         }
        }
        """#

    private static let processorConfigJSON = #"""
        {
         "candidate_resolutions": [[1024, 1024]],
         "downsample_ratio": 4,
         "image_mean": [0.5, 0.5, 0.5],
         "image_std": [0.5, 0.5, 0.5],
         "image_token": "<image>",
         "patch_size": 16,
         "size": {
          "shortest_edge": 1024,
          "longest_edge": 1024
         }
        }
        """#
}

/// A `Linear` that counts how often it is applied and records its input shapes.
private final class ProbeLinear: Linear {
    var callCount = 0
    var inputShapes = [[Int]]()

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        callCount += 1
        inputShapes.append(x.shape)
        return super.callAsFunction(x)
    }
}

private struct DeterministicTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        guard !text.isEmpty else { return [] }
        // Split on whitespace but keep DeepSeek special tokens as atomic pieces
        // (mirrors HF tokenizer single-id encoding for <|grounding|> / <|ref|> / …).
        var ids = [Int]()
        var remaining = text[...]
        while !remaining.isEmpty {
            if remaining.first?.isWhitespace == true {
                remaining = remaining.drop(while: \.isWhitespace)
                continue
            }
            if let match = DeepseekOCRSpecialTokens.allCases.first(where: {
                remaining.hasPrefix($0.rawValue)
            }) {
                ids.append(match.defaultId)
                remaining = remaining.dropFirst(match.rawValue.count)
                continue
            }
            if remaining.hasPrefix("<image>") {
                ids.append(999)
                remaining = remaining.dropFirst("<image>".count)
                continue
            }
            // Next whitespace-delimited word → synthetic id.
            let word = remaining.prefix(while: { !$0.isWhitespace && $0 != "<" })
            if word.isEmpty {
                // Unknown '<' fragment — consume one scalar.
                ids.append(20 + ids.count)
                remaining = remaining.dropFirst()
            } else {
                ids.append(20 + ids.count)
                remaining = remaining.dropFirst(word.count)
            }
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }

    func convertTokenToId(_ token: String) -> Int? {
        if let special = DeepseekOCRSpecialTokens(rawValue: token) {
            return special.defaultId
        }
        switch token {
        case "<image>": return 999
        case "<s>": return 0
        default: return nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { "<s>" }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var ids = [0]
        for message in messages {
            // The production DeepSeek/Unlimited template renders message.content directly.
            // It does not translate Qwen-style structured image parts into <image>.
            if let content = message["content"] as? String {
                ids.append(contentsOf: encode(text: content, addSpecialTokens: false))
            }
        }
        return ids
    }
}
