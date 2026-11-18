// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

/// With `use_bidirectional_attention: "vision"`, the soft tokens of one Gemma 4 image
/// attend to each other in both directions. Prefilling a prefix and then the rest of the
/// prompt on the warm cache must give the logits of one cold prefill.
///
/// Tiny random models; no weights are downloaded.
struct Gemma4WarmImagePrefillTests {

    private static let imageToken = 100
    private static let boiToken = 101
    private static let eoiToken = 102
    /// 8x8 pixels, patch size 4, pooling 1: four soft tokens per image.
    private static let softTokensPerImage = 4

    /// The bound of `Gemma4ChunkedPrefillTests` for text-only continuations in f16.
    private static let tolerance: Float = 5e-2

    // MARK: - Gemma4

    /// Prefix lengths below and above the sliding window, so the warm rotating cache wraps.
    @Test(arguments: [true, false], [5, 13])
    func warmImageSuffixMatchesColdPrefill(bidirectional: Bool, prefixLength: Int) throws {
        let model = try Self.makeGemma4(bidirectional: bidirectional)
        let prefix = Self.text(prefixLength, seed: 1)
        let tokens = prefix + Self.imageBlock() + Self.text(3, seed: 2)
        let image = LMInput.ProcessedImage(pixels: Self.pixels(seed: 0))

        let suffixCount = tokens.count - prefixLength
        let coldCache = try model.newCache(parameters: nil)
        let cold = try Self.suffixLogits(
            model.prepare(
                Self.input(tokens, image: image), cache: coldCache, state: nil,
                prefill: .init(stepSize: 512)), count: suffixCount)

        let warmCache = try model.newCache(parameters: nil)
        _ = try model.prepare(
            Self.input(prefix), cache: warmCache, state: nil, prefill: .init(stepSize: 512))
        let warm = try Self.suffixLogits(
            model.prepare(
                Self.input(Array(tokens[prefixLength...]), image: image), cache: warmCache,
                state: nil, prefill: .init(stepSize: 512)), count: suffixCount)

        #expect(warmCache.map(\.offset) == coldCache.map(\.offset))
        let drift = Self.maxDrift(warm, cold)
        #expect(drift < Self.tolerance, "warm and cold logits differ by \(drift)")
    }

    /// A second turn adds an image after an earlier image turn: the split keeps only the
    /// new image, and the warm prefill matches a cold prefill of the whole conversation.
    @Test(arguments: [true, false])
    func splitImageTurnMatchesColdPrefill(bidirectional: Bool) throws {
        let model = try Self.makeGemma4(bidirectional: bidirectional)
        let firstTurn = Self.text(4, seed: 3) + Self.imageBlock() + Self.text(9, seed: 4)
        let tokens = firstTurn + Self.text(3, seed: 5) + Self.imageBlock() + Self.text(3, seed: 6)
        let first = Self.pixels(seed: 0)
        let second = Self.pixels(seed: 1)
        let full = Self.input(
            tokens, image: LMInput.ProcessedImage(pixels: concatenated([first, second])))

        let suffixCount = tokens.count - firstTurn.count
        let cold = try Self.suffixLogits(
            model.prepare(
                full, cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(stepSize: 512)), count: suffixCount)

        let warmCache = try model.newCache(parameters: nil)
        _ = try model.prepare(
            Self.input(firstTurn, image: LMInput.ProcessedImage(pixels: first)),
            cache: warmCache, state: nil, prefill: .init(stepSize: 512))
        let suffix = try #require(
            model.splitPreparedInput(full, droppingFirst: firstTurn.count))
        #expect(suffix.text.tokens.asArray(Int.self) == Array(tokens[firstTurn.count...]))
        #expect(suffix.image?.pixels.dim(0) == 1)
        let warm = try Self.suffixLogits(
            model.prepare(suffix, cache: warmCache, state: nil, prefill: .init(stepSize: 512)),
            count: suffixCount)

        let drift = Self.maxDrift(warm, cold)
        #expect(drift < Self.tolerance, "warm and cold logits differ by \(drift)")
    }

    @Test func splitKeepsFramesOfTheSuffixImages() throws {
        let model = try Self.makeGemma4(bidirectional: true)
        let firstTurn = Self.text(2, seed: 7) + Self.imageBlock()
        let tokens = firstTurn + Self.imageBlock() + Self.imageBlock() + Self.text(2, seed: 8)
        let frames = [THW(1, 8, 8), THW(1, 8, 4), THW(1, 4, 8)]
        let pixels = concatenated((0 ..< 3).map { Self.pixels(seed: $0) })
        let full = Self.input(
            tokens, image: LMInput.ProcessedImage(pixels: pixels, frames: frames))

        let suffix = try #require(
            model.splitPreparedInput(full, droppingFirst: firstTurn.count))
        #expect(suffix.image?.frames?.map { [$0.t, $0.h, $0.w] } == [[1, 8, 4], [1, 4, 8]])
        let kept = try #require(suffix.image?.pixels)
        #expect(kept.shape == [2, 3, 8, 8])
        #expect(allClose(kept, pixels[1...]).item(Bool.self))
        #expect(suffix.text.mask?.shape == [1, tokens.count - firstTurn.count])
    }

    @Test func splitRefusesUnsafeBoundaries() throws {
        let model = try Self.makeGemma4(bidirectional: true)
        let tokens = Self.text(3, seed: 9) + Self.imageBlock() + Self.text(3, seed: 10)
        let firstSoftToken = 4
        let full = Self.input(tokens, image: LMInput.ProcessedImage(pixels: Self.pixels(seed: 0)))

        // Inside the image.
        #expect(model.splitPreparedInput(full, droppingFirst: firstSoftToken + 2) == nil)
        // After the only image: the suffix would carry no media.
        #expect(model.splitPreparedInput(full, droppingFirst: tokens.count - 2) == nil)
        // Placeholder runs that do not match the payload.
        let twoImages = Self.input(
            tokens + Self.imageBlock(), image: LMInput.ProcessedImage(pixels: Self.pixels(seed: 0)))
        #expect(model.splitPreparedInput(twoImages, droppingFirst: 2) == nil)
        // A padding mask.
        var mask = [Int8](repeating: 1, count: tokens.count)
        mask[0] = 0
        let padded = LMInput(
            text: .init(
                tokens: MLXArray(tokens).expandedDimensions(axis: 0),
                mask: MLXArray(mask).expandedDimensions(axis: 0)),
            image: LMInput.ProcessedImage(pixels: Self.pixels(seed: 0)))
        #expect(model.splitPreparedInput(padded, droppingFirst: 2) == nil)

        // A cut just before the image's first soft token keeps the whole image.
        #expect(model.splitPreparedInput(full, droppingFirst: firstSoftToken) != nil)
    }

    // MARK: - Mask

    /// The overlay must apply on a warm cache, whose mask spans the cached keys too.
    @Test(arguments: [false, true])
    func overlayAppliesOnWarmCache(rotating: Bool) throws {
        let cached = 10
        let window = 8
        let cache: KVCache = rotating ? RotatingKVCache(maxSize: window) : KVCacheSimple()
        let rows = MLXArray.zeros([1, 1, cached, 4])
        _ = cache.update(keys: rows, values: rows)

        // text, image, image, image, text, text
        let tokenTypeIds = MLXArray([0, 1, 1, 1, 0, 0].map { Int32($0) }).reshaped(1, 6)
        let h = MLXArray.zeros([1, 6, 4])
        let windowSize: Int? = rotating ? window : nil
        let base = createAttentionMask(
            h: h, cache: cache, windowSize: windowSize, returnArray: true)
        guard case .array(let baseMask) = base else {
            Issue.record("Expected an array mask")
            return
        }
        let overlaid = gemma4ApplyBlockwiseBidirectionalOverlay(
            base, tokenTypeIds: tokenTypeIds, sequenceLength: 6, cacheOffset: cache.offset,
            windowSize: windowSize)
        guard case .array(let mask) = overlaid else {
            Issue.record("Expected an array mask")
            return
        }
        let keys = baseMask.dim(-1)
        #expect(keys > 6)
        #expect(mask.shape.suffix(2) == [6, keys])
        let newStart = keys - 6
        let rowsOut = mask.reshaped(6, keys).asArray(Bool.self)
        func attends(_ query: Int, _ key: Int) -> Bool { rowsOut[query * keys + key] }

        // The first image token sees the later ones; text after the image does not.
        #expect(attends(1, newStart + 3))
        #expect(!attends(0, newStart + 1))
        #expect(!attends(3, newStart + 4))
        // Cached keys stay as the base mask has them.
        let baseRows = broadcast(baseMask, to: [6, keys]).asArray(Bool.self)
        for query in 0 ..< 6 {
            for key in 0 ..< newStart {
                #expect(attends(query, key) == baseRows[query * keys + key])
            }
        }
    }

    /// `.causal` from a warm cache spans the cached keys too.
    @Test func overlayBuildsCausalBaseFromCacheOffset() {
        let tokenTypeIds = MLXArray([1, 1, 0].map { Int32($0) }).reshaped(1, 3)
        let overlaid = gemma4ApplyBlockwiseBidirectionalOverlay(
            .causal, tokenTypeIds: tokenTypeIds, sequenceLength: 3, cacheOffset: 5,
            windowSize: nil)
        guard case .array(let mask) = overlaid else {
            Issue.record("Expected an array mask")
            return
        }
        let expected = MLXArray(
            [
                true, true, true, true, true, true, true, false,
                true, true, true, true, true, true, true, false,
                true, true, true, true, true, true, true, true,
            ]
        ).reshaped(3, 8)
        #expect(mask.shape.suffix(2) == [3, 8])
        #expect(all(mask.reshaped(3, 8) .== expected).item(Bool.self))
    }

    // MARK: - Gemma4Unified

    /// A unified config without `use_bidirectional_attention` defaults to `"vision"`, so
    /// there is no unified control.
    @Test func unifiedSplitImageTurnMatchesColdPrefill() throws {
        let model = try Self.makeUnified()
        let firstTurn = Self.text(4, seed: 11) + Self.imageBlock() + Self.text(9, seed: 12)
        let tokens =
            firstTurn + Self.text(3, seed: 13) + Self.imageBlock() + Self.text(3, seed: 14)
        let patches = concatenated([Self.patches(seed: 0), Self.patches(seed: 1)])
        let positions = concatenated([Self.patchPositions(), Self.patchPositions(shift: 1)])
        let full = Self.input(
            tokens, image: LMInput.ProcessedImage(pixels: patches, positionIds: positions))

        let suffixCount = tokens.count - firstTurn.count
        let cold = try Self.suffixLogits(
            model.prepare(
                full, cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init(stepSize: 512)), count: suffixCount)

        let warmCache = try model.newCache(parameters: nil)
        _ = try model.prepare(
            Self.input(
                firstTurn,
                image: LMInput.ProcessedImage(
                    pixels: patches[..<1], positionIds: positions[..<1])),
            cache: warmCache, state: nil, prefill: .init(stepSize: 512))
        let suffix = try #require(
            model.splitPreparedInput(full, droppingFirst: firstTurn.count))
        let keptPositions = try #require(suffix.image?.positionIds)
        #expect(all(keptPositions .== positions[1...]).item(Bool.self))
        #expect(suffix.image?.pixels.dim(0) == 1)
        let warm = try Self.suffixLogits(
            model.prepare(suffix, cache: warmCache, state: nil, prefill: .init(stepSize: 512)),
            count: suffixCount)

        let drift = Self.maxDrift(warm, cold)
        #expect(drift < Self.tolerance, "warm and cold logits differ by \(drift)")
    }

    // MARK: - Helpers

    private static func text(_ count: Int, seed: Int) -> [Int] {
        (0 ..< count).map { 1 + ($0 * 7 + seed * 13) % 97 }
    }

    private static func imageBlock() -> [Int] {
        [boiToken] + Array(repeating: imageToken, count: softTokensPerImage) + [eoiToken]
    }

    private static func input(_ tokens: [Int], image: LMInput.ProcessedImage? = nil) -> LMInput {
        let array = MLXArray(tokens).expandedDimensions(axis: 0)
        guard let image else { return LMInput(tokens: array) }
        return LMInput(
            text: .init(tokens: array, mask: ones(like: array).asType(.int8)), image: image)
    }

    private static func pixels(seed: Int) -> MLXArray {
        sin(MLXArray(0 ..< 192).reshaped([1, 3, 8, 8]).asType(.float32) * Float(seed + 1) * 0.37)
    }

    /// Unified vision input: one row of 4 model patches of 48 values each.
    private static func patches(seed: Int) -> MLXArray {
        sin(MLXArray(0 ..< 192).reshaped([1, 4, 48]).asType(.float32) * Float(seed + 1) * 0.29)
    }

    private static func patchPositions(shift: Int32 = 0) -> MLXArray {
        MLXArray([0, 0, 1, 0, 0, 1, 1, 1].map { $0 + shift } as [Int32]).reshaped(1, 4, 2)
    }

    /// Logits at the last `count` positions, which includes the image tokens of the suffix.
    private static func suffixLogits(_ result: PrepareResult, count: Int) throws -> MLXArray {
        guard case .logits(let output) = result else {
            throw Gemma4WarmPrefillTestError.expectedLogits
        }
        let length = output.logits.dim(1)
        try #require(length >= count)
        let logits = output.logits[0..., (length - count)..., 0...]
        eval(logits)
        return logits
    }

    private static func maxDrift(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private static func makeGemma4(bidirectional: Bool) throws -> Gemma4 {
        let config = try JSONDecoder().decode(
            Gemma4Configuration.self, from: Data(gemma4JSON(bidirectional).utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 7)) { Gemma4(config) }
        model.apply { $0.dtype.isFloatingPoint ? $0.asType(.float16) : $0 }
        eval(model)
        return model
    }

    private static func makeUnified() throws -> Gemma4Unified {
        let config = try JSONDecoder().decode(
            Gemma4UnifiedConfiguration.self, from: Data(unifiedJSON.utf8))
        #expect(config.textConfiguration.useBidirectionalAttention == "vision")
        let model = withRandomState(MLXRandom.RandomState(seed: 7)) { Gemma4Unified(config) }
        model.apply { $0.dtype.isFloatingPoint ? $0.asType(.float16) : $0 }
        eval(model)
        return model
    }

    /// Layers alternate sliding and full attention; the sliding window is 8.
    private static func gemma4JSON(_ bidirectional: Bool) -> String {
        let bidirectionalField = bidirectional ? #""use_bidirectional_attention": "vision","# : ""
        return """
            {
              "model_type": "gemma4",
              "image_token_id": \(imageToken), "boi_token_id": \(boiToken),
              "eoi_token_id": \(eoiToken),
              "text_config": {
                \(bidirectionalField)
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                "global_head_dim": 32, "vocab_size": 200, "vocab_size_per_layer_input": 200,
                "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 8,
                "sliding_window": 8, "sliding_window_pattern": 2, "max_position_embeddings": 512
              },
              "vision_config": {
                "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32,
                "num_attention_heads": 2, "head_dim": 8, "patch_size": 4,
                "default_output_length": 4, "pooling_kernel_size": 1, "position_embedding_size": 8
              }
            }
            """
    }

    private static let unifiedJSON = """
        {
          "model_type": "gemma4_unified", "vocab_size": 200,
          "image_token_id": \(imageToken), "boi_token_id": \(boiToken),
          "eoi_token_id": \(eoiToken), "audio_token_id": 103, "video_token_id": 104,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
            "vocab_size": 200, "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "sliding_window": 8, "sliding_window_pattern": 2, "max_position_embeddings": 512
          },
          "vision_config": {
            "model_type": "gemma4_unified_vision",
            "patch_size": 2, "pooling_kernel_size": 2, "model_patch_size": 4,
            "mm_embed_dim": 8, "mm_posemb_size": 4, "num_soft_tokens": 4,
            "output_proj_dims": 8
          },
          "audio_config": null
        }
        """
}

private enum Gemma4WarmPrefillTestError: Error {
    case expectedLogits
}
