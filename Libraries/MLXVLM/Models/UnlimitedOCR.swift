//
//  UnlimitedOCR.swift
//  mlx-swift-lm
//
//  Unlimited-OCR delta over DeepSeek-OCR: native model_type registration and
//  Reference Sliding Window Attention (R-SWA) via RingSlidingKVCache.
//  Port of mlx_vlm.models.unlimited_ocr (config + unlimitedocr + processor).
//

import Foundation
import MLXLMCommon

/// Configuration for Unlimited-OCR HF packs (`model_type`: `unlimited-ocr` /
/// `unlimited_ocr`). Same JSON schema as DeepSeek-OCR plus R-SWA window fields.
public typealias UnlimitedOCRConfiguration = DeepseekOCRConfiguration

/// Processor for Unlimited-OCR.
///
/// Python `processing_unlimitedocr.py` subclasses DeepseekOCRProcessor with
/// torch-free `from_pretrained`, default `sft_format="unlimitedocr"`,
/// `max_num=32` tiling, and multipage single-`<image>` support. Single-page
/// inference templates match ``DeepseekOCRProcessor`` (TASK-023 audit), so this
/// remains a typealias. Multipage fused prepare (`Multi page parsing.`, base
/// mode) and optional `max_num=32` live on the shared processor (TASK-022):
/// pass multiple `UserInput.Image`s and ``DeepseekOCRProcessor/modeContext(_:)``
/// `.base`, or ``DeepseekOCRProcessor/unlimitedContext(_:)`` for Unlimited tiling.
///
/// Audit: coordination-hub `docs/guides/unlimited-ocr-processor-audit.md`.
public typealias UnlimitedOCRProcessorConfiguration = DeepseekOCRProcessorConfiguration
public typealias UnlimitedOCRProcessor = DeepseekOCRProcessor

/// Unlimited-OCR VLM: DeepSeek-OCR weights + R-SWA decode cache.
///
/// Python defaults `sliding_window_size` to 128 when unset; native Unlimited
/// routing always uses ``RingSlidingKVCache`` (never unbounded
/// ``KVCacheSimple``).
///
/// ## Optional n-gram no-repeat (TASK-021)
///
/// Upstream Unlimited-OCR / mlx-vlm leave the sliding-window no-repeat n-gram
/// logits processor **off** by default. Opt in via generate parameters:
///
/// ```swift
/// var parameters = GenerateParameters(temperature: 0, maxTokens: 8192)
/// parameters.noRepeatNgramSize = SlidingWindowNoRepeatNGramProcessor.unlimitedOCRNgramSize // 35
/// parameters.noRepeatNgramWindowSize = SlidingWindowNoRepeatNGramProcessor.unlimitedOCRSingleImageWindow // 128
/// // multi-page/PDF examples often use unlimitedOCRMultiPageWindow (1024)
/// ```
///
/// Or attach ``SlidingWindowNoRepeatNGramProcessor/unlimitedOCRSingleImage()``
/// through a custom ``LogitProcessor`` chain. Disabled (`noRepeatNgramSize` nil/0)
/// matches Python Unlimited defaults.
public final class UnlimitedOCR: DeepseekOCR {

    public override func newCache(parameters: GenerateParameters?) -> [KVCache] {
        _ = parameters
        let window = config.resolvedSlidingWindowSize ?? 128
        return makeCaches(numLayers: kvHeads.count, slidingWindowSize: window)
    }
}
