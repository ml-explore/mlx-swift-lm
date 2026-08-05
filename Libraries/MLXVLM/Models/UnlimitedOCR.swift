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
/// `max_num=32` tiling, and multipage single-`<image>` support. Comparing the
/// two processors tensor-by-tensor showed the single-page inference templates
/// are identical, so a distinct Swift type would carry no behavior — hence a
/// typealias rather than a subclass. Multipage fused prepare
/// (`Multi page parsing.`, base mode) and optional `max_num=32` live on the
/// shared processor: pass multiple `UserInput.Image`s and
/// ``DeepseekOCRProcessor/modeContext(_:)`` `.base`, or
/// ``DeepseekOCRProcessor/unlimitedContext(_:)`` for Unlimited tiling.
public typealias UnlimitedOCRProcessorConfiguration = DeepseekOCRProcessorConfiguration
public typealias UnlimitedOCRProcessor = DeepseekOCRProcessor

/// Unlimited-OCR VLM: DeepSeek-OCR weights + R-SWA decode cache.
///
/// Python defaults `sliding_window_size` to 128 when unset; native Unlimited
/// routing always uses `RingSlidingKVCache` (never unbounded
/// `KVCacheSimple`).
///
/// ## Optional n-gram no-repeat
///
/// Upstream Unlimited-OCR / mlx-vlm leave the sliding-window no-repeat n-gram
/// logits processor **off** by default. Opt in by attaching the processor
/// through `GenerationComponents`:
///
/// ```swift
/// let parameters = GenerateParameters(temperature: 0, maxTokens: 8192)
/// let components = GenerationComponents(
///     logitProcessorFactory: { SlidingWindowNoRepeatNGramProcessor.unlimitedOCRSingleImage() }
/// )
/// // multi-page/PDF examples often use .unlimitedOCRMultiPage() (window 1024)
/// ```
///
/// Leaving `components` empty matches Python Unlimited defaults (guard off).
public final class UnlimitedOCR: DeepseekOCR {

    public override func newCache(parameters: GenerateParameters?) -> [KVCache] {
        _ = parameters
        let window = config.resolvedSlidingWindowSize ?? 128
        return makeCaches(numLayers: kvHeads.count, slidingWindowSize: window)
    }
}
