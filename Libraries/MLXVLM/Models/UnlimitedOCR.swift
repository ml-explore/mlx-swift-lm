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

/// Processor for Unlimited-OCR. Python `processing_unlimitedocr.py` subclasses
/// DeepseekOCRProcessor with torch-free `from_pretrained` and
/// `sft_format="unlimitedocr"`; Swift already shares the inference tokenize path.
public typealias UnlimitedOCRProcessorConfiguration = DeepseekOCRProcessorConfiguration
public typealias UnlimitedOCRProcessor = DeepseekOCRProcessor

/// Unlimited-OCR VLM: DeepSeek-OCR weights + R-SWA decode cache.
///
/// Python defaults `sliding_window_size` to 128 when unset; native Unlimited
/// routing always uses ``RingSlidingKVCache`` (never unbounded
/// ``KVCacheSimple``).
public final class UnlimitedOCR: DeepseekOCR {

    public override func newCache(parameters: GenerateParameters?) -> [KVCache] {
        _ = parameters
        let window = config.resolvedSlidingWindowSize ?? 128
        return makeCaches(numLayers: kvHeads.count, slidingWindowSize: window)
    }
}
