// Copyright © 2024 Apple Inc.

import MLX
import MLXLMCommon

public protocol VLMModel: LanguageModel, LoRAModel {
}

// The iterator parity tests cover these models with text-only prompts and fresh caches.
extension Idefics3: BatchDecodableModel {}
extension Qwen2VL: BatchDecodableModel {}
extension Qwen25VL: BatchDecodableModel {}
