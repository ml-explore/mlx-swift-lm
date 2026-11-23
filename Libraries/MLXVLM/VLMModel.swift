// Copyright © 2024 Apple Inc.

import MLX
import MLXLMCommon

public protocol VLMModel: LanguageModel, LoRAModel {
}

extension LMInput.Text {
    /// Whether the mask, if any, keeps every token: the all-ones mask a processor pairs with
    /// media. Reduced on the GPU, so only one scalar is read back.
    var masksNothing: Bool {
        guard let mask else { return true }
        return mask.size == tokens.size && (mask .== 1).all().item(Bool.self)
    }
}
