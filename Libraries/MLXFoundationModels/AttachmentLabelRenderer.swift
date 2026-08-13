// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation

/// Turns FoundationModels attachment labels into the text an MLX chat message
/// can carry.
///
/// FoundationModels puts the label immediately before its image, as the literal
/// string `[label]`. That is not reachable here: an MLX `Chat.Message` holds one
/// text string plus an unnamed array of images, and every VLM message generator
/// emits all of a message's images ahead of its text. So the closest available
/// arrangement is a single line, placed before the caller's own text, naming the
/// labels in image order. Measured on Qwen3-VL-4B and gemma-4-e4b at two and
/// three images, both attribute a label to the right picture from this form as
/// reliably as they do from a per-image marker.
///
/// The bracket delimiters match FoundationModels deliberately. Models echo the
/// marker into their prose, so keeping the shape identical means an app can
/// scrub leaked labels with the same pattern it would use against the on-device
/// model.
///
/// Labels reach the tokenizer as ordinary prompt text, so one whose rendered
/// form is a tokenizer special token would corrupt the prompt's image
/// placeholders. ``AttachmentLabelValidator`` rejects those once the tokenizer is
/// known, and it checks the bracketed form written below rather than the bare
/// label, because the brackets are part of what gets tokenized: Mistral-family
/// image tokens are bracketed, so a label of `IMG` renders as `[IMG]`.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct AttachmentLabelRenderer {

    /// The renderer used by ``TranscriptConverter``.
    static let `default` = AttachmentLabelRenderer()

    /// Text naming `labels` in image order, or `nil` when no image carries a
    /// label.
    ///
    /// Returning `nil` for an entirely unlabeled set mirrors FoundationModels,
    /// which renders an unlabeled attachment as the bare image with no marker.
    ///
    /// - Parameter labels: One entry per image in the message, in the order the
    ///   images appear, with `nil` for an image that carries no label.
    func legend(for labels: [String?]) -> String? {
        guard labels.contains(where: { $0 != nil }) else { return nil }

        let names = labels.map { label in
            // An unlabeled image still occupies a slot: order is the only thing
            // connecting a name to a picture, so dropping it would shift every
            // name after it onto the wrong image.
            label.map { "[\($0)]" } ?? "(unlabeled)"
        }

        if names.count == 1 {
            return "The image above is \(names[0])."
        }
        return "The \(names.count) images above are, in order: \(names.joined(separator: ", "))."
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
