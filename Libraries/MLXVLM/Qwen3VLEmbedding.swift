// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Text and image embeddings from a Qwen3-VL-Embedding checkpoint.
/// Load the checkpoint with ``VLMModelFactory`` and reuse its container for every call.
public struct Qwen3VLEmbedding: Sendable {
    /// One independently pooled item. Images are transferred into the container's isolation.
    public struct Input {
        public let text: String
        public let images: [UserInput.Image]

        public init(text: String = "", images: [UserInput.Image] = []) {
            self.text = text
            self.images = images
        }
    }

    public enum Error: Swift.Error, LocalizedError {
        case unsupportedModel
        case emptyInput
        case contextExceeded
        case invalidEmbedding

        public var errorDescription: String? {
            switch self {
            case .unsupportedModel: "Qwen3-VL embeddings require a Qwen3-VL model."
            case .emptyInput: "An embedding input must contain text or images."
            case .contextExceeded: "The embedding input exceeds 32768 tokens."
            case .invalidEmbedding: "The model returned an invalid embedding."
            }
        }
    }

    public static let defaultInstruction = "Represent the user's input."
    private let container: ModelContainer

    /// The container must contain embedding-trained weights, rather than instruction-model weights.
    public init(container: ModelContainer) async throws {
        try Task.checkCancellation()
        guard await container.perform({ $0.model is Qwen3VL }) else {
            throw Error.unsupportedModel
        }
        try Task.checkCancellation()
        self.container = container
    }

    /// Returns one unit-length float32 vector, pooling the checkpoint's terminal token.
    public func embed(
        _ input: consuming sending Input, instruction: String = defaultInstruction
    ) async throws -> [Float] {
        let vectors = try await embed([input], instruction: instruction)
        return vectors[0]
    }

    /// Embeds independent items in input order, bounding visual working memory to one item.
    /// Use the same instruction for items that will be compared in the same embedding space.
    public func embed(
        _ inputs: consuming sending [Input], instruction: String = defaultInstruction
    ) async throws -> [[Float]] {
        try Task.checkCancellation()
        return try await container.perform(nonSendable: inputs) { context, inputs in
            guard let model = context.model as? Qwen3VL else { throw Error.unsupportedModel }
            for input in inputs {
                guard
                    !input.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        || !input.images.isEmpty
                else { throw Error.emptyInput }
            }
            var vectors: [[Float]] = []
            for input in inputs {
                try Task.checkCancellation()
                let prepared = try await context.processor.prepare(
                    input: UserInput(
                        chat: [.system(instruction), .user(input.text, images: input.images)],
                        processing: .init(minPixels: 4_096, maxPixels: 1_310_720)))
                // Embedding checkpoints apply their tokenizer postprocessor after the chat template.
                // Preserve the terminal token they were trained to pool.
                let prompt = context.tokenizer.decode(
                    tokenIds: prepared.text.tokens.asArray(Int.self), skipSpecialTokens: false)
                let tokens = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
                guard !tokens.isEmpty else { throw Error.emptyInput }
                guard tokens.count <= 32_768 else { throw Error.contextExceeded }
                let completed = LMInput(
                    text: .init(tokens: MLXArray(tokens).expandedDimensions(axis: 0)),
                    image: prepared.image, video: prepared.video)
                let hidden = try model.hiddenStates(completed)
                let last = hidden[0, -1, 0...].asType(.float32)
                let vector = last / MLX.maximum(MLXLinalg.norm(last), MLXArray(Float(1e-12)))
                try MLX.checkedEval(vector)
                let values = vector.asArray(Float.self)
                guard values.allSatisfy(\.isFinite), values.contains(where: { $0 != 0 }) else {
                    throw Error.invalidEmbedding
                }
                vectors.append(values)
            }
            try Task.checkCancellation()
            return vectors
        }
    }
}
