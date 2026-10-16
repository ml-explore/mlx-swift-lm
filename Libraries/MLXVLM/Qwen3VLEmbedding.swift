// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Text and image embeddings from a Qwen3-VL-Embedding checkpoint.
/// Load a local checkpoint or reuse a ``VLMModelFactory`` container for every call.
/// Vision weights load on the first image; text-only calls do not read them.
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

    /// Loads a local embedding checkpoint, with vision weights and its processor read on first use.
    /// Exclude `.vision` to keep the checkpoint text-only. Files must remain readable until use.
    public init(
        modelDirectory: URL, tokenizerLoader: any TokenizerLoader,
        excludedComponents: Set<ModelComponent> = [],
        componentLoading: ComponentLoading = .onFirstUse
    ) async throws {
        try Task.checkCancellation()
        let data = try Data(contentsOf: modelDirectory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3VLConfiguration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = Qwen3VL(config)
        try await loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: base.perLayerQuantization,
            excludedComponents: excludedComponents, componentLoading: componentLoading)
        try Task.checkCancellation()
        let tokenizer = try await tokenizerLoader.load(from: modelDirectory)
        try Task.checkCancellation()
        var configuration = ModelConfiguration(directory: modelDirectory)
        configuration.excludedComponents = excludedComponents
        self.container = ModelContainer(
            context: ModelContext(
                configuration: configuration,
                model: model,
                processor: Processor(directory: modelDirectory, tokenizer: tokenizer),
                tokenizer: tokenizer))
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
                if !input.images.isEmpty, context.configuration.excludedComponents.contains(.vision)
                {
                    throw OnDemandComponentError.unavailable(.vision)
                }
            }
            var vectors: [[Float]] = []
            vectors.reserveCapacity(inputs.count)
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
                if prepared.image != nil || prepared.video != nil {
                    try await model.loadMediaComponents([.vision])
                    try Task.checkCancellation()
                }
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

extension Qwen3VLEmbedding {
    private final class Processor: UserInputProcessor, @unchecked Sendable {
        let directory: URL
        let tokenizer: any Tokenizer
        private let lock = NSLock()
        private var mediaProcessor: Qwen3VLProcessor?

        init(directory: URL, tokenizer: any Tokenizer) {
            self.directory = directory
            self.tokenizer = tokenizer
        }

        func prepare(input: UserInput) async throws -> LMInput {
            if input.images.isEmpty, input.videos.isEmpty {
                let messages = Qwen3VLMessageGenerator().generate(
                    from: input.removingSpecialTokenLabels(using: tokenizer))
                let tokens = try tokenizer.applyChatTemplate(
                    messages: messages, tools: input.tools,
                    additionalContext: input.additionalContext)
                return LMInput(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
            }
            let processor = try lock.withLock {
                if let mediaProcessor { return mediaProcessor }
                let data = try Data(
                    contentsOf: directory.appendingPathComponent("preprocessor_config.json"))
                let config = try JSONDecoder().decode(
                    Qwen3VLProcessorConfiguration.self, from: data)
                let processor = Qwen3VLProcessor(config, tokenizer: tokenizer)
                mediaProcessor = processor
                return processor
            }
            return try await processor.prepare(input: input)
        }
    }
}
