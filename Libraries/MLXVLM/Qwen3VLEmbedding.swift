// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Text and image embeddings from a Qwen3-VL-Embedding checkpoint.
/// Load an embedding-trained checkpoint with ``MultimodalEmbeddingFactory``, or reuse a
/// ``VLMModelFactory`` container. Vision weights load on the first image.
/// This implementation accepts text and images; video and audio are not supported.
public struct Qwen3VLEmbedding: MultimodalEmbedding {
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
        case unsupportedMedia

        public var errorDescription: String? {
            switch self {
            case .unsupportedModel: "Qwen3-VL embeddings require a Qwen3-VL model."
            case .emptyInput: "An embedding input must contain text or images."
            case .contextExceeded: "The embedding input exceeds 32768 tokens."
            case .invalidEmbedding: "The model returned an invalid embedding."
            case .unsupportedMedia: "Qwen3-VL embeddings support text and images only."
            }
        }
    }

    public static let defaultInstruction = "Represent the user's input."
    private let container: ModelContainer

    public var supportedModalities: Set<EmbeddingModality> {
        get async {
            await container.perform { context in
                context.configuration.excludedComponents.contains(.vision)
                    ? [.text] : [.text, .image]
            }
        }
    }

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
        var configuration = ResolvedModelConfiguration(directory: modelDirectory)
        configuration.excludedComponents = excludedComponents
        try await self.init(
            configuration: configuration, tokenizerLoader: tokenizerLoader,
            componentLoading: componentLoading)
    }

    package init(
        configuration: ResolvedModelConfiguration, tokenizerLoader: any TokenizerLoader,
        componentLoading: ComponentLoading = .onFirstUse
    ) async throws {
        try Task.checkCancellation()
        let modelDirectory = configuration.modelDirectory
        let data = try Data(contentsOf: modelDirectory.appendingPathComponent("config.json"))
        let config = try JSONDecoder.json5().decode(Qwen3VLConfiguration.self, from: data)
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        let model = Qwen3VL(config)
        try await loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: base.perLayerQuantization,
            weightFileSelection: configuration.weightFileSelection,
            excludedComponents: configuration.excludedComponents, componentLoading: componentLoading
        )
        try Task.checkCancellation()
        let tokenizer = try await tokenizerLoader.load(from: configuration.tokenizerDirectory)
        try Task.checkCancellation()
        var contextConfiguration = ModelConfiguration(directory: modelDirectory)
        contextConfiguration.excludedComponents = configuration.excludedComponents
        self.container = ModelContainer(
            context: ModelContext(
                configuration: contextConfiguration,
                model: model,
                processor: Processor(directory: modelDirectory, tokenizer: tokenizer),
                tokenizer: tokenizer))
    }

    /// Embeds items in reading order. Task instructions are general defaults; use the
    /// instruction overload to tailor a prompt to a particular retrieval task.
    /// Titles are not used by this model.
    public func embed(
        _ inputs: sending [EmbeddingInput], task: EmbeddingTask
    ) async throws -> [[Float]] {
        let instruction: String? =
            switch task {
            case .none: nil
            case .searchQuery, .document: Self.defaultInstruction
            case .clustering: "Identify the topic or theme of the input."
            case .classification: "Classify the input."
            case .questionAnswering: "Retrieve information that answers the question."
            case .factChecking: "Retrieve information that verifies the claim."
            case .codeRetrieval: "Retrieve code matching the input."
            case .sentenceSimilarity: "Find inputs with similar meaning."
            }
        return try await embedParts(inputs, instruction: instruction)
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
        let items = inputs.map {
            EmbeddingInput($0.images.map(EmbeddingInput.Part.image) + [.text($0.text)])
        }
        return try await embedParts(items, instruction: instruction)
    }

    private func embedParts(
        _ inputs: sending [EmbeddingInput], instruction: String?
    ) async throws -> [[Float]] {
        try Task.checkCancellation()
        return try await container.perform(nonSendable: inputs) { context, inputs in
            guard let model = context.model as? Qwen3VL else { throw Error.unsupportedModel }
            for input in inputs {
                var hasContent = false
                for part in input.parts {
                    switch part {
                    case .text(let text):
                        hasContent =
                            hasContent
                            || !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    case .image:
                        hasContent = true
                        if context.configuration.excludedComponents.contains(.vision) {
                            throw OnDemandComponentError.unavailable(.vision)
                        }
                    case .video, .audio: throw Error.unsupportedMedia
                    }
                }
                guard hasContent else { throw Error.emptyInput }
            }
            var vectors: [[Float]] = []
            vectors.reserveCapacity(inputs.count)
            for input in inputs {
                try Task.checkCancellation()
                var parts: [[String: any Sendable]] = []
                var images: [UserInput.Image] = []
                for part in input.parts {
                    switch part {
                    case .text(let text): parts.append(["type": "text", "text": text])
                    case .image(let image):
                        let imageInput = UserInput(chat: [.user("", images: [image])])
                            .removingSpecialTokenLabels(using: context.tokenizer)
                        let generated = Qwen3VLMessageGenerator().generate(from: imageInput)
                        guard let imageParts = generated.first?["content"] as? [[String: String]]
                        else {
                            throw Error.invalidEmbedding
                        }
                        parts.append(contentsOf: imageParts.dropLast())
                        images.append(image)
                    case .video, .audio: throw Error.unsupportedMedia
                    }
                }
                var messages: [Message] = []
                if let instruction {
                    messages.append([
                        "role": "system", "content": [["type": "text", "text": instruction]],
                    ])
                }
                messages.append(["role": "user", "content": parts])
                let prepared = try await context.processor.prepare(
                    input: UserInput(
                        prompt: .messages(messages), images: images,
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
