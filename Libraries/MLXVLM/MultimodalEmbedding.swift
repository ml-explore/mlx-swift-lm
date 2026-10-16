// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// A kind of input that an embedding model places in the same vector space as text.
public enum EmbeddingModality: Hashable, Sendable {
    case text, image, video, audio
}

/// The task an embedding is prepared for. Each model renders it as its own instruction.
/// Use `.searchQuery` for queries and `.document` for corpus items.
public enum EmbeddingTask: Hashable, Sendable {
    case searchQuery
    case document
    case clustering
    case classification
    case questionAnswering
    case factChecking
    case codeRetrieval
    case sentenceSimilarity
    /// The input is embedded as given, without a task instruction.
    case none
}

/// One independently embedded item: text and media in reading order.
public struct EmbeddingInput {

    /// One piece of an input.
    public enum Part {
        case text(String)
        case image(UserInput.Image)
        case video(UserInput.Video)
        case audio(UserInput.Audio)
    }

    public let parts: [Part]
    /// A document title or file name; models trained with titles use it for `.document`.
    public let title: String?

    public init(_ parts: [Part], title: String? = nil) {
        self.parts = parts
        self.title = title
    }

    /// Text followed by images.
    public init(text: String = "", title: String? = nil, images: [UserInput.Image] = []) {
        self.init([.text(text)] + images.map(Part.image), title: title)
    }
}

/// A loaded embedding model that maps text and media into one space of unit-length
/// float32 vectors. Each model owns its prompts, sequence layout, and batching.
public protocol MultimodalEmbedding: Sendable {

    /// The modalities that the loaded checkpoint accepts.
    var supportedModalities: Set<EmbeddingModality> { get async }

    /// Embeds independent items; returns one vector per item, in input order.
    func embed(_ inputs: sending [EmbeddingInput], task: EmbeddingTask) async throws -> [[Float]]
}
