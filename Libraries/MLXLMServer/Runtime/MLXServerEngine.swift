// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

public struct MLXServerModel: Codable, Sendable, Equatable {
    public var id: String
    public var object: String
    public var created: Int?
    public var ownedBy: String

    private enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case ownedBy = "owned_by"
    }

    public init(
        id: String,
        object: String = "model",
        created: Int? = nil,
        ownedBy: String = "local"
    ) {
        self.id = id
        self.object = object
        self.created = created
        self.ownedBy = ownedBy
    }
}

public struct ServerGenerationInfo: Sendable, Equatable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var promptTime: TimeInterval
    public var generationTime: TimeInterval
    public var stopReason: String

    public init(
        promptTokens: Int,
        completionTokens: Int,
        promptTime: TimeInterval,
        generationTime: TimeInterval,
        stopReason: String
    ) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.promptTime = promptTime
        self.generationTime = generationTime
        self.stopReason = stopReason
    }

    public init(_ info: GenerateCompletionInfo) {
        self.init(
            promptTokens: info.promptTokenCount,
            completionTokens: info.generationTokenCount,
            promptTime: info.promptTime,
            generationTime: info.generateTime,
            stopReason: info.stopReason.openAIFinishReason
        )
    }
}

public enum MLXServerGenerationEvent: Sendable, Equatable {
    case content(String)
    case toolCall(ToolCall)
    case info(ServerGenerationInfo)
}

public protocol MLXServerEngine: Sendable {
    func availableModels() async throws -> [MLXServerModel]
    func streamChatCompletion(
        request: OpenAIChatCompletionRequest
    ) async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error>
    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse
}

extension GenerateStopReason {
    var openAIFinishReason: String {
        switch self {
        case .stop:
            return "stop"
        case .length:
            return "length"
        case .cancelled:
            return "stop"
        }
    }
}
