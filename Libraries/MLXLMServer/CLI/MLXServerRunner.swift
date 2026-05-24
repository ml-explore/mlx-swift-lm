// Copyright © 2026 Apple Inc.

import Hummingbird
import MLXLMCommon

public enum MLXServer {
    public static func run(configuration: MLXServerConfiguration) async throws {
        var modelConfiguration = ModelConfiguration(
            id: configuration.model,
            revision: configuration.revision
        )
        if let parser = configuration.toolCallParser {
            modelConfiguration.toolCallFormat = try ServerToolParser.resolve(
                requested: parser,
                modelType: configuration.modelType
            )
        }

        let model = try await MLXServerModelLoader.load(
            configuration: modelConfiguration
        )
        let engine = MLXModelContainerEngine(
            modelID: configuration.model,
            model: model,
            modelType: configuration.modelType,
            defaultToolCallParser: configuration.toolCallParser
        )
        let embeddingEngine: MLXEmbedderContainerEngine?
        if let embeddingModelID = configuration.embeddingModel {
            let embeddingModel = try await MLXServerEmbedderLoader.load(
                configuration: .init(id: embeddingModelID)
            )
            embeddingEngine = MLXEmbedderContainerEngine(
                modelID: embeddingModelID,
                model: embeddingModel
            )
        } else {
            embeddingEngine = nil
        }
        let service = MLXOpenAIService(
            engine: engine,
            embeddingEngine: embeddingEngine,
            defaultReasoningParser: configuration.reasoningParser
        )
        let app = MLXServerApplication.buildApplication(
            service: service,
            host: configuration.host,
            port: configuration.port
        )
        try await app.runService()
    }
}
