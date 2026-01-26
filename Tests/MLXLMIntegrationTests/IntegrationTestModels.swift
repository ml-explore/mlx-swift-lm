// Copyright © 2025 Apple Inc.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM

enum IntegrationTestModelIDs {
    static let llmModelId = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    static let vlmModelId = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
}

actor IntegrationTestModels {
    static let shared = IntegrationTestModels()

    private var llmTask: Task<ModelContainer, Error>?
    private var vlmTask: Task<ModelContainer, Error>?

    func llmContainer() async throws -> ModelContainer {
        if let task = llmTask {
            return try await task.value
        }

        let task = Task {
            try await LLMModelFactory.shared.loadContainer(
                configuration: .init(id: IntegrationTestModelIDs.llmModelId)
            )
        }
        llmTask = task
        return try await task.value
    }

    func vlmContainer() async throws -> ModelContainer {
        if let task = vlmTask {
            return try await task.value
        }

        let task = Task {
            try await VLMModelFactory.shared.loadContainer(
                configuration: .init(id: IntegrationTestModelIDs.vlmModelId)
            )
        }
        vlmTask = task
        return try await task.value
    }
}
