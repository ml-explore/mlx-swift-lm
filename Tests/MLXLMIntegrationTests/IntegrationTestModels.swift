// Copyright © 2025 Apple Inc.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM

enum IntegrationTestModelIDs {
    static let llmModelId = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    static let vlmModelId = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

    static let lfm2ModelId = "mlx-community/LFM2-2.6B-Exp-4bit"
    static let glm4ModelId = "mlx-community/GLM-4-9B-0414-4bit"
    static let mistral3ModelId = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"
    static let nemotronModelId = "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"
}

actor IntegrationTestModels {
    static let shared = IntegrationTestModels()

    private init() {}

    private var llmTask: Task<ModelContainer, Error>?
    private var vlmTask: Task<ModelContainer, Error>?

    private var lfm2Task: Task<ModelContainer, Error>?
    private var glm4Task: Task<ModelContainer, Error>?
    private var mistral3Task: Task<ModelContainer, Error>?
    private var nemotronTask: Task<ModelContainer, Error>?

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

    func lfm2Container() async throws -> ModelContainer {
        if let task = lfm2Task {
            return try await task.value
        }

        let task = Task {
            try await LLMModelFactory.shared.loadContainer(
                configuration: .init(id: IntegrationTestModelIDs.lfm2ModelId)
            )
        }
        lfm2Task = task
        return try await task.value
    }

    func glm4Container() async throws -> ModelContainer {
        if let task = glm4Task {
            return try await task.value
        }

        let task = Task {
            try await LLMModelFactory.shared.loadContainer(
                configuration: .init(id: IntegrationTestModelIDs.glm4ModelId)
            )
        }
        glm4Task = task
        return try await task.value
    }

    func mistral3Container() async throws -> ModelContainer {
        if let task = mistral3Task {
            return try await task.value
        }

        let task = Task {
            try await LLMModelFactory.shared.loadContainer(
                configuration: .init(id: IntegrationTestModelIDs.mistral3ModelId)
            )
        }
        mistral3Task = task
        return try await task.value
    }

    func nemotronContainer() async throws -> ModelContainer {
        if let task = nemotronTask {
            return try await task.value
        }

        let task = Task {
            try await LLMModelFactory.shared.loadContainer(
                configuration: .init(id: IntegrationTestModelIDs.nemotronModelId)
            )
        }
        nemotronTask = task
        return try await task.value
    }
}
