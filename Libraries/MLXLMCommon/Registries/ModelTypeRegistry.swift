// Copyright © 2024 Apple Inc.

import Foundation

public actor ModelTypeRegistry {

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: (Data) throws -> any LanguageModel]) {
        self.creators = creators
    }

    private var creators: [String: (Data) throws -> any LanguageModel]

    /// Add a new model to the type registry.
    public func registerModelType(
        _ type: String, creator: @escaping (Data) throws -> any LanguageModel
    ) {
        creators[type] = creator
    }

    /// Given a `modelType` and configuration data instantiate a new `LanguageModel`.
    public func createModel(configuration: Data, modelType: String) throws -> sending LanguageModel
    {
        guard let creator = creators[modelType] else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try creator(configuration)
    }

}
