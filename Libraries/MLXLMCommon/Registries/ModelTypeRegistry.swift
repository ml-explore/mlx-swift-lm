// Copyright © 2024 Apple Inc.

import Foundation

public actor ModelTypeRegistry<T> {

    private var creators: [String: (Data) throws -> T]

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: (Data) throws -> T]) {
        self.creators = creators
    }

    /// Add a new model to the type registry.
    public func registerModelType(
        _ type: String, creator: @escaping (Data) throws -> T
    ) {
        creators[type] = creator
    }

    /// Given a `modelType` and configuration data instantiate a new `LanguageModel`.
    public func createModel(configuration: Data, modelType: String) throws -> sending T {
        guard let creator = creators[modelType] else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try creator(configuration)
    }

}
