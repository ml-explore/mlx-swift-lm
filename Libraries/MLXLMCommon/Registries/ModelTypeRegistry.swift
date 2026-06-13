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

    /// Whether a creator is registered for `modelType` — i.e. this registry can
    /// instantiate that architecture. Lets a caller check support without
    /// attempting a (throwing, allocating) `createModel`, e.g. to decide before
    /// a multi-GB download whether a Hub repo's `model_type` is runnable.
    public func contains(_ modelType: String) -> Bool {
        creators[modelType] != nil
    }

}
