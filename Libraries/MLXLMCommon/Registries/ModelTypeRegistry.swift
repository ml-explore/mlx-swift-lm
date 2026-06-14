// Copyright © 2024 Apple Inc.

import Foundation

public actor ModelTypeRegistry<T> {

    private struct CreatorEntry {
        var matches: (Data) -> Bool
        var create: (Data) throws -> T
    }

    private var creators: [String: [CreatorEntry]]

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: (Data) throws -> T]) {
        self.creators = creators.mapValues { creator in
            [CreatorEntry(matches: { _ in true }, create: creator)]
        }
    }

    /// Add a new model to the type registry.
    public func registerModelType(
        _ type: String, creator: @escaping (Data) throws -> T
    ) {
        creators[type] = [CreatorEntry(matches: { _ in true }, create: creator)]
    }

    /// Add a new model to the type registry with a predicate for ambiguous
    /// `model_type` strings shared by multiple package targets.
    public func registerModelType(
        _ type: String,
        matches: @escaping (Data) -> Bool,
        creator: @escaping (Data) throws -> T
    ) {
        creators[type, default: []].append(CreatorEntry(matches: matches, create: creator))
    }

    /// Given a `modelType` and configuration data instantiate a new `LanguageModel`.
    public func createModel(configuration: Data, modelType: String) throws -> sending T {
        guard let entries = creators[modelType] else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        for entry in entries where entry.matches(configuration) {
            return try entry.create(configuration)
        }
        throw ModelFactoryError.unsupportedModelType(modelType)
    }

}
