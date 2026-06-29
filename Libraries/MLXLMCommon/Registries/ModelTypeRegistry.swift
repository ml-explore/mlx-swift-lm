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

    /// Whether a creator is registered for `modelType` — i.e. this registry can
    /// instantiate that architecture. Lets a caller check support without
    /// attempting a (throwing, allocating) `createModel`, e.g. to decide before
    /// a multi-GB download whether a Hub repo's `model_type` is runnable.
    public func contains(_ modelType: String) -> Bool {
        creators[modelType] != nil
    }

}
