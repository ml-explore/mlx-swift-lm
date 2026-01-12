// Copyright © 2024 Apple Inc.

import Foundation
import Tokenizers

public actor ProcessorTypeRegistry {

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: (URL, any Tokenizer) throws -> any UserInputProcessor]) {
        self.creators = creators
    }

    private var creators: [String: (URL, any Tokenizer) throws -> any UserInputProcessor]

    /// Add a new model to the type registry.
    public func registerProcessorType(
        _ type: String,
        creator:
            @escaping (
                URL,
                any Tokenizer
            ) throws -> any UserInputProcessor
    ) {
        creators[type] = creator
    }

    /// Given a `processorType` and configuration file instantiate a new `UserInputProcessor`.
    public func createModel(configuration: URL, processorType: String, tokenizer: any Tokenizer)
        throws -> sending any UserInputProcessor
    {
        guard let creator = creators[processorType] else {
            throw ModelFactoryError.unsupportedProcessorType(processorType)
        }
        return try creator(configuration, tokenizer)
    }

}
