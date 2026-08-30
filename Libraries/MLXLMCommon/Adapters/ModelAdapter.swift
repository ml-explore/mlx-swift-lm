//
//  ModelAdapters.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 02.06.2025.
//

import Foundation
import MLX
import MLXNN

/// Errors that can occur when working with a `ModelAdapter`.
public enum ModelAdapterError: Error {
    case unsupportedAdapterType(String)
    case incompatibleModelType
}

/// Protocol defining an adapter that can modify a `LanguageModel`.
public protocol ModelAdapter: Sendable {

    /// Loads the adapter into the specified model.
    ///
    /// Prefer ``LanguageModel/load(adapter:)`` when invoking an adapter so the
    /// model can rebuild derived inference state after the topology change.
    func load(into model: LanguageModel) throws

    /// Permanently fuses the adapter into the specified model.
    ///
    /// Prefer ``LanguageModel/fuse(with:)`` when invoking an adapter so the
    /// model can rebuild derived inference state after the topology change.
    func fuse(with model: LanguageModel) throws

    /// Unloads the adapter from the specified model.
    ///
    /// Prefer ``LanguageModel/unload(adapter:)`` when invoking an adapter so
    /// the model can rebuild derived inference state after the topology change.
    func unload(from model: LanguageModel)
}

public typealias SendableModelAdapter = ModelAdapter & Sendable

/// Extension to `LanguageModel` providing convenience methods for adapter usage.
extension LanguageModel {

    /// Loads an adapter into the model.
    ///
    /// Example:
    /// ```swift
    /// let model: any LanguageModel = ...
    /// let adapter: any ModelAdapter = ...
    /// try model.load(adapter: adapter)
    /// ```
    public func load(adapter: ModelAdapter) throws {
        defer { prepareInferenceState(in: self) }
        try adapter.load(into: self)
    }

    /// Fuses an adapter permanently into the model.
    ///
    /// Example:
    /// ```swift
    /// let model: any LanguageModel = ...
    /// let adapter: any ModelAdapter = ...
    /// try model.fuse(with: adapter)
    /// ```
    public func fuse(with adapter: ModelAdapter) throws {
        defer { prepareInferenceState(in: self) }
        try adapter.fuse(with: self)
    }

    /// Unloads an adapter from the model.
    ///
    /// Example:
    /// ```swift
    /// let model: any LanguageModel = ...
    /// let adapter: any ModelAdapter = ...
    /// model.unload(adapter: adapter)
    /// ```
    public func unload(adapter: ModelAdapter) {
        adapter.unload(from: self)
        prepareInferenceState(in: self)
    }

    /// Temporarily loads an adapter, performs a synchronous action, then unloads the adapter.
    ///
    /// Example:
    /// ```swift
    /// let model: any LanguageModel = ...
    /// let adapter: any ModelAdapter = ...
    /// try model.perform(with: adapter) {
    ///     generate(inputs: ...)
    /// }
    /// // Adapter is automatically unloaded after execution
    /// ```
    public func perform<R>(
        with adapter: ModelAdapter, perform: () throws -> R
    ) throws -> R {
        defer {
            adapter.unload(from: self)
            prepareInferenceState(in: self)
        }
        try adapter.load(into: self)
        prepareInferenceState(in: self)
        let result = try perform()
        return result
    }

    /// Temporarily loads an adapter, performs an asynchronous action, then unloads the adapter.
    ///
    /// Example:
    /// ```swift
    /// let model: any LanguageModel = ...
    /// let adapter: any ModelAdapter = ...
    /// try await model.perform(with: adapter) {
    ///     await generate(inputs: ...)
    /// }
    /// // Adapter is automatically unloaded after execution
    /// ```
    public func perform<R>(
        with adapter: ModelAdapter, perform: () async throws -> R
    ) async throws -> R {
        defer {
            adapter.unload(from: self)
            prepareInferenceState(in: self)
        }
        try adapter.load(into: self)
        prepareInferenceState(in: self)
        let result = try await perform()
        return result
    }
}
