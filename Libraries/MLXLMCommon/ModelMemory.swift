// Copyright © 2026 Apple Inc.

import MLX

/// The memory held by a model's tensors, measured from their MLX buffers.
///
/// A buffer shared by several tensors is counted once. It is attributed to the
/// model first, then to on-demand components in name order. A component counts
/// once its load has finished.
///
/// KV caches, temporary arrays, and the tokenizer are not included. For the
/// cache, use ``KVCacheStatus/memoryBytes``.
///
/// ```swift
/// let memory = try await container.memory()
/// print(memory.model, memory.total)
/// for component in memory.components {
///     print(component.kind.name, component.bytes)
/// }
/// ```
public struct ModelMemory: Sendable, Equatable {
    /// The share of buffers attributed to a loaded component.
    public struct Component: Sendable, Equatable {
        /// The loaded model component.
        public let kind: ModelComponent
        /// Bytes attributed to this component after shared buffers are counted once.
        public let bytes: Int
    }

    /// Bytes held by the model's own tensors.
    public let model: Int
    /// Bytes attributed to loaded on-demand components, ordered by component name.
    public let components: [Component]
    /// Total bytes attributed to the model and its loaded components.
    public var total: Int { model + components.reduce(0) { $0 + $1.bytes } }

    public init(model: Int) {
        self.model = model
        self.components = []
    }

    init(model: Int, components: [Component]) {
        self.model = model
        self.components = components
    }
}

/// A type that owns a loaded model and can measure its tensor buffers.
///
/// ``ModelContainer`` and `EmbedderModelContainer` conform, so code that only
/// needs memory can accept either one.
public protocol ModelMemoryReporting: Sendable {
    /// Measure the current model tensor buffers, including loaded on-demand components.
    func memory() async throws -> ModelMemory
}

extension BaseLanguageModel {
    /// Measure the buffers held by the model and its loaded on-demand components.
    ///
    /// Models from the model factories are evaluated before they are returned,
    /// so this succeeds right after loading.
    ///
    /// - Throws: An MLX error if a tensor has never been evaluated.
    public func memory() throws -> ModelMemory {
        try withError {
            var arrays = innerState()
            let model = Memory.bufferSize(arrays)
            var totalBytes = model
            var breakdown = [ModelMemory.Component]()
            if let provider = self as? any ModelComponentsProviding {
                let components = provider.onDemandComponents.sorted {
                    $0.component.name < $1.component.name
                }
                for component in components {
                    guard let tensors = component.loadedTensors() else { continue }
                    arrays.append(contentsOf: tensors)
                    let currentBytes = Memory.bufferSize(arrays)
                    let bytes = currentBytes - totalBytes
                    if let last = breakdown.last, last.kind == component.component {
                        breakdown[breakdown.count - 1] = .init(
                            kind: last.kind, bytes: last.bytes + bytes)
                    } else {
                        breakdown.append(
                            .init(kind: component.component, bytes: bytes))
                    }
                    totalBytes = currentBytes
                }
            }
            return ModelMemory(model: model, components: breakdown)
        }
    }
}
