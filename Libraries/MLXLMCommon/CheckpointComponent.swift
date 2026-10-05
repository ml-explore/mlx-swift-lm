// Copyright © 2026 Apple Inc.

import Foundation

/// A checkpoint component's serialized namespaces and runtime destination.
public struct CheckpointComponent: Sendable {
    public let name: String
    public let namespaces: [String]
    public let destination: String
    public let excludedNamespaces: [String]

    public init(
        name: String, namespaces: [String], destination: String,
        excludedNamespaces: [String] = []
    ) {
        self.name = name
        self.namespaces = namespaces
        self.destination = destination
        self.excludedNamespaces = excludedNamespaces
    }

    public enum Source: Equatable, Sendable {
        case embedded(namespace: String)
        case standalone
    }

    public struct Selection {
        public let checkpoint: ModelCheckpoint
        public let source: Source
    }

    /// Resolve one embedded namespace, or a root component explicitly declared by the caller.
    /// Unknown paths remain in the result for strict model validation.
    public func select(from checkpoint: ModelCheckpoint, standalone: Bool = false) throws
        -> Selection
    {
        let present = Set(namespaces).filter { namespace in
            checkpoint.weights.keys.contains {
                CheckpointNameMapping.relativeName($0, in: namespace) != nil
            }
        }.sorted()
        guard present.count <= 1 else {
            throw SelectionError.ambiguousComponent(name, namespaces: present)
        }
        if let namespace = present.first {
            let exclusions = CheckpointNameMapping(excludedNamespaces.map { .excludePrefix($0) })
            // Select the component before excluding its enclosing sibling namespaces.
            let selected = try checkpoint.mapNames { name in
                if CheckpointNameMapping.relativeName(name, in: namespace) != nil { return name }
                return exclusions.mapName(name)
            }
            let mapping = CheckpointNameMapping([.replacePrefix(namespace, with: destination)])
            return Selection(
                checkpoint: try selected.mapNames(using: mapping),
                source: .embedded(namespace: namespace))
        }
        guard standalone, !checkpoint.weights.isEmpty else {
            throw SelectionError.missingComponent(name)
        }
        let mapping = CheckpointNameMapping([.replacePrefix("", with: destination)])
        return Selection(checkpoint: try checkpoint.mapNames(using: mapping), source: .standalone)
    }

    /// Exclude this component when loading a sibling, including its layer settings.
    public func excluding(from checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        try checkpoint.mapNames(using: .init(namespaces.map { .excludePrefix($0) }))
    }

    public enum SelectionError: LocalizedError {
        case missingComponent(String)
        case ambiguousComponent(String, namespaces: [String])

        public var errorDescription: String? {
            switch self {
            case .missingComponent(let name):
                "Checkpoint does not contain the requested '\(name)' component."
            case .ambiguousComponent(let name, let namespaces):
                "Checkpoint contains competing '\(name)' namespaces: \(namespaces.joined(separator: ", "))."
            }
        }
    }
}
