// Copyright © 2026 Apple Inc.

/// Ordered module-path mappings shared by tensors and per-layer quantization settings.
public struct CheckpointNameMapping: Sendable {
    public enum Rule: Sendable {
        case replacePrefix(String, with: String)
        case excludePrefix(String)
        case excludeModule(String)
    }

    public let rules: [Rule]

    /// Prefixes are dot-separated module paths without a trailing dot.
    /// An empty replacement removes a wrapper; an empty source adds one.
    public init(_ rules: [Rule]) {
        self.rules = rules
    }

    public func mapName(_ name: String) -> String? {
        var name = name
        for rule in rules {
            switch rule {
            case .replacePrefix(let source, let destination):
                if let relative = Self.relativeName(name, in: source) {
                    name = Self.join(destination, relative)
                }
            case .excludePrefix(let prefix):
                if Self.relativeName(name, in: prefix) != nil { return nil }
            case .excludeModule(let module):
                if name.split(separator: ".").contains(Substring(module)) { return nil }
            }
        }
        return name
    }

    static func relativeName(_ name: String, in namespace: String) -> String? {
        if namespace.isEmpty { return name }
        if name == namespace { return "" }
        guard name.hasPrefix(namespace + ".") else { return nil }
        return String(name.dropFirst(namespace.count + 1))
    }

    private static func join(_ namespace: String, _ name: String) -> String {
        if namespace.isEmpty { return name }
        if name.isEmpty { return namespace }
        return namespace + "." + name
    }
}

extension ModelCheckpoint {
    /// Apply the same declarative mapping to tensors, provenance, and layer settings.
    public func mapNames(using mapping: CheckpointNameMapping) throws -> Self {
        try mapNames(mapping.mapName)
    }
}
