// Copyright © 2026 Apple Inc.

import Foundation

/// Ordered module-path mappings shared by tensors and per-layer quantization settings.
///
/// Rules run in order, each on the result of the previous one. A pattern rule renames every
/// tensor that matches it, so one row covers all layers:
///
/// ```swift
/// static let names = CheckpointNameMapping([
///     .replacePrefix("model.llm", with: "model.language_model"),
///     .replaceMatches(#"\.attn\.wq\."#, with: ".self_attn.q_proj."),
///     .replaceMatches(#"\.queries\.(\d+)$"#, with: ".layers.$1.query"),
/// ])
/// ```
public struct CheckpointNameMapping: Sendable {
    public enum Rule: Sendable {
        case replacePrefix(String, with: String)
        case excludePrefix(String)
        case excludeModule(String)
        /// Replaces every match of a regular expression with a template. `$1` in the template
        /// is the first capture group. The pattern must be valid. Layer settings are keyed by
        /// module path, so a pattern that ends in a parameter name leaves them alone.
        case replaceMatches(String, with: String)
    }

    public let rules: [Rule]
    private let expressions: [NSRegularExpression?]

    /// Prefixes are dot-separated module paths without a trailing dot.
    /// An empty replacement removes a wrapper; an empty source adds one.
    public init(_ rules: [Rule]) {
        self.rules = rules
        self.expressions = rules.map { rule in
            guard case .replaceMatches(let pattern, _) = rule else { return nil }
            do {
                return try NSRegularExpression(pattern: pattern)
            } catch {
                preconditionFailure("Invalid checkpoint name pattern '\(pattern)': \(error)")
            }
        }
    }

    public func mapName(_ name: String) -> String? {
        var name = name
        for (rule, expression) in zip(rules, expressions) {
            switch rule {
            case .replacePrefix(let source, let destination):
                if let relative = Self.relativeName(name, in: source) {
                    name = Self.join(destination, relative)
                }
            case .excludePrefix(let prefix):
                if Self.relativeName(name, in: prefix) != nil { return nil }
            case .excludeModule(let module):
                if name.split(separator: ".").contains(Substring(module)) { return nil }
            case .replaceMatches(_, let template):
                guard let expression else { continue }
                name = expression.stringByReplacingMatches(
                    in: name, range: NSRange(name.startIndex..., in: name), withTemplate: template)
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
