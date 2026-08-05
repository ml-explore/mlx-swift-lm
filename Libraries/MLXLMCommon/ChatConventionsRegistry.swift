// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - ChatConventionsResolving

/// Resolves a model's chat conventions from identity signals the model instance
/// cannot see for itself, today the Hugging Face repo id.
///
/// ``ChatConventionsProviding`` covers everything a model knows about itself. Some
/// conventions are not model-intrinsic: DeepSeek-R1-Distill checkpoints report their
/// *base* `model_type` (`qwen2` or `llama`) and are indistinguishable from plain
/// Qwen2.5/Llama by the model alone, so they can only be recognized by repo id.
///
/// Register a resolver with ``ChatConventionsRegistry`` to supply such conventions,
/// including from outside this package. Both requirements are optional; implement
/// only the one that applies.
public protocol ChatConventionsResolving: Sendable {
    /// The tool-call format for this model id / type, or `nil` to defer.
    func toolCallFormat(modelId: String, modelType: String) -> ToolCallFormat?

    /// The reasoning protocol for this model id / type, or `nil` to defer.
    func reasoningConfig(modelId: String, modelType: String) -> ReasoningConfig?
}

extension ChatConventionsResolving {
    public func toolCallFormat(modelId: String, modelType: String) -> ToolCallFormat? { nil }
    public func reasoningConfig(modelId: String, modelType: String) -> ReasoningConfig? { nil }
}

// MARK: - ChatConventionsRegistry

/// Ordered registry of ``ChatConventionsResolving`` consulted by the model factories.
///
/// Resolution precedence when a model is loaded:
///
/// 1. an explicit value already set on the `ModelConfiguration` (registry entry or caller),
/// 2. a registered resolver here,
/// 3. the model's own ``ChatConventionsProviding`` declaration.
///
/// Resolvers sit above the model because repo-id knowledge is more specific than
/// model-intrinsic knowledge, and so that a downstream package can correct a
/// built-in declaration without constructing a `ModelConfiguration`.
///
/// Most recently registered wins: ``register(_:)`` appends, and lookup walks the
/// resolvers in reverse registration order, returning the first non-`nil` answer.
public final class ChatConventionsRegistry: @unchecked Sendable {

    /// Shared instance, preloaded with this package's built-in resolvers.
    public static let shared = ChatConventionsRegistry(
        resolvers: [DeepSeekR1ConventionsResolver()])

    private let lock = NSLock()
    private var resolvers: [any ChatConventionsResolving]

    public init(resolvers: [any ChatConventionsResolving] = []) {
        self.resolvers = resolvers
    }

    /// Add a resolver. It takes precedence over everything registered before it.
    public func register(_ resolver: any ChatConventionsResolving) {
        lock.withLock { resolvers.append(resolver) }
    }

    /// First tool-call format offered by a resolver, in reverse registration order.
    public func toolCallFormat(modelId: String, modelType: String) -> ToolCallFormat? {
        snapshot().lazy
            .compactMap { $0.toolCallFormat(modelId: modelId, modelType: modelType) }
            .first
    }

    /// First reasoning config offered by a resolver, in reverse registration order.
    public func reasoningConfig(modelId: String, modelType: String) -> ReasoningConfig? {
        snapshot().lazy
            .compactMap { $0.reasoningConfig(modelId: modelId, modelType: modelType) }
            .first
    }

    /// Resolvers in precedence order (most recently registered first).
    ///
    /// Copied out under the lock so resolvers run *outside* it: they can be
    /// implemented outside this package, so invoking one while holding a
    /// non-recursive lock would deadlock if it re-entered the registry, and would
    /// hold the registry closed for the duration of arbitrary work.
    private func snapshot() -> [any ChatConventionsResolving] {
        lock.withLock { resolvers.reversed() }
    }
}

// MARK: - Built-in resolvers

/// Recognizes DeepSeek-R1 and its distills by repo id.
///
/// Two reasons this cannot be a model declaration:
///
/// - R1-Distill checkpoints report a base `model_type` (`qwen2` / `llama`), which is
///   indistinguishable from plain Qwen2.5 / Llama.
/// - R1 proper reports `deepseek_v3`, an architecture it shares with DeepSeek-V3.
///   Always-on reasoning belongs to the R1 checkpoints and their chat template, not
///   to that architecture, so declaring it on `DeepseekV3Model` would wrongly
///   advertise reasoning for plain V3.
public struct DeepSeekR1ConventionsResolver: ChatConventionsResolving {

    public init() {}

    public func reasoningConfig(modelId: String, modelType: String) -> ReasoningConfig? {
        let id = modelId.lowercased()
        // "r1-distill" also catches re-uploads not prefixed "deepseek-".
        guard id.contains("deepseek-r1") || id.contains("r1-distill") else { return nil }
        return .alwaysOnThinking
    }
}
