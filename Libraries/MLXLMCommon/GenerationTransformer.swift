// Copyright © 2026 Apple Inc.

/// Provides the base event that a chat session uses to maintain its state.
public protocol GenerationProviding: Sendable {
    /// A decoded event or completion; `nil` for an observation-only event.
    var generation: Generation? { get }
}

extension Generation: GenerationProviding {
    public var generation: Generation? { self }
}

extension LogProbabilityGeneration: GenerationProviding where Output: GenerationProviding {
    public var generation: Generation? {
        switch self {
        case .generation(let output): output.generation
        case .probability: nil
        }
    }
}

/// Configures a chat request and composes its per-pass token handlers.
///
/// A session may run several generation passes when it dispatches tools. Keep
/// request configuration here and decoding state in the handler created for each pass.
public protocol GenerationTransformer: Sendable {
    associatedtype Handler: TokenLoopHandler where Handler.Output: GenerationProviding

    /// Configures a request-local copy of the session's generation parameters.
    func configure(_ parameters: inout GenerateParameters)

    /// Creates a fresh handler on each pass, including automatic tool restarts.
    ///
    /// Forward every base event exactly once and in order, exposing it through
    /// ``GenerationProviding/generation``. Preserve the base's stop-token properties,
    /// dispositions, end-of-generation output, and completion metadata.
    /// Automatically dispatched tool-call events are consumed by the session.
    func makeHandler(wrapping base: TextToolTokenLoopHandler) -> Handler
}

extension GenerationTransformer {
    public func configure(_ parameters: inout GenerateParameters) {}
}

/// Preserves the session's standard decoded output and generation parameters.
public struct TextGenerationTransformer: GenerationTransformer {
    public init() {}

    public func makeHandler(wrapping base: TextToolTokenLoopHandler) -> TextToolTokenLoopHandler {
        base
    }
}

/// Adds ordered per-token log probabilities to a chat request.
public struct LogProbabilityGenerationTransformer: GenerationTransformer {
    /// Number of top candidates. Values at or below zero report only the chosen token.
    public var topCount: Int

    public init(topCount: Int = 0) {
        self.topCount = topCount
    }

    public func configure(_ parameters: inout GenerateParameters) {
        parameters.logProbabilities = topCount
    }

    public func makeHandler(wrapping base: TextToolTokenLoopHandler)
        -> LogProbabilityTokenLoopHandler<TextToolTokenLoopHandler>
    {
        LogProbabilityTokenLoopHandler(base)
    }
}

extension GenerationTransformer where Self == LogProbabilityGenerationTransformer {
    /// Observe chosen-token log probabilities and, optionally, the top candidates.
    public static func logProbabilities(topCount: Int = 0) -> Self {
        .init(topCount: topCount)
    }
}

extension GenerationTransformer where Self == TextGenerationTransformer {
    /// Use the standard text/tool handler without changing generation parameters.
    public static var text: Self { .init() }
}
