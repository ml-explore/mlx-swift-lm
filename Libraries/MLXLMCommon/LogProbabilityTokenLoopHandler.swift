// Copyright © 2026 Apple Inc.

/// Events from a handler composed with ``LogProbabilityTokenLoopHandler``.
public enum LogProbabilityGeneration<Output: Sendable>: Sendable {
    /// An event produced by the wrapped handler.
    case generation(Output)

    /// Log probabilities for one token, emitted before the wrapped handler's events.
    case probability(GenerateTokenLogProbabilities)

    /// One token's log probabilities; `nil` for an event from the wrapped handler.
    public var logProbabilities: GenerateTokenLogProbabilities? {
        if case .probability(let values) = self { values } else { nil }
    }
}

/// Adds per-token log probabilities to another handler's output.
///
/// Enable reporting with ``GenerateParameters/logProbabilities`` when constructing
/// a ``TokenIterator``. Text chunks can span several tokens; probability events
/// follow the token sequence, including tokens consumed by tool or reasoning parsers.
/// This handler materializes the values, so only streams that use it copy them from the GPU.
public struct LogProbabilityTokenLoopHandler<Base: TokenLoopHandler>: TokenLoopHandler {
    public typealias Output = LogProbabilityGeneration<Base.Output>

    private var base: Base

    public init(_ base: consuming Base) {
        self.base = base
    }

    public var additionalStopTokenIDs: Set<Int> { base.additionalStopTokenIDs }
    public var receivesStopTokens: Bool { base.receivesStopTokens }

    public mutating func onToken(
        _ token: Int,
        logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending Output) -> Bool
    ) -> TokenLoopDisposition {
        if let logProbabilities, !emit(.probability(logProbabilities.materialize())) {
            return .cancelled
        }
        return base.onToken(token, logProbabilities: logProbabilities) {
            emit(.generation($0))
        }
    }

    public mutating func onStopToken(
        _ token: Int,
        logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending Output) -> Bool
    ) -> TokenLoopDisposition {
        if let logProbabilities, !emit(.probability(logProbabilities.materialize())) {
            return .cancelled
        }
        return base.onStopToken(token, logProbabilities: logProbabilities) {
            emit(.generation($0))
        }
    }

    public mutating func onGenerationEnd(
        emit: (sending Output) -> Bool
    ) -> TokenLoopDisposition {
        base.onGenerationEnd { emit(.generation($0)) }
    }

    public func infoEvent(_ info: GenerateCompletionInfo) -> Output {
        .generation(base.infoEvent(info))
    }
}
