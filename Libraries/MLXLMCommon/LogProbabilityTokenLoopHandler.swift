// Copyright © 2026 Apple Inc.

/// Events from a handler composed with ``LogProbabilityTokenLoopHandler``.
public enum LogProbabilityGeneration<Output: Sendable>: Sendable {
    /// An event produced by the wrapped handler.
    case generation(Output)

    /// Log probabilities for one token, emitted before the wrapped handler's events.
    case probability(GenerateTokenLogProbabilities)
}

/// Adds per-token log probabilities to another handler's output.
///
/// Enable reporting with ``GenerateParameters/logProbabilities`` when constructing
/// a ``TokenIterator``. Text chunks can span several tokens; probability events
/// follow the token sequence, including tokens consumed by tool or reasoning parsers.
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
        logProbabilities: GenerateTokenLogProbabilities?,
        emit: (sending Output) -> Bool
    ) -> TokenLoopDisposition {
        if let logProbabilities, !emit(.probability(logProbabilities)) {
            return .cancelled
        }
        return base.onToken(token, logProbabilities: logProbabilities) {
            emit(.generation($0))
        }
    }

    public mutating func onStopToken(
        _ token: Int,
        logProbabilities: GenerateTokenLogProbabilities?,
        emit: (sending Output) -> Bool
    ) -> TokenLoopDisposition {
        if let logProbabilities, !emit(.probability(logProbabilities)) {
            return .cancelled
        }
        return base.onStopToken(token, logProbabilities: logProbabilities) {
            emit(.generation($0))
        }
    }

    public mutating func onGenerationEnd(
        emit: (sending Output) -> Bool
    ) {
        base.onGenerationEnd { emit(.generation($0)) }
    }

    public func infoEvent(_ info: GenerateCompletionInfo) -> Output {
        .generation(base.infoEvent(info))
    }
}
