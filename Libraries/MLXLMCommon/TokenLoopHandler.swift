// Copyright © 2026 Apple Inc.

/// Controls whether the generation loop continues.
public enum TokenLoopDisposition: Sendable {
    /// Continue generating tokens.
    case more
    /// Stop at a semantic response boundary.
    case stop
    /// Stop because the consumer ended or cancelled.
    case cancelled

    var shouldContinue: Bool {
        if case .more = self { return true }
        return false
    }
}

/// Converts generated tokens into stream events.
///
/// Handlers run on the generation worker and may own non-Sendable decoding state.
/// Only their output crosses into the consumer's task.
/// The `emit` closure returns `false` when the consumer has ended the stream.
public protocol TokenLoopHandler: SendableMetatype {
    associatedtype Output: Sendable

    /// Semantic boundaries contributed by the response protocol handled by
    /// this consumer. Raw-token consumers intentionally contribute none.
    var additionalStopTokenIDs: Set<Int> { get }

    /// Whether semantic parsing needs to observe EOS tokens even though they
    /// are not included in the public output or generation token count.
    var receivesStopTokens: Bool { get }

    /// Return `.stop` for semantic generation stops, or `.cancelled` for consumer termination.
    mutating func onToken(
        _ token: Int,
        logProbabilities: GenerateTokenLogProbabilities?,
        emit: (sending Output) -> Bool
    ) -> TokenLoopDisposition

    /// Called when `includeStopToken` is true or ``receivesStopTokens`` is true
    /// and a stop token was hit.
    mutating func onStopToken(
        _ token: Int,
        logProbabilities: GenerateTokenLogProbabilities?,
        emit: (sending Output) -> Bool
    ) -> TokenLoopDisposition

    /// Called after the token loop finishes, before the info event.
    mutating func onGenerationEnd(
        emit: (sending Output) -> Bool
    )

    func infoEvent(_ info: GenerateCompletionInfo) -> Output
}

extension TokenLoopHandler {
    public var additionalStopTokenIDs: Set<Int> { [] }
    public var receivesStopTokens: Bool { false }
}

/// Decodes response text and tool calls into ``Generation`` events.
public struct TextToolTokenLoopHandler: TokenLoopHandler {
    public typealias Output = Generation

    private static let logger = Logger(
        subsystem: "mlx-swift-lm", category: "TokenStreamProtocol")
    private var decoder: any TokenStreamDecoder

    public init(
        tokenizer: Tokenizer, stopStrings: Set<String> = [], format: ToolCallFormat,
        tools: [[String: any Sendable]]? = nil,
        toolCallPolicy: ToolCallPolicy = .init()
    ) {
        self.decoder = format.makeTokenStreamDecoder(
            tokenizer: tokenizer, tools: tools, stopStrings: stopStrings,
            toolCallPolicy: toolCallPolicy)
    }

    public var additionalStopTokenIDs: Set<Int> { decoder.additionalStopTokenIDs }
    public var receivesStopTokens: Bool { decoder.receivesStopTokens }

    public mutating func onToken(
        _ token: Int,
        logProbabilities: GenerateTokenLogProbabilities?,
        emit: (sending Generation) -> Bool
    ) -> TokenLoopDisposition {
        process(token, emit: emit)
    }

    public mutating func onStopToken(
        _ token: Int,
        logProbabilities: GenerateTokenLogProbabilities?,
        emit: (sending Generation) -> Bool
    ) -> TokenLoopDisposition {
        guard decoder.receivesStopTokens else { return .more }
        return process(token, emit: emit)
    }

    public mutating func onGenerationEnd(
        emit: (sending Generation) -> Bool
    ) {
        var decoder = self.decoder
        var disposition = TokenLoopDisposition.more
        _ = decoder.finish { event in
            disposition = process(event, emit: emit)
            return disposition.shouldContinue
        }
        self.decoder = decoder
    }

    public func infoEvent(_ info: GenerateCompletionInfo) -> Generation {
        .info(
            info.withToolCallCounts(
                rejected: decoder.rejectedToolCallCount,
                recovered: decoder.recoveredToolCallCount))
    }

    private mutating func process(
        _ token: Int,
        emit: (sending Generation) -> Bool
    ) -> TokenLoopDisposition {
        var decoder = self.decoder
        var disposition = TokenLoopDisposition.more
        let completed = decoder.push(token) { event in
            disposition = process(event, emit: emit)
            return disposition.shouldContinue
        }
        self.decoder = decoder

        if disposition.shouldContinue {
            return completed ? .more : .cancelled
        }
        return disposition
    }

    private mutating func process(
        _ event: TokenStreamEvent,
        emit: (sending Generation) -> Bool
    ) -> TokenLoopDisposition {
        switch event {
        case .reasoning:
            // The public Generation stream intentionally exposes only response
            // text and tool calls. Protocol-aware clients consume reasoning via
            // the package-level TokenStreamDecoder contract.
            return .more

        case .response(let response):
            if !emit(.chunk(response)) {
                return .cancelled
            }
            return .more

        case .toolCall(let toolCall):
            if !emit(.toolCall(toolCall)) {
                return .cancelled
            }
            return .more

        case .protocolError(let message):
            Self.logger.error("\(message)")
            return .more

        case .rejectedToolCall(let rejection):
            if !emit(.rejectedToolCall(rejection)) {
                return .cancelled
            }
            return .more

        case .stop:
            return .stop
        }
    }
}

/// Emits raw token IDs and completion information as ``TokenGeneration`` events.
public struct RawTokenLoopHandler: TokenLoopHandler {
    public typealias Output = TokenGeneration

    public let additionalStopTokenIDs: Set<Int>
    public let receivesStopTokens: Bool

    public init(additionalStopTokenIDs: Set<Int> = [], receivesStopTokens: Bool = false) {
        self.additionalStopTokenIDs = additionalStopTokenIDs
        self.receivesStopTokens = receivesStopTokens
    }

    public mutating func onToken(
        _ token: Int,
        logProbabilities: GenerateTokenLogProbabilities?,
        emit: (sending TokenGeneration) -> Bool
    ) -> TokenLoopDisposition {
        if !emit(.token(token)) {
            return .cancelled
        }
        return .more
    }

    public mutating func onStopToken(
        _ token: Int,
        logProbabilities: GenerateTokenLogProbabilities?,
        emit: (sending TokenGeneration) -> Bool
    ) -> TokenLoopDisposition {
        if !emit(.token(token)) {
            return .cancelled
        }
        return .more
    }

    public mutating func onGenerationEnd(
        emit: (sending TokenGeneration) -> Bool
    ) {}

    public func infoEvent(_ info: GenerateCompletionInfo) -> TokenGeneration {
        .info(info)
    }
}
