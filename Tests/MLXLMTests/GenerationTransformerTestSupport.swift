// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

final class TransformerFactoryCount: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    var value: Int { lock.withLock { count } }
    func increment() { lock.withLock { count += 1 } }
}

enum TokenCountingEvent: GenerationProviding {
    case output(Generation, tokenCount: Int)
    case token(Int)

    var generation: Generation? {
        if case .output(let generation, _) = self { generation } else { nil }
    }
}

struct TokenCountingTransformer: GenerationTransformer {
    let factories: TransformerFactoryCount
    var maxTokens: Int? = nil

    func configure(_ parameters: inout GenerateParameters) {
        if let maxTokens { parameters.maxTokens = maxTokens }
    }

    func makeHandler(wrapping base: TextToolTokenLoopHandler) -> TokenCountingHandler {
        factories.increment()
        return TokenCountingHandler(base: base)
    }
}

struct TokenCountingHandler: TokenLoopHandler {
    var base: TextToolTokenLoopHandler
    var count = 0

    var additionalStopTokenIDs: Set<Int> { base.additionalStopTokenIDs }
    var receivesStopTokens: Bool { base.receivesStopTokens }

    mutating func onToken(
        _ token: Int, logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending TokenCountingEvent) -> Bool
    ) -> TokenLoopDisposition {
        count += 1
        guard emit(.token(count)) else { return .cancelled }
        let count = count
        return base.onToken(token, logProbabilities: logProbabilities) {
            emit(.output($0, tokenCount: count))
        }
    }

    mutating func onStopToken(
        _ token: Int, logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending TokenCountingEvent) -> Bool
    ) -> TokenLoopDisposition {
        let count = count
        return base.onStopToken(token, logProbabilities: logProbabilities) {
            emit(.output($0, tokenCount: count))
        }
    }

    mutating func onGenerationEnd(
        emit: (sending TokenCountingEvent) -> Bool
    ) -> TokenLoopDisposition {
        let count = count
        return base.onGenerationEnd { emit(.output($0, tokenCount: count)) }
    }

    func infoEvent(_ info: GenerateCompletionInfo) -> TokenCountingEvent {
        .output(base.infoEvent(info), tokenCount: count)
    }
}
