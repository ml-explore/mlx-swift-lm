// Copyright © 2025 Apple Inc.

import MLX

/// Chains multiple `LogitProcessor` instances, applying them in order.
///
/// Grammar processors should come first (hard constraints that mask invalid tokens),
/// followed by soft preference processors (repetition penalty, temperature scaling).
///
/// Like the other `LogitProcessor` value types in this module, this is not
/// `Sendable`: `LogitProcessor` has `mutating` requirements and erases potentially
/// stateful conformers, and the generation loop (`TokenIterator`) is itself not
/// `Sendable`, so no concurrency boundary requires it.
public struct CompositeLogitProcessor: LogitProcessor {
    private var processors: [any LogitProcessor]

    public init(_ processors: [any LogitProcessor]) {
        self.processors = processors
    }

    public mutating func prompt(_ prompt: MLXArray) {
        for i in processors.indices {
            processors[i].prompt(prompt)
        }
    }

    public func process(logits: MLXArray) -> MLXArray {
        var result = logits
        for processor in processors {
            result = processor.process(logits: result)
        }
        return result
    }

    public mutating func didSample(token: MLXArray) {
        for i in processors.indices {
            processors[i].didSample(token: token)
        }
    }
}
