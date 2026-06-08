// Copyright © 2025 Apple Inc.

import MLX

/// Chains multiple `LogitProcessor` instances, applying them in order.
///
/// Grammar processors should come first (hard constraints that mask invalid tokens),
/// followed by soft preference processors (repetition penalty, temperature scaling).
///
/// Thread safety: marked `@unchecked Sendable` because all access is serialized
/// through `ModelContainer.perform`.
public struct CompositeLogitProcessor: LogitProcessor, @unchecked Sendable {
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
