// Copyright © 2025 Apple Inc.

import MLX
import MLXGuidedGeneration
import MLXLMCommon
import Testing

// MARK: - Test Processors

/// Adds a constant to all logits. Tracks lifecycle calls.
private struct AddConstantProcessor: LogitProcessor {
    let constant: Float
    var promptCalled = false
    var didSampleCalled = false

    mutating func prompt(_ prompt: MLXArray) {
        promptCalled = true
    }

    func process(logits: MLXArray) -> MLXArray {
        logits + constant
    }

    mutating func didSample(token: MLXArray) {
        didSampleCalled = true
    }
}

/// Multiplies logits by a scalar.
private struct ScaleProcessor: LogitProcessor {
    let scale: Float

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        logits * scale
    }

    mutating func didSample(token: MLXArray) {}
}

/// Tracks all LogitProcessor lifecycle callbacks including didEmit and finalizeGeneration.
private final class LifecycleTrackingProcessor: LogitProcessor {
    var emittedTokens: [Int] = []
    var finalizedCount = 0

    func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray { logits }

    func didSample(token: MLXArray) {}

    func didEmit(token: Int) {
        emittedTokens.append(token)
    }

    func finalizeGeneration() {
        finalizedCount += 1
    }

    func copy() -> LifecycleTrackingProcessor {
        LifecycleTrackingProcessor()
    }
}

// MARK: - Tests

@Suite
struct CompositeLogitProcessorTests {

    @Test
    func singleProcessorPassthrough() {
        let input = MLXArray([1.0, 2.0, 3.0] as [Float])
        let chained = ChainedLogitProcessor(processors: [AddConstantProcessor(constant: 5.0)])

        let result = chained.process(logits: input)
        let values = result.asArray(Float.self)

        #expect(values == [6.0, 7.0, 8.0])
    }

    @Test
    func multipleProcessorsAppliedInOrder() {
        // (original + 1.0) * 2.0
        let input = MLXArray([1.0, 2.0, 3.0] as [Float])
        let chained = ChainedLogitProcessor(processors: [
            AddConstantProcessor(constant: 1.0),
            ScaleProcessor(scale: 2.0),
        ])

        let result = chained.process(logits: input)
        let values = result.asArray(Float.self)

        #expect(values == [4.0, 6.0, 8.0])
    }

    @Test
    func emptyProcessorsReturnsUnmodified() {
        let input = MLXArray([1.0, 2.0, 3.0] as [Float])
        let chained = ChainedLogitProcessor(processors: [])

        let result = chained.process(logits: input)
        let values = result.asArray(Float.self)

        #expect(values == [1.0, 2.0, 3.0])
    }

    @Test
    func promptCallsAllProcessors() {
        var chained = ChainedLogitProcessor(processors: [
            AddConstantProcessor(constant: 1.0),
            AddConstantProcessor(constant: 2.0),
        ])

        chained.prompt(MLXArray([UInt32(1), UInt32(2)]))

        let result = chained.process(logits: MLXArray([0.0] as [Float]))
        let values = result.asArray(Float.self)
        #expect(values == [3.0])
    }

    @Test
    func didSampleCallsAllProcessors() {
        var chained = ChainedLogitProcessor(processors: [
            AddConstantProcessor(constant: 1.0),
            AddConstantProcessor(constant: 2.0),
        ])

        chained.didSample(token: MLXArray(UInt32(42)))

        let result = chained.process(logits: MLXArray([0.0] as [Float]))
        let values = result.asArray(Float.self)
        #expect(values == [3.0])
    }

    @Test
    func processPreservesShape() {
        let input = MLXArray(Array(repeating: Float(1.0), count: 128))
        let chained = ChainedLogitProcessor(processors: [
            AddConstantProcessor(constant: 1.0),
            ScaleProcessor(scale: 0.5),
        ])

        let result = chained.process(logits: input)
        #expect(result.shape == input.shape)
    }

    @Test
    func copyProducesIndependentProcessors() {
        let first = AddConstantProcessor(constant: 1.0)
        let chained = ChainedLogitProcessor(processors: [first])
        var cloned = chained.copy()

        cloned.didSample(token: MLXArray(UInt32(10)))

        let origResult = chained.process(logits: MLXArray([0.0] as [Float]))
        let clonedResult = cloned.process(logits: MLXArray([0.0] as [Float]))
        #expect(origResult.asArray(Float.self) == [1.0])
        #expect(clonedResult.asArray(Float.self) == [1.0])
    }

    @Test
    func removingAllRemovesMatchingNestedProcessors() {
        let chained = ChainedLogitProcessor(processors: [
            AddConstantProcessor(constant: 1.0),
            ChainedLogitProcessor(processors: [
                ScaleProcessor(scale: 2.0),
                AddConstantProcessor(constant: 3.0),
            ]),
        ])

        let remaining = chained.removingAll { $0 is ScaleProcessor }
        let values = remaining?.process(logits: MLXArray([0.0] as [Float])).asArray(Float.self)

        #expect(values == [4.0])
        #expect(chained.removingAll { _ in true } == nil)
    }

    @Test
    @available(*, deprecated)
    func deprecatedCompositeAliasResolvesToChainedAndForwardsLifecycle() {
        let tracker = LifecycleTrackingProcessor()
        var composite: CompositeLogitProcessor = CompositeLogitProcessor([
            AddConstantProcessor(constant: 2.0),
            tracker,
        ])

        composite.didEmit(token: 7)
        composite.finalizeGeneration()

        let values = composite.process(logits: MLXArray([1.0] as [Float])).asArray(Float.self)
        #expect(values == [3.0])
        #expect(tracker.emittedTokens == [7])
        #expect(tracker.finalizedCount == 1)
    }
}
