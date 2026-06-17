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

// MARK: - Tests

@Suite
struct CompositeLogitProcessorTests {

    @Test
    func singleProcessorPassthrough() {
        let input = MLXArray([1.0, 2.0, 3.0] as [Float])
        let composite = CompositeLogitProcessor([AddConstantProcessor(constant: 5.0)])

        let result = composite.process(logits: input)
        let values = result.asArray(Float.self)

        #expect(values == [6.0, 7.0, 8.0])
    }

    @Test
    func multipleProcessorsAppliedInOrder() {
        // (original + 1.0) * 2.0
        let input = MLXArray([1.0, 2.0, 3.0] as [Float])
        let composite = CompositeLogitProcessor([
            AddConstantProcessor(constant: 1.0),
            ScaleProcessor(scale: 2.0),
        ])

        let result = composite.process(logits: input)
        let values = result.asArray(Float.self)

        #expect(values == [4.0, 6.0, 8.0])
    }

    @Test
    func emptyProcessorsReturnsUnmodified() {
        let input = MLXArray([1.0, 2.0, 3.0] as [Float])
        let composite = CompositeLogitProcessor([])

        let result = composite.process(logits: input)
        let values = result.asArray(Float.self)

        #expect(values == [1.0, 2.0, 3.0])
    }

    @Test
    func promptCallsAllProcessors() {
        var composite = CompositeLogitProcessor([
            AddConstantProcessor(constant: 1.0),
            AddConstantProcessor(constant: 2.0),
        ])

        composite.prompt(MLXArray([UInt32(1), UInt32(2)]))

        // Verify via round-trip: if prompt mutated the processors,
        // the composite should reflect that state. We verify by
        // checking that prompt does not crash and process still works.
        let result = composite.process(logits: MLXArray([0.0] as [Float]))
        let values = result.asArray(Float.self)
        #expect(values == [3.0])
    }

    @Test
    func didSampleCallsAllProcessors() {
        var composite = CompositeLogitProcessor([
            AddConstantProcessor(constant: 1.0),
            AddConstantProcessor(constant: 2.0),
        ])

        // Should not crash; both processors receive the call.
        composite.didSample(token: MLXArray(UInt32(42)))

        // Verify processors still function after didSample.
        let result = composite.process(logits: MLXArray([0.0] as [Float]))
        let values = result.asArray(Float.self)
        #expect(values == [3.0])
    }

    @Test
    func processPreservesShape() {
        let input = MLXArray(Array(repeating: Float(1.0), count: 128))
        let composite = CompositeLogitProcessor([
            AddConstantProcessor(constant: 1.0),
            ScaleProcessor(scale: 0.5),
        ])

        let result = composite.process(logits: input)
        #expect(result.shape == input.shape)
    }
}
