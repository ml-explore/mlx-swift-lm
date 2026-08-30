import Foundation
import MLX

/// One static-shaped piece of a hybrid model's single-token decode graph.
///
/// Full-attention cache writes split segments because the cache grows at every
/// token and cannot be mutated inside an MLX compiled function. Linear-layer
/// state is passed explicitly through the compiled function instead.
package struct CompiledDecodeSegment: Sendable, Equatable {
    package var attentionPostLayer: Int?
    package var linearLayers: [Int]
    package var attentionPreLayer: Int?

    package init(
        attentionPostLayer: Int? = nil,
        linearLayers: [Int] = [],
        attentionPreLayer: Int? = nil
    ) {
        self.attentionPostLayer = attentionPostLayer
        self.linearLayers = linearLayers
        self.attentionPreLayer = attentionPreLayer
    }

    /// First linear-state input, after the hidden state and an optional
    /// `[attention, gate]` pair.
    package var stateInputOffset: Int { attentionPostLayer == nil ? 1 : 3 }

    /// First `[queries, gate, keys, values]` output after the hidden state and
    /// two updated state tensors per linear layer.
    package var attentionOutputOffset: Int { 1 + 2 * linearLayers.count }

    /// Build the maximal segments separated by full-attention cache writes.
    package static func schedule(linearLayers: [Bool]) -> [Self] {
        var segments: [Self] = []
        var current = Self()
        for (index, isLinear) in linearLayers.enumerated() {
            if isLinear {
                current.linearLayers.append(index)
            } else {
                current.attentionPreLayer = index
                segments.append(current)
                current = Self(attentionPostLayer: index)
            }
        }
        segments.append(current)
        return segments
    }
}

/// Lazily compiles and retains one MLX function for every decode segment.
///
/// Models create this after their schedule is known. Compilation remains lazy
/// because model weights are loaded after initialization. The lock ensures two
/// concurrent first decode calls cannot install different traces.
package final class CompiledDecodeSegmentCache {
    private let lock = NSLock()
    private var functions: [(([MLXArray]) -> [MLXArray])?]

    package init(count: Int) {
        precondition(count > 0, "compiled decode requires at least one segment")
        self.functions = Array(repeating: nil, count: count)
    }

    package var compiledCount: Int {
        lock.withLock { functions.compactMap { $0 }.count }
    }

    package func call(
        at index: Int,
        arguments: [MLXArray],
        body: @escaping ([MLXArray]) -> [MLXArray]
    ) -> [MLXArray] {
        let function = lock.withLock {
            if let function = functions[index] {
                return function
            }
            let function = compile(body)
            functions[index] = function
            return function
        }
        return function(arguments)
    }
}
