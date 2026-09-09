import Foundation
import MLX
import MLXNN

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

    /// Every decoder layer this segment runs, in order. The weights of these
    /// layers are the compile state of the segment's trace.
    package var layerIndices: [Int] {
        (attentionPostLayer.map { [$0] } ?? []) + linearLayers
            + (attentionPreLayer.map { [$0] } ?? [])
    }

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

/// Lazily compiles and retains one `CompiledTrace` per decode segment.
///
/// Models create this with their schedule; compilation stays lazy because
/// weights load after initialization. Each segment declares the modules its
/// body reads, so the trace sees current weights rather than the ones it was
/// first traced with.
package final class CompiledDecodeSegmentCache<Owner: Module>: CompiledTraceInvalidating {

    /// Runs segment `index` over the flat argument list.
    package typealias Body = @Sendable (Owner, Int, [MLXArray]) -> [MLXArray]

    /// Returns the modules whose weights segment `index` reads: the layers in
    /// `layerIndices`, plus anything else the body touches such as the
    /// embedding or the final norm.
    package typealias StateProvider = @Sendable (Owner, Int) -> [Module]

    private let traces: [CompiledTrace<Owner>]

    package init(count: Int, state: @escaping StateProvider, body: @escaping Body) {
        precondition(count > 0, "compiled decode requires at least one segment")
        self.traces = (0 ..< count).map { index in
            CompiledTrace<Owner>(
                state: { state($0, index) },
                body: { owner, arguments in body(owner, index, arguments) })
        }
    }

    package var compiledCount: Int {
        traces.filter(\.isCompiled).count
    }

    package func callAsFunction(
        _ owner: Owner, at index: Int, _ arguments: [MLXArray]
    ) -> [MLXArray] {
        traces[index](owner, arguments)
    }

    /// Drops every segment trace. See `CompiledTrace.invalidate()`.
    package func invalidate() {
        traces.forEach { $0.invalidate() }
    }
}
