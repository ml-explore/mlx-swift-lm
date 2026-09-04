// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// A module that can fold projections sharing one input (`gate`/`up`,
/// `q`/`k`/`v`) into a single stacked matmul.
///
/// Stacking concatenates weights along the output axis, so every output row
/// keeps its own accumulation order and quantization groups: results are
/// bitwise identical, only the launch count drops. Speculative decoding
/// feels the difference most, where a round is a few hundred small launches.
protocol SameInputProjectionStacking: Module {
    /// Stack once. Returns false, changing nothing, when the projections are
    /// not plain quantized layers with matching quantization.
    func stackSameInputProjections() -> Bool
}

/// Stack every foldable projection group in `model`; returns the count.
public func stackSameInputProjections(in model: Module) -> Int {
    var stacked = 0
    for module in model.modules() {
        if let stacking = module as? SameInputProjectionStacking,
            stacking.stackSameInputProjections()
        {
            stacked += 1
        }
    }
    if stacked > 0 {
        model.invalidateCompiledTraces()
    }
    return stacked
}

/// `module` when it is exactly a `QuantizedLinear`. Subclasses transform
/// their input before the matmul (a ParoQuant rotation, say), which folding
/// their weights into a plain layer would drop.
func plainQuantizedLinear(_ module: Module) -> QuantizedLinear? {
    type(of: module) == QuantizedLinear.self ? module as? QuantizedLinear : nil
}

/// One layer computing `layers` on a shared input, or nil when they differ in
/// quantization or carry a bias.
func stackedQuantizedLinear(_ layers: [QuantizedLinear]) -> QuantizedLinear? {
    guard let first = layers.first,
        layers.allSatisfy({
            $0.bias == nil && $0.groupSize == first.groupSize && $0.bits == first.bits
                && $0.mode == first.mode
        })
    else { return nil }
    let quantBiases = layers.compactMap(\.biases)
    let biases: MLXArray?
    switch quantBiases.count {
    case layers.count: biases = concatenated(quantBiases, axis: 0)
    case 0: biases = nil
    default: return nil
    }
    let weight = concatenated(layers.map(\.weight), axis: 0)
    let scales = concatenated(layers.map(\.scales), axis: 0)
    eval(weight, scales, biases ?? weight)
    return QuantizedLinear(
        weight: weight, bias: nil, scales: scales, biases: biases,
        groupSize: first.groupSize, bits: first.bits, mode: first.mode)
}

extension Module {
    /// Replace the folded originals with placeholders so their weights free.
    /// Registered `@ModuleInfo` properties only change through `update`.
    func releaseStackedProjections(_ keys: [String]) {
        update(
            modules: ModuleChildren.unflattened(
                keys.map { ($0, Linear(weight: MLXArray.zeros([1, 1]))) }))
    }
}
