// Copyright © 2026 Apple Inc.
//
// MLX traces a compiled function once: every array the body reads without
// receiving it as an argument or declaring it as `inputs:` state becomes a
// constant of the trace. These tests pin that behaviour for plain `compile`
// (the hazard) and pin that `CompiledTrace` does not have it (the fix).

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

private final class ScaleLayer: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ value: Float) {
        self._weight.wrappedValue = MLXArray([value])
        super.init()
    }

    func forward(_ x: MLXArray) -> MLXArray {
        x * weight
    }

    let compiledForward = CompiledTrace<ScaleLayer> { layer, arguments in
        [layer.forward(arguments[0])]
    }
}

private final class ScaleContainer: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    override init() {
        _linear.wrappedValue = Linear(1, 1, bias: false)
        super.init()
    }

    func forward(_ x: MLXArray) -> MLXArray {
        linear(x)
    }

    let compiledForward = CompiledTrace<ScaleContainer> { container, arguments in
        [container.forward(arguments[0])]
    }
}

/// Traces held in a collection, like the per-segment traces a model keeps.
private final class ScaleBank: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let traces: [CompiledTrace<ScaleBank>]

    init(_ value: Float) {
        self._weight.wrappedValue = MLXArray([value])
        self.traces = (0 ..< 2).map { _ in
            CompiledTrace<ScaleBank> { bank, arguments in [arguments[0] * bank.weight] }
        }
        super.init()
    }
}

private final class ScaleParent: Module {
    @ModuleInfo(key: "child") var child: ScaleLayer

    init(_ child: ScaleLayer) {
        _child.wrappedValue = child
        super.init()
    }
}

final class CompiledTraceTests: XCTestCase {

    private func setWeight(_ layer: ScaleLayer, _ value: Float) {
        layer.update(parameters: ModuleParameters.unflattened(["weight": MLXArray([value])]))
    }

    private func value(_ array: MLXArray) -> Float {
        eval(array)
        return array.item(Float.self)
    }

    /// The hazard, straight from MLX: a body that reads weights through a
    /// capture keeps the values of its first call. Reproducing it takes
    /// `MLX.compile` by name, since the `compile` this package exposes rejects
    /// the capture at build time. A failure here means MLX changed its capture
    /// rules and `CompiledTrace` can be revisited.
    func testPlainCompileFreezesCapturedWeights() {
        let layer = ScaleLayer(2)
        let captured = MLX.compile { [unowned layer] (x: MLXArray) in layer.forward(x) }

        XCTAssertEqual(value(captured(MLXArray([Float(1)]))), 2)

        setWeight(layer, 5)
        XCTAssertEqual(value(layer.forward(MLXArray([Float(1)]))), 5)
        XCTAssertEqual(
            value(captured(MLXArray([Float(1)]))), 2,
            "plain compile baked the captured weight into the trace")
    }

    /// The same body through ``CompiledTrace``: weights are compile `inputs:`,
    /// so every call reads their current values.
    func testCompiledTraceSeesWeightUpdates() {
        let layer = ScaleLayer(2)

        XCTAssertEqual(value(layer.compiledForward(layer, MLXArray([Float(1)]))), 2)

        setWeight(layer, 5)
        XCTAssertEqual(value(layer.compiledForward(layer, MLXArray([Float(1)]))), 5)

        setWeight(layer, -3)
        XCTAssertEqual(value(layer.compiledForward(layer, MLXArray([Float(1)]))), -3)
    }

    /// Gradients flow to the declared state, which is what training a LoRA
    /// adapter through a compiled forward needs.
    func testCompiledTraceIsDifferentiableThroughWeights() throws {
        let layer = ScaleLayer(2)
        let x = MLXArray([Float(3)])

        let gradients = valueAndGrad(model: layer) { (layer: ScaleLayer, x: MLXArray) in
            [layer.compiledForward(layer, x).sum()]
        }(layer, x).1
        let flattened = Dictionary(uniqueKeysWithValues: gradients.flattened())

        XCTAssertEqual(value(try XCTUnwrap(flattened["weight"])), 3)
    }

    /// A trace is stored on the module it traces, so it must not retain it.
    func testCompiledTraceDoesNotRetainItsOwner() {
        var layer: ScaleLayer? = ScaleLayer(2)
        _ = value(layer!.compiledForward(layer!, MLXArray([Float(1)])))

        weak var released = layer
        layer = nil

        XCTAssertNil(released, "the compiled trace retained its owner")
    }

    /// Loading, fusing, or unloading an adapter replaces modules, which leaves
    /// the trace holding the old tree. Invalidating rebuilds it.
    func testInvalidateCompiledTracesPicksUpReplacedModules() throws {
        func weight(_ value: Float) -> MLXArray {
            MLXArray([value]).reshaped(1, 1)
        }
        let one = MLXArray([Float(1)]).reshaped(1, 1)

        let container = ScaleContainer()
        container.update(parameters: ModuleParameters.unflattened(["linear.weight": weight(2)]))
        XCTAssertEqual(value(container.compiledForward(container, one)), 2)

        let replacement = Linear(1, 1, bias: false)
        replacement.update(parameters: ModuleParameters.unflattened(["weight": weight(7)]))
        container.update(modules: ModuleChildren.unflattened([("linear", replacement)]))

        container.invalidateCompiledTraces()
        XCTAssertEqual(value(container.compiledForward(container, one)), 7)
    }

    /// `invalidateCompiledTraces()` reaches owners nested in the tree.
    func testInvalidateCompiledTracesWalksTheModuleTree() {
        let layer = ScaleLayer(2)
        let parent = ScaleParent(layer)

        XCTAssertFalse(layer.compiledForward.isCompiled)
        _ = value(layer.compiledForward(layer, MLXArray([Float(1)])))
        XCTAssertTrue(layer.compiledForward.isCompiled)

        parent.invalidateCompiledTraces()
        XCTAssertFalse(layer.compiledForward.isCompiled)
    }

    /// Traces reach invalidation whether a module stores one directly or keeps
    /// several in a collection.
    func testInvalidateCompiledTracesFindsTracesInCollections() {
        let bank = ScaleBank(2)
        for trace in bank.traces {
            _ = value(trace(bank, MLXArray([Float(1)]))[0])
        }
        XCTAssertTrue(bank.traces.allSatisfy(\.isCompiled))

        bank.invalidateCompiledTraces()
        XCTAssertTrue(bank.traces.allSatisfy { !$0.isCompiled })
    }

    func testCompileRunsBodyWithoutState() {
        let f = compile(shapeless: true) { (a: MLXArray, b: MLXArray) in a * b + 1 }
        XCTAssertEqual(value(f(MLXArray([Float(2)]), MLXArray([Float(3)]))), 7)
    }
}
