// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

/// The LoRA and DoRA layers copy their base layer's arrays, so they must refuse layers whose
/// state or forward pass the copy cannot reproduce (#800).
final class LoRABaseLayerTests: XCTestCase {

    private static let fineTuneTypes: [LoRAConfiguration.FineTuneType] = [.lora, .dora]

    private static func rotatedLayers() -> [Linear] {
        [
            HadamardQuantizedLinear(
                replacing: Linear(64, 32, bias: false), groupSize: 64, bits: 4,
                rotation: SignedBlockHadamard(blockSize: 64)),
            RotateQuantizedLinear(
                inputDims: 64, outputDims: 32, hasBias: false, groupSize: 64, bits: 4, krot: 1),
        ]
    }

    private func assertUnsupported(_ error: Error, _ layer: Linear) {
        guard case ModelAdapterError.unsupportedLayer(let key, let type) = error else {
            return XCTFail("unexpected error \(error)")
        }
        XCTAssertEqual(key, "projection")
        XCTAssertEqual(type, String(describing: Swift.type(of: layer)))
    }

    func testContainerRejectsRotatedLayersBeforeChangingTheModel() throws {
        for rotated in Self.rotatedLayers() {
            for fineTuneType in Self.fineTuneTypes {
                let model = TwoBlockModel(Linear(64, 32, bias: false), rotated)
                let plain = model.first.projection
                let configuration = LoRAConfiguration(
                    numLayers: 2, fineTuneType: fineTuneType, loraParameters: .init(rank: 4))

                XCTAssertThrowsError(
                    try LoRAContainer.from(model: model, configuration: configuration)
                ) { assertUnsupported($0, rotated) }
                XCTAssertThrowsError(
                    try LoRAContainer(configuration: configuration, parameters: .init())
                        .load(into: model)
                ) { assertUnsupported($0, rotated) }

                XCTAssertTrue(model.first.projection === plain)
                XCTAssertTrue(model.second.projection === rotated)
                XCTAssertFalse(
                    model.trainableParameters().flattened().isEmpty,
                    "a rejected adapter froze the model")
            }
        }
    }

    func testAdaptingAnAdapterWrapsItsBaseLayer() throws {
        let linear = Linear(weight: MLXRandom.normal([32, 64], key: MLXRandom.key(0)))
        let quantized = try XCTUnwrap(
            linear.toQuantized(groupSize: 64, bits: 4, mode: .affine) as? QuantizedLinear)
        let adapters: [(Linear) -> LoRALayer] = [
            { LoRALinear.from(linear: $0) }, { DoRALinear.from(linear: $0) },
        ]

        for base in [linear, quantized] {
            base.train(false)
            for first in adapters {
                for second in adapters {
                    let adapted = second(try XCTUnwrap(first(base) as? Linear))
                    let reverted = try XCTUnwrap(adapted.reverted() as? Linear)

                    XCTAssertFalse(adapted.training)
                    XCTAssertTrue(type(of: reverted) == type(of: base))
                    let expected = Dictionary(uniqueKeysWithValues: base.parameters().flattened())
                    let actual = Dictionary(uniqueKeysWithValues: reverted.parameters().flattened())
                    XCTAssertEqual(Set(actual.keys), Set(expected.keys))
                    for (key, value) in expected {
                        XCTAssertTrue(arrayEqual(try XCTUnwrap(actual[key]), value).item(Bool.self))
                    }
                }
            }
        }
    }

    func testQuantizedAdaptersKeepTheGlobalScale() throws {
        let weight = 0.05 * MLXRandom.normal([32, 64], key: MLXRandom.key(0))
        let globalScale = abs(weight).max()
        let base = QuantizedLinear(
            weight: weight, bias: nil, groupSize: 16, bits: 4, mode: .nvfp4,
            globalScale: globalScale)
        let x = MLXRandom.normal([3, 64], key: MLXRandom.key(1))
        let expected = base(x)

        for adapter in [LoRALinear.from(linear: base), DoRALinear.from(linear: base)] {
            // A fresh adapter's update is zero, so every form matches the base layer.
            let layer = try XCTUnwrap(adapter as? Linear)
            XCTAssertEqual(layer(x.asType(.float16)).dtype, .float16)
            let reverted = try XCTUnwrap(adapter.reverted() as? QuantizedLinear)
            let fused = try XCTUnwrap(adapter.fused() as? QuantizedLinear)
            XCTAssertNotNil(reverted.globalScale)
            XCTAssertNotNil(fused.globalScale)
            // DoRA keeps its magnitude in the bfloat16 the nvfp4 weight dequantizes to.
            for (name, output) in [
                ("adapter", layer(x)), ("reverted", reverted(x)), ("fused", fused(x)),
            ] {
                XCTAssertTrue(
                    allClose(output, expected, rtol: 1e-2, atol: 1e-2).item(Bool.self),
                    "\(type(of: adapter)) \(name)")
            }
        }
    }
}

private final class Block: Module {
    @ModuleInfo(key: "projection") var projection: Linear

    init(_ projection: Linear) {
        _projection.wrappedValue = projection
        super.init()
    }
}

private final class TwoBlockModel: Module, LanguageModel, LoRAModel, KVCacheDimensionProvider {
    @ModuleInfo var first: Block
    @ModuleInfo var second: Block

    let kvHeads: [Int] = []

    var loraLayers: [Module] { [first, second] }
    var loraDefaultKeys: [String] { ["projection"] }

    init(_ first: Linear, _ second: Linear) {
        _first.wrappedValue = Block(first)
        _second.wrappedValue = Block(second)
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        first.projection(inputs)
    }
}
