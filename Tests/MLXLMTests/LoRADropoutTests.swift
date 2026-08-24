// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers
import XCTest

final class LoRADropoutTests: XCTestCase {

    func testConfigurationDefaultsMissingDropoutToZero() throws {
        let data = Data(
            """
            {
              "num_layers": 4,
              "fine_tune_type": "lora",
              "lora_parameters": {
                "rank": 8,
                "scale": 2.0
              }
            }
            """.utf8)

        let configuration = try JSONDecoder().decode(LoRAConfiguration.self, from: data)

        XCTAssertEqual(configuration.loraParameters.dropout, 0.0)
    }

    func testConfigurationRoundTripsDropout() throws {
        let configuration = LoRAConfiguration(
            loraParameters: .init(rank: 16, scale: 2.0, dropout: 0.25))

        let data = try JSONEncoder().encode(configuration)
        let decoded = try JSONDecoder().decode(LoRAConfiguration.self, from: data)

        XCTAssertEqual(decoded.loraParameters.dropout, 0.25)
    }

    func testConfigurationRejectsInvalidDropout() throws {
        for dropout in [-0.1, 1.0] {
            let data = Data(
                """
                {
                  "num_layers": 4,
                  "fine_tune_type": "lora",
                  "lora_parameters": {
                    "rank": 8,
                    "scale": 2.0,
                    "dropout": \(dropout)
                  }
                }
                """.utf8)

            XCTAssertThrowsError(try JSONDecoder().decode(LoRAConfiguration.self, from: data)) {
                error in
                guard case DecodingError.dataCorrupted = error else {
                    return XCTFail("expected dataCorrupted, got \(error)")
                }
            }
        }
    }

    func testPEFTDropoutIsPreserved() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configuration = Data(
            """
            {
              "peft_type": "LORA",
              "r": 4,
              "lora_alpha": 8,
              "lora_dropout": 0.2,
              "target_modules": ["q_proj"],
              "use_dora": false
            }
            """.utf8)
        try configuration.write(to: directory.appending(component: "adapter_config.json"))
        try save(
            arrays: [
                "base_model.model.encoder.layers.0.self_attn.q_proj.lora_A.weight":
                    MLXArray.zeros([4, 8]),
                "base_model.model.encoder.layers.0.self_attn.q_proj.lora_B.weight":
                    MLXArray.zeros([8, 4]),
            ],
            url: directory.appending(component: "adapter_model.safetensors"))

        let adapter = try LoRAContainer.fromPEFT(directory: directory)

        XCTAssertEqual(adapter.configuration.loraParameters.dropout, 0.2)
    }

    func testPEFTConfigurationRejectsInvalidDropout() throws {
        let data = Data(
            """
            {
              "peft_type": "LORA",
              "r": 4,
              "lora_alpha": 8,
              "lora_dropout": 1.0,
              "target_modules": ["q_proj"],
              "use_dora": false
            }
            """.utf8)

        XCTAssertThrowsError(
            try JSONDecoder().decode(PEFTAdapterConfiguration.self, from: data)
        ) { error in
            guard case DecodingError.dataCorrupted = error else {
                return XCTFail("expected dataCorrupted, got \(error)")
            }
        }
    }

    func testLoRALinearAppliesDropoutOnlyDuringTraining() throws {
        let base = Linear(weight: MLXArray.eye(32))
        let layer = try XCTUnwrap(
            LoRALinear.from(linear: base, rank: 32, scale: 1.0, dropout: 0.5)
                as? LoRALinear)

        try assertDropoutMode(on: layer)
    }

    func testConversionPreservesEvaluationMode() throws {
        let base = Linear(weight: MLXArray.eye(32))
        base.train(false)

        let lora = try XCTUnwrap(
            LoRALinear.from(linear: base, dropout: 0.5) as? LoRALinear)
        let dora = try XCTUnwrap(
            DoRALinear.from(linear: base, dropout: 0.5) as? DoRALinear)

        XCTAssertFalse(lora.training)
        XCTAssertFalse(dora.training)
    }

    func testModelContextDisablesLoadedAdapterDropoutForInference() throws {
        let configuration = LoRAConfiguration(
            numLayers: 1,
            loraParameters: .init(
                rank: 32, scale: 1.0, dropout: 0.5, keys: ["projection"]))
        let adapter = LoRAContainer(
            configuration: configuration,
            parameters: ModuleParameters.unflattened([
                "projection.lora_a": MLXArray.eye(32),
                "projection.lora_b": MLXArray.eye(32),
            ]))
        let model = DropoutInferenceModel()
        let processor = TestInputProcessor()
        let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer)

        try adapter.load(into: model)

        XCTAssertFalse(context.model.training)
        let layer = try XCTUnwrap(model.projection as? LoRALinear)
        XCTAssertFalse(layer.training)

        let input = MLXArray.ones([1, 32])
        let first = model(input, cache: nil)
        eval(first)
        let second = model(input, cache: nil)
        eval(second)
        XCTAssertTrue(allClose(first, second, rtol: 0, atol: 0).item(Bool.self))
    }

    func testQLoRALinearAppliesDropoutOnlyDuringTraining() throws {
        let base = Linear(weight: MLXArray.eye(32))
        let quantized = try XCTUnwrap(
            base.toQuantized(groupSize: 32, bits: 4, mode: .affine) as? QuantizedLinear)
        let layer = try XCTUnwrap(
            LoRALinear.from(linear: quantized, rank: 32, scale: 1.0, dropout: 0.5)
                as? QLoRALinear)

        try assertDropoutMode(on: layer)
    }

    func testDoRALinearAppliesDropoutOnlyDuringTraining() throws {
        let base = Linear(weight: MLXArray.eye(32))
        let layer = try XCTUnwrap(
            DoRALinear.from(linear: base, rank: 32, scale: 1.0, dropout: 0.5)
                as? DoRALinear)

        try assertDropoutMode(on: layer)
    }

    func testDoRAWeightNormIsDetachedFromGradient() throws {
        let base = Linear(weight: MLXArray([Float(2.0)], [1, 1]))
        let layer = DoRALinear(linear: base, rank: 1, scale: 1.0, dropout: 0.0)
        try layer.update(
            parameters: ModuleParameters.unflattened([
                "lora_a": MLXArray.ones([1, 1]),
                "lora_b": MLXArray.ones([1, 1]),
            ]),
            verify: [])

        let valueAndGradient = valueAndGrad(model: layer) { model, inputs in
            [model(inputs[0]).sum()]
        }
        let (_, gradients) = valueAndGradient(layer, [MLXArray.ones([1, 1])])
        let flattened = Dictionary(uniqueKeysWithValues: gradients.flattened())
        let loraAGradient = try XCTUnwrap(flattened["lora_a"])
        let loraBGradient = try XCTUnwrap(flattened["lora_b"])

        // With stopGradient(norm(weight + B @ A)), each low-rank gradient is 2 / 3.
        // Without detaching the norm, the scalar DoRA output algebraically cancels and both
        // gradients are zero, so this directly guards parity with mlx-lm.
        XCTAssertEqual(loraAGradient.item(Float.self), 2.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(loraBGradient.item(Float.self), 2.0 / 3.0, accuracy: 1e-5)
    }

    func testQDoRALinearAppliesDropoutOnlyDuringTraining() throws {
        let base = Linear(weight: MLXArray.eye(32))
        let quantized = try XCTUnwrap(
            base.toQuantized(groupSize: 32, bits: 4, mode: .affine) as? QuantizedLinear)
        let layer = try XCTUnwrap(
            DoRALinear.from(linear: quantized, rank: 32, scale: 1.0, dropout: 0.5)
                as? QDoRALinear)

        try assertDropoutMode(on: layer)
    }

    func testQuantizedAdaptersPreserveQuantizationMode() throws {
        let base = Linear(weight: MLXArray.eye(32))
        let quantized = try XCTUnwrap(
            base.toQuantized(groupSize: 32, bits: 4, mode: .mxfp4) as? QuantizedLinear)
        let lora = try XCTUnwrap(LoRALinear.from(linear: quantized) as? QLoRALinear)
        let dora = try XCTUnwrap(DoRALinear.from(linear: quantized) as? QDoRALinear)

        XCTAssertEqual(lora.mode, .mxfp4)
        XCTAssertEqual((lora.reverted() as? QuantizedLinear)?.mode, .mxfp4)
        XCTAssertEqual((lora.fused() as? QuantizedLinear)?.mode, .mxfp4)
        XCTAssertEqual(dora.mode, .mxfp4)
        XCTAssertEqual((dora.reverted() as? QuantizedLinear)?.mode, .mxfp4)
        XCTAssertEqual((dora.fused() as? QuantizedLinear)?.mode, .mxfp4)
    }

    func testEvaluateUsesEvaluationModeAndRestoresMode() {
        let model = Module()
        model.train()
        var observedTraining: Bool?

        _ = LoRATrain.evaluate(
            model: model,
            dataset: ["sample"],
            loss: { model, _, _, _ in
                observedTraining = model.training
                return (MLXArray(0.0), MLXArray(1))
            },
            tokenizer: TestTokenizer(),
            batchSize: 1,
            batchCount: 1)

        XCTAssertEqual(observedTraining, false)
        XCTAssertTrue(model.training)
    }

    func testTrainUsesTrainingModeAndRestoresMode() throws {
        let model = Linear(1, 1, bias: false)
        model.train(false)
        var observedModes = Set<Bool>()

        try LoRATrain.train(
            model: model,
            train: ["sample"],
            validate: ["sample"],
            optimizer: SGD(learningRate: 0.01),
            loss: { model, _, _, _ in
                observedModes.insert(model.training)
                let prediction = (model as! Linear)(MLXArray.ones([1, 1]))
                return ((prediction * prediction).mean(), MLXArray(1))
            },
            tokenizer: TestTokenizer(),
            parameters: .init(
                batchSize: 1, iterations: 1, stepsPerReport: 1, stepsPerEval: 1,
                validationBatches: 1),
            progress: { _ in
                XCTAssertTrue(model.training)
                return .more
            })

        XCTAssertEqual(observedModes, [false, true])
        XCTAssertFalse(model.training)
    }

    func testTrainingResumesFromCompletedIteration() throws {
        let model = Linear(1, 1, bias: false)
        var reportedIterations = [Int]()

        try LoRATrain.train(
            model: model,
            train: ["sample"],
            validate: ["sample"],
            optimizer: SGD(learningRate: 0.01),
            loss: { model, _, _, _ in
                let prediction = (model as! Linear)(MLXArray.ones([1, 1]))
                return ((prediction * prediction).mean(), MLXArray(1))
            },
            tokenizer: TestTokenizer(),
            parameters: .init(
                batchSize: 1, iterations: 4, stepsPerReport: 1, stepsPerEval: 100,
                validationBatches: 1, completedIterations: 2),
            progress: { progress in
                if case .train(let iteration, _, _, _) = progress {
                    reportedIterations.append(iteration)
                }
                return .more
            })

        XCTAssertEqual(reportedIterations, [2, 3])
    }

    func testLoadLoRAWeightsRestoresSavedAdapter() throws {
        let layer = try XCTUnwrap(
            LoRALinear.from(linear: Linear(weight: MLXArray.eye(4)), rank: 4)
                as? LoRALinear)
        let url = FileManager.default.temporaryDirectory
            .appending(component: UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try layer.update(
            parameters: ModuleParameters.unflattened([
                "lora_a": MLXArray.eye(4),
                "lora_b": MLXArray.eye(4),
            ]),
            verify: [])
        try LoRATrain.saveLoRAWeights(model: layer, url: url)

        try layer.update(
            parameters: ModuleParameters.unflattened([
                "lora_a": MLXArray.zeros([4, 4]),
                "lora_b": MLXArray.zeros([4, 4]),
            ]),
            verify: [])
        try LoRATrain.loadLoRAWeights(model: layer, url: url)

        let restored = Dictionary(uniqueKeysWithValues: layer.trainableParameters().flattened())
        XCTAssertTrue(
            allClose(restored["lora_a"]!, MLXArray.eye(4), rtol: 0, atol: 0).item(Bool.self))
        XCTAssertTrue(
            allClose(restored["lora_b"]!, MLXArray.eye(4), rtol: 0, atol: 0).item(Bool.self))
    }

    private func assertDropoutMode(on layer: Linear) throws {
        try layer.update(
            parameters: ModuleParameters.unflattened([
                "lora_a": MLXArray.eye(32),
                "lora_b": MLXArray.eye(32),
            ]),
            verify: [])

        let input = MLXArray.ones([1, 32])

        layer.train(false)
        let firstEvaluationOutput = layer(input)
        eval(firstEvaluationOutput)

        layer.train()
        let trainingOutput = layer(input)
        eval(trainingOutput)

        layer.train(false)
        let secondEvaluationOutput = layer(input)
        eval(secondEvaluationOutput)

        XCTAssertTrue(
            allClose(firstEvaluationOutput, secondEvaluationOutput, rtol: 0, atol: 0)
                .item(Bool.self))
        XCTAssertFalse(
            allClose(firstEvaluationOutput, trainingOutput, rtol: 0, atol: 0).item(Bool.self))
    }
}

private final class DropoutInferenceModel: Module, LanguageModel, LoRAModel,
    KVCacheDimensionProvider
{
    @ModuleInfo(key: "projection") var projection: Linear

    let kvHeads: [Int] = []

    var loraLayers: [Module] { [self] }
    var loraDefaultKeys: [String] { ["projection"] }

    override init() {
        _projection.wrappedValue = Linear(weight: MLXArray.eye(32))
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projection(inputs)
    }
}
