// Copyright © 2025 INQTR. All rights reserved.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers
import os

private let logger = Logger(subsystem: "mlx-swift-lm", category: "paroquant")

// MARK: - Detection

/// Returns `true` if the model directory contains a ParoQuant checkpoint.
///
/// Inspects `config.json` for `quant_method == "paroquant"` in `quantization_config`
/// and verifies a supported architecture is declared.
public func isParoQuantModel(directory: URL) -> Bool {
    guard let configData = try? Data(contentsOf: directory.appendingPathComponent("config.json"))
    else {
        return false
    }
    return isSupportedParoQuantModel(directory: directory, configData: configData)
}

// MARK: - Config

struct ParoQuantConfig: Sendable {
    let bits: Int
    let groupSize: Int
    let krot: Int
}

/// Reads ParoQuant quantization config from config.json data.
private func readParoQuantConfig(_ configData: Data) -> ParoQuantConfig? {
    guard let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
        let qc = json["quantization_config"] as? [String: Any]
    else { return nil }

    let bits = qc["bits"] as? Int ?? 4
    let groupSize = qc["group_size"] as? Int ?? 128
    let krot = qc["krot"] as? Int ?? 8
    return ParoQuantConfig(bits: bits, groupSize: groupSize, krot: krot)
}

/// Checks whether a model directory contains a ParoQuant checkpoint by inspecting
/// `quant_method` and `architectures` in config.json.
private func isSupportedParoQuantModel(directory: URL, configData: Data) -> Bool {
    guard let json = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
        let qc = json["quantization_config"] as? [String: Any],
        let method = qc["quant_method"] as? String,
        method == "paroquant"
    else { return false }

    let architectures = json["architectures"] as? [String] ?? []
    return architectures.contains("Qwen3_5ForConditionalGeneration")
}

// MARK: - AutoAWQ Conversion

private enum AWQ {
    static let bits = 4
    static let packFactor = 32 / bits  // 8 values per uint32
    static let mask: Int32 = (1 << bits) - 1
    static let shifts: [Int32] = (0 ..< 8).map { Int32($0 * bits) }
    /// Inverse of AutoAWQ reorder [0,2,4,6,1,3,5,7] → [0,4,1,5,2,6,3,7]
    static let inverseReorder = [0, 4, 1, 5, 2, 6, 3, 7]

    // Pre-computed MLXArrays (created once, reused across calls)
    nonisolated(unsafe) static let shiftsArray = MLXArray(shifts.map { Int64($0) }).reshaped(
        1, 1, 8)
    nonisolated(unsafe) static let reorderIndices = MLXArray(inverseReorder.map { Int32($0) })
}

/// Unpack AutoAWQ int32 → raw uint8 values, undoing the [0,2,4,6,1,3,5,7] reorder.
private func unpackAndReorder(_ packed: MLXArray) -> MLXArray {
    let rows = packed.dim(0)
    let cols = packed.dim(1)

    let expanded = packed.asType(.int64).expandedDimensions(axis: 2)
    let raw = ((expanded >> AWQ.shiftsArray) & Int64(AWQ.mask)).asType(.uint8)
    let reordered = raw.take(AWQ.reorderIndices, axis: 2)

    return reordered.reshaped(rows, cols * 8)
}

/// Pack raw uint8 values into uint32 (MLX sequential layout).
private func packMLX(_ w: MLXArray) -> MLXArray {
    let rows = w.dim(0)
    let reshaped = w.reshaped(rows, -1, AWQ.packFactor)  // [rows, cols/8, 8]

    var packed = reshaped[0..., 0..., 0].asType(.uint32)
    for i in 1 ..< AWQ.packFactor {
        packed = packed | (reshaped[0..., 0..., i].asType(.uint32) << UInt32(i * AWQ.bits))
    }
    return packed
}

/// Convert AutoAWQ checkpoint weights to MLX quantized format in-place.
private func convertAutoAWQ(
    _ weights: inout [String: MLXArray], groupSize: Int
) {
    let prefixes = Set(
        weights.keys
            .filter { $0.hasSuffix(".qweight") }
            .compactMap { key -> String? in
                let pfx = String(key.dropLast("qweight".count))
                return weights["\(pfx)theta"] != nil ? pfx : nil
            }
    )

    guard !prefixes.isEmpty else { return }

    // Pass 1: compute biases from qzeros + scales BEFORE scales are transposed.
    for pfx in prefixes {
        guard let qzeros = weights.removeValue(forKey: "\(pfx)qzeros"),
            let scales = weights["\(pfx)scales"]
        else { continue }
        let zeros = unpackAndReorder(qzeros).asType(.float32)
        weights["\(pfx)biases"] = (-scales.asType(.float32) * zeros).transposed().asType(.float16)
    }

    // Pass 2: convert remaining keys (qweight, scales, channel_scales)
    let keysToConvert = weights.keys.filter { key in
        prefixes.contains(where: { key.hasPrefix($0) })
    }

    for key in keysToConvert {
        guard let pfx = prefixes.first(where: { key.hasPrefix($0) }) else { continue }
        let suffix = String(key.dropFirst(pfx.count))

        switch suffix {
        case "qweight":
            let val = weights.removeValue(forKey: key)!
            weights["\(pfx)weight"] = packMLX(unpackAndReorder(val).transposed())

        case "scales":
            weights[key] = weights[key]!.transposed()

        case "channel_scales":
            if let val = weights[key], val.ndim == 1 {
                weights[key] = val.reshaped(1, -1)
            }

        default:
            break  // theta, pairs, bias — keep as-is
        }
    }
}

// MARK: - Layer Patching

private func requireTensor(
    _ key: String, weights: [String: MLXArray]
) throws -> MLXArray {
    guard let tensor = weights[key] else {
        throw ParoQuantError.missingTensor(key)
    }
    return tensor
}

private func verifyTensorShape(
    _ tensor: MLXArray, key: String, expected: [Int]
) throws {
    guard tensor.shape == expected else {
        throw ParoQuantError.invalidTensorShape(
            key: key,
            expected: expected,
            actual: tensor.shape
        )
    }
}

private func rotationLeafModules(model: Module) -> [String: Module] {
    Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
}

private func rotationModuleSpec(
    prefix: String,
    leafModules: [String: Module],
    weights: [String: MLXArray],
    bits: Int,
    groupSize: Int
) throws -> (inputDims: Int, outputDims: Int, hasBias: Bool, krot: Int) {
    guard let original = leafModules[prefix] else {
        throw ParoQuantError.rotationLayerNotFound(prefix)
    }
    guard let linear = original as? Linear else {
        throw ParoQuantError.rotationLayerTypeMismatch(
            path: prefix,
            actualType: String(describing: type(of: original))
        )
    }

    let outputDims = linear.shape.0
    let inputDims = linear.shape.1
    let groups = inputDims / groupSize
    let packedInputDims = inputDims * bits / 32
    let expectsBias = linear.bias != nil

    let theta = try requireTensor("\(prefix).theta", weights: weights)
    let pairs = try requireTensor("\(prefix).pairs", weights: weights)
    let channelScales = try requireTensor("\(prefix).channel_scales", weights: weights)
    let weight = try requireTensor("\(prefix).weight", weights: weights)
    let scales = try requireTensor("\(prefix).scales", weights: weights)
    let biases = try requireTensor("\(prefix).biases", weights: weights)

    let krot = theta.dim(0)
    try verifyTensorShape(theta, key: "\(prefix).theta", expected: [krot, inputDims / 2])
    try verifyTensorShape(pairs, key: "\(prefix).pairs", expected: [krot, inputDims])
    try verifyTensorShape(
        channelScales, key: "\(prefix).channel_scales", expected: [1, inputDims])
    try verifyTensorShape(
        weight, key: "\(prefix).weight", expected: [outputDims, packedInputDims])
    try verifyTensorShape(scales, key: "\(prefix).scales", expected: [outputDims, groups])
    try verifyTensorShape(biases, key: "\(prefix).biases", expected: [outputDims, groups])

    if expectsBias {
        _ = try requireTensor("\(prefix).bias", weights: weights)
    }

    return (inputDims, outputDims, expectsBias, krot)
}

/// Replace Linear layers with RotateQuantizedLinear where rotation parameters exist.
private func patchRotationLayers(
    model: Module, weights: [String: MLXArray],
    bits: Int, groupSize: Int
) throws {
    let prefixes = weights.keys
        .filter { $0.hasSuffix(".theta") }
        .map { String($0.dropLast(".theta".count)) }
        .sorted()

    guard !prefixes.isEmpty else { return }

    let leafModules = rotationLeafModules(model: model)
    var updates = [(String, Module)]()

    for prefix in prefixes {
        let spec = try rotationModuleSpec(
            prefix: prefix,
            leafModules: leafModules,
            weights: weights,
            bits: bits,
            groupSize: groupSize
        )

        let replacement = RotateQuantizedLinear(
            inputDims: spec.inputDims,
            outputDims: spec.outputDims,
            hasBias: spec.hasBias,
            groupSize: groupSize,
            bits: bits,
            krot: spec.krot
        )

        updates.append((prefix, replacement))
    }

    if !updates.isEmpty {
        try model.update(modules: ModuleChildren.unflattened(updates), verify: [.noUnusedKeys])

        let patchedLeaves = rotationLeafModules(model: model)
        for (path, _) in updates {
            guard patchedLeaves[path] is RotateQuantizedLinear else {
                throw ParoQuantError.rotationLayerPatchFailed(path)
            }
        }
    }
}

/// Predicate for the native MLX quantization pass.
private func isParoQuantIOLayer(path: String, module: Module) -> Bool {
    guard module is Quantizable else { return false }
    return path.hasSuffix("embed_tokens") || path.hasSuffix("lm_head")
}

/// Layers already represented in MLX quantized checkpoint form.
private func isCheckpointQuantizedLayer(
    path: String, weights: [String: MLXArray]
) -> Bool {
    weights["\(path).scales"] != nil && weights["\(path).theta"] == nil
}

// MARK: - UserInputProcessor

/// Local UserInputProcessor for ParoQuant models.
private struct ParoQuantInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let configuration: ModelConfiguration
    let messageGenerator: MessageGenerator

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        do {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools,
                additionalContext: input.additionalContext)
            return LMInput(tokens: MLXArray(promptTokens))
        } catch {
            // Fallback for missing chat template or other tokenizer errors
            let prompt =
                messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}

// MARK: - Load Entry Point

/// Load a ParoQuant model from a local directory, returning a ``ModelContainer``.
///
/// Handles AutoAWQ weight conversion, rotation layer patching, and IO layer
/// quantization. Rotation parameters (theta, pairs, channel_scales) are kept
/// in the model and applied to activations at runtime via Metal kernel.
///
/// - Parameters:
///   - directory: Local path to the model checkpoint directory.
///   - typeRegistry: Registry used to create the underlying model architecture.
///   - toolCallFormat: Optional tool-call format for the model configuration.
/// - Returns: A ``ModelContainer`` ready for inference.
public func loadParoQuantModel(
    from directory: URL,
    typeRegistry: ModelTypeRegistry,
    toolCallFormat: ToolCallFormat? = nil
) async throws -> ModelContainer {
    // 1. Parse config.json (flatten VLM text_config if present)
    let configURL = directory.appendingPathComponent("config.json")
    var configData = try Data(contentsOf: configURL)
    guard isSupportedParoQuantModel(directory: directory, configData: configData) else {
        throw ParoQuantError.unsupportedModel
    }
    guard var configJSON = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
    else {
        throw ParoQuantError.missingConfig
    }
    if let textConfig = configJSON["text_config"] as? [String: Any] {
        for (key, value) in textConfig {
            configJSON[key] = value
        }
        configJSON["model_type"] = "qwen3_5"
        configData = try JSONSerialization.data(withJSONObject: configJSON)
    }
    let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: configData)

    // 2. Read ParoQuant params
    guard let paroConfig = readParoQuantConfig(configData) else {
        throw ParoQuantError.missingConfig
    }
    logger.info(
        "ParoQuant config: bits=\(paroConfig.bits), groupSize=\(paroConfig.groupSize), krot=\(paroConfig.krot)"
    )

    // 3. Create model via standard typeRegistry
    let model =
        try await typeRegistry
        .createModel(configuration: configData, modelType: baseConfig.modelType)

    // 4. EOS token override from generation_config.json
    var eosTokenIds = Set(baseConfig.eosTokenIds?.values ?? [])
    let genConfigURL = directory.appendingPathComponent("generation_config.json")
    if let genData = try? Data(contentsOf: genConfigURL),
        let genConfig = try? JSONDecoder().decode(GenerationConfigFile.self, from: genData),
        let genEos = genConfig.eosTokenIds?.values
    {
        eosTokenIds = Set(genEos)
    }

    var config = ModelConfiguration(directory: directory, toolCallFormat: toolCallFormat)
    config.eosTokenIds = eosTokenIds

    // 5. Load raw safetensors
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: directory, includingPropertiesForKeys: nil)!
    while let url = enumerator.nextObject() as? URL {
        if url.pathExtension == "safetensors",
            url.lastPathComponent != "prerotated_cache.safetensors"
        {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }

    logger.info("Loaded \(weights.count) weight keys from safetensors")

    // 6. Convert AutoAWQ format → MLX format (BEFORE sanitize)
    if weights.keys.contains(where: { $0.hasSuffix(".qweight") }) {
        convertAutoAWQ(&weights, groupSize: paroConfig.groupSize)
        logger.info("Converted AutoAWQ weights to MLX format")
    }

    // 7. Model-specific sanitization
    weights = model.sanitize(weights: weights)

    // 8. Patch rotation layers
    try patchRotationLayers(
        model: model, weights: weights,
        bits: paroConfig.bits, groupSize: paroConfig.groupSize
    )

    // 9. Quantize non-rotation layers in MLX quantized form
    quantize(model: model) { path, module in
        guard module is Quantizable else { return nil }
        guard isCheckpointQuantizedLayer(path: path, weights: weights) else {
            return nil
        }
        return (paroConfig.groupSize, paroConfig.bits, .affine)
    }

    // 10. Load checkpoint weights into the patched model
    let parameters = ModuleParameters.unflattened(weights)
    let verify: Module.VerifyUpdate = [.allModelKeysSet, .shapeMismatch]
    try model.update(parameters: parameters, verify: verify)

    // 11. Quantize IO embedding path from FP16 weights
    quantize(model: model) { path, module in
        guard isParoQuantIOLayer(path: path, module: module) else {
            return nil
        }
        return (paroConfig.groupSize, paroConfig.bits, .affine)
    }

    // 12. Materialize
    eval(model)
    logger.info("ParoQuant model loaded and evaluated")

    // 13. Load tokenizer
    let tokenizer = try await loadTokenizer(configuration: config, hub: defaultHubApi)

    // 14. Create processor with messageGenerator
    // Use DefaultMessageGenerator — LLMModel.messageGenerator(tokenizer:) is in MLXLLM
    // and this loader lives in MLXLMCommon. Callers who need custom message generation
    // can swap the processor after loading.
    let messageGenerator: MessageGenerator = DefaultMessageGenerator()
    let processor = ParoQuantInputProcessor(
        tokenizer: tokenizer, configuration: config,
        messageGenerator: messageGenerator
    )

    // 15. Assemble ModelContext → ModelContainer
    let context = ModelContext(
        configuration: config, model: model,
        processor: processor, tokenizer: tokenizer
    )
    return ModelContainer(context: context)
}

// MARK: - Errors

public enum ParoQuantError: LocalizedError {
    case missingConfig
    case unsupportedModel
    case missingTensor(String)
    case invalidTensorShape(key: String, expected: [Int], actual: [Int])
    case rotationLayerNotFound(String)
    case rotationLayerTypeMismatch(path: String, actualType: String)
    case rotationLayerPatchFailed(String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig:
            return "Missing quantization_config in config.json for ParoQuant model"
        case .unsupportedModel:
            return "The custom ParoQuant loader only supports z-lab/Qwen3.5-4B-PARO"
        case .missingTensor(let key):
            return "Missing required ParoQuant tensor: \(key)"
        case .invalidTensorShape(let key, let expected, let actual):
            return "Invalid ParoQuant tensor shape for \(key): expected \(expected), got \(actual)"
        case .rotationLayerNotFound(let path):
            return "Unable to find ParoQuant rotation layer in model: \(path)"
        case .rotationLayerTypeMismatch(let path, let actualType):
            return
                "ParoQuant rotation layer \(path) is not a Linear-compatible module: \(actualType)"
        case .rotationLayerPatchFailed(let path):
            return "Failed to replace ParoQuant layer with RotateQuantizedLinear: \(path)"
        }
    }
}
