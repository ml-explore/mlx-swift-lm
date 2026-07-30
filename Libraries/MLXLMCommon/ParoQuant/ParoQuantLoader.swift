import Foundation
import MLX
import MLXNN

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
    let supported: Set<String> = [
        "Qwen3_5ForConditionalGeneration",  // dense Qwen3.5/3.6 PARO
        "Qwen3_5MoeForConditionalGeneration",  // MoE (Qwen3.6-35B-A3B PARO)
    ]
    return architectures.contains(where: supported.contains)
}

// MARK: - Config Flattening

/// Flatten a VLM config's `text_config` onto the top level so
/// `BaseConfiguration` and the model-type registry see the text-model fields.
///
/// The checkpoint's *top-level* `model_type` is the registry key ("qwen3_5",
/// "qwen3_5_moe"); `text_config` carries a "*_text" variant ("qwen3_5_text",
/// "qwen3_5_moe_text") that no registry resolves, so the flatten would
/// clobber the key. Preserve the top-level value instead of force-setting
/// "qwen3_5" (which mis-resolved MoE checkpoints to the dense architecture).
///
/// Internal (not private) so the preservation contract is unit-testable.
func flattenParoQuantTextConfig(_ configJSON: [String: Any]) -> [String: Any] {
    guard let textConfig = configJSON["text_config"] as? [String: Any] else {
        return configJSON
    }
    var flattened = configJSON
    let checkpointModelType = configJSON["model_type"]
    for (key, value) in textConfig {
        flattened[key] = value
    }
    if let checkpointModelType {
        flattened["model_type"] = checkpointModelType
    }
    return flattened
}

// MARK: - AutoAWQ Conversion

private enum AWQ {
    static let bits = 4
    static let packFactor = 32 / bits  // 8 values per uint32
    static let mask: Int32 = (1 << bits) - 1
    /// Inverse of AutoAWQ reorder [0,2,4,6,1,3,5,7] → [0,4,1,5,2,6,3,7]
    static let inverseReorder = [0, 4, 1, 5, 2, 6, 3, 7]
}

/// Unpack AutoAWQ int32 → raw uint8 values, undoing the [0,2,4,6,1,3,5,7] reorder.
///
/// The shift table and reorder indices are rebuilt per call rather than cached
/// as module-level statics — they're tiny (8 × 8 bytes) and only touched at
/// model load time, so caching bought nothing and only created thread-safety
/// concerns around unevaluated `MLXArray`s.
private func unpackAndReorder(_ packed: MLXArray) -> MLXArray {
    let rows = packed.dim(0)
    let cols = packed.dim(1)

    let shifts = MLXArray((0 ..< 8).map { Int64($0 * AWQ.bits) }).reshaped(1, 1, 8)
    let reorderIndices = MLXArray(AWQ.inverseReorder.map { Int32($0) })

    let expanded = packed.asType(.int64).expandedDimensions(axis: 2)
    let raw = ((expanded >> shifts) & Int64(AWQ.mask)).asType(.uint8)
    let reordered = raw.take(reorderIndices, axis: 2)

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

/// Split pre-fused `in_proj_ba` tensors back into separate `in_proj_b` / `in_proj_a`
/// entries. PARO and some AutoAWQ-Mamba checkpoints concatenate the B and A
/// projections along the output axis before quantising; the architecture code
/// (e.g. `Qwen35TextModel`) expects them as distinct projections. Runs before
/// model-specific `sanitize` so downstream layers see the standard layout.
///
/// No-op for keys that don't exist (non-Mamba PARO checkpoints) and for keys
/// that already have an `in_proj_b` entry (idempotent on repeat calls).
private func splitFusedMambaProjections(_ weights: inout [String: MLXArray]) {
    for k in Array(weights.keys) where k.hasSuffix(".in_proj_ba.weight") {
        let prefix = String(k.dropLast(".in_proj_ba.weight".count))
        guard weights["\(prefix).in_proj_b.weight"] == nil else { continue }
        for suffix in [".weight", ".scales", ".biases", ".bias"] {
            let baKey = "\(prefix).in_proj_ba\(suffix)"
            guard let baVal = weights.removeValue(forKey: baKey) else { continue }
            let half = baVal.dim(0) / 2
            weights["\(prefix).in_proj_b\(suffix)"] = baVal[0 ..< half]
            weights["\(prefix).in_proj_a\(suffix)"] = baVal[half...]
        }
    }
}

/// Convert AutoAWQ checkpoint weights to MLX quantized format in-place.
///
/// Internal (not private) so the conversion contract is unit-testable.
func convertAutoAWQ(
    _ weights: inout [String: MLXArray], groupSize: Int
) {
    // Every `.qweight` prefix converts, with or without a sibling `theta`:
    // MoE per-expert weights carry no theta (their rotations are shared per
    // layer under `experts.*_weight_theta`) and would otherwise be skipped.
    let prefixes = Set(
        weights.keys
            .filter { $0.hasSuffix(".qweight") }
            .map { String($0.dropLast("qweight".count)) }
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

    // Pass 2: convert remaining keys (qweight, scales, channel_scales).
    // Each key's candidate prefix is derived by suffix-stripping and tested
    // for set membership — O(keys), where the previous
    // `prefixes.contains(where: key.hasPrefix)` scan was O(keys × prefixes)
    // and cost ~29 s of the 35B MoE load (~12K per-expert prefixes).
    // `channel_scales` must be tested before `scales` (suffix shadowing).
    // theta, pairs, bias keep as-is.
    for key in Array(weights.keys) {
        if key.hasSuffix("qweight") {
            // The stripped prefix is in `prefixes` by construction.
            let pfx = String(key.dropLast("qweight".count))
            let val = weights.removeValue(forKey: key)!
            weights["\(pfx)weight"] = packMLX(unpackAndReorder(val).transposed())
        } else if key.hasSuffix("channel_scales") {
            guard prefixes.contains(String(key.dropLast("channel_scales".count))) else {
                continue
            }
            if let val = weights[key], val.ndim == 1 {
                weights[key] = val.reshaped(1, -1)
            }
        } else if key.hasSuffix("scales") {
            guard prefixes.contains(String(key.dropLast("scales".count))) else { continue }
            // float16 to match `biases` in `quantizedMM` — f32 scales with
            // f16 biases mismatch (upstream fix z-lab/paroquant#38).
            weights[key] = weights[key]!.transposed().asType(.float16)
        }
    }
}

// MARK: - MoE Passes

/// Stack per-expert 2-D tensors into the 3-D `switch_mlp` layout that
/// `SwitchGLU`/`QuantizedSwitchLinear` consume:
/// `…experts.{e}.{proj}.{suffix}` → `…switch_mlp.{proj}.{suffix}` with a new
/// leading experts axis (suffix ∈ weight / scales / biases, i.e. the
/// already-converted AWQ triple). A stack is only emitted when every expert
/// contributed the suffix; partial groups are left untouched so the strict
/// `verify: [.allModelKeysSet]` update fails loudly on the real gap.
///
/// No-op for dense checkpoints (no `…experts.{e}.…` keys). Idempotent: an
/// existing destination key is never overwritten.
///
/// Internal (not private) so the pass contract is unit-testable.
func stackMoEExpertWeights(_ weights: inout [String: MLXArray]) {
    let stackable = ["weight", "scales", "biases"]

    // group key "{base}.switch_mlp.{proj}" → expert index → suffix → source key
    var groups = [String: [Int: [String: String]]]()
    for key in weights.keys {
        guard let range = key.range(of: ".experts.") else { continue }
        // "{e}.{proj}.{suffix}"
        let parts = key[range.upperBound...].split(separator: ".")
        guard parts.count == 3, let expert = Int(parts[0]),
            stackable.contains(String(parts[2]))
        else { continue }
        let groupKey = "\(key[..<range.lowerBound]).switch_mlp.\(parts[1])"
        groups[groupKey, default: [:]][expert, default: [:]][String(parts[2])] = key
    }

    for (groupKey, experts) in groups {
        let numExperts = experts.keys.max()! + 1
        for suffix in stackable {
            let dest = "\(groupKey).\(suffix)"
            guard weights[dest] == nil else { continue }
            let sources = (0 ..< numExperts).compactMap { experts[$0]?[suffix] }
            guard sources.count == numExperts else { continue }
            weights[dest] = MLX.stacked(sources.map { weights[$0]! }, axis: 0)
            for key in sources {
                weights.removeValue(forKey: key)
            }
        }
    }
}

/// Remap the shared per-layer expert rotation keys onto the nested
/// `PairwiseRotation` children of `RotateSwitchGLU`:
/// `…experts.{gate_up,down}_weight_{theta,pairs,channel_scales}` →
/// `…switch_mlp.{gate_up,down}_rot.{theta,pairs,channel_scales}`.
///
/// Upstream z-lab keeps these keys flat on the GLU (`…gate_up_rot_theta`);
/// the nested layout is a deliberate divergence (#208) so the rotation is a
/// reusable `PairwiseRotation` Module rather than mixin state.
///
/// No-op for dense checkpoints. Internal for unit testing.
func remapSharedMoERotations(_ weights: inout [String: MLXArray]) {
    var renames = [(String, String)]()  // (checkpoint tail, module tail)
    for proj in ["gate_up", "down"] {
        for suffix in ["theta", "pairs", "channel_scales"] {
            renames.append(
                (".experts.\(proj)_weight_\(suffix)", ".switch_mlp.\(proj)_rot.\(suffix)"))
        }
    }

    for key in Array(weights.keys) {
        guard let (old, new) = renames.first(where: { key.hasSuffix($0.0) }) else { continue }
        var value = weights.removeValue(forKey: key)!
        // The 35B checkpoint ships channel_scales as [1, dims] already, but
        // normalise 1-D just like convertAutoAWQ does for dense rotations.
        if new.hasSuffix("channel_scales"), value.ndim == 1 {
            value = value.reshaped(1, -1)
        }
        weights[String(key.dropLast(old.count)) + new] = value
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

/// Replace SwitchGLU modules with RotateSwitchGLU where the remapped shared
/// rotation keys exist (`…switch_mlp.gate_up_rot.theta` marks one MoE block).
///
/// Must run *before* `patchRotationLayers`: the remapped rotation keys also
/// end in `.theta`, and the dense scan must find a `PairwiseRotation` leaf
/// (which it skips) at those prefixes — not an unpatched non-Linear module.
///
/// Internal (not private) so the swap contract is unit-testable.
func patchMoESwitchGLULayers(
    model: Module, weights: [String: MLXArray], groupSize: Int
) throws {
    let marker = ".gate_up_rot.theta"
    let gluPaths = weights.keys
        .filter { $0.hasSuffix(marker) }
        .map { String($0.dropLast(marker.count)) }
        .sorted()

    guard !gluPaths.isEmpty else { return }

    let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
    var updates = [(String, Module)]()

    for path in gluPaths {
        guard let module = modules[path] else {
            throw ParoQuantError.rotationLayerNotFound(path)
        }
        guard let glu = module as? SwitchGLU else {
            throw ParoQuantError.rotationLayerTypeMismatch(
                path: path,
                actualType: String(describing: type(of: module))
            )
        }

        // Both rotations must arrive complete; a partial set means a broken
        // checkpoint and should fail here, with a key name, not at update().
        let theta = try requireTensor("\(path).gate_up_rot.theta", weights: weights)
        for child in ["gate_up_rot", "down_rot"] {
            for suffix in ["theta", "pairs", "channel_scales"] {
                _ = try requireTensor("\(path).\(child).\(suffix)", weights: weights)
            }
        }

        let replacement = RotateSwitchGLU(
            inputDims: glu.inputDims,
            hiddenDims: glu.hiddenDims,
            numExperts: glu.numExperts,
            groupSize: groupSize,
            krot: theta.dim(0)
        )
        updates.append((path, replacement))
    }

    try model.update(modules: ModuleChildren.unflattened(updates), verify: [.noUnusedKeys])

    let patched = Dictionary(uniqueKeysWithValues: model.namedModules())
    for (path, _) in updates {
        guard patched[path] is RotateSwitchGLU else {
            throw ParoQuantError.rotationLayerPatchFailed(path)
        }
    }
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
        // MoE shared rotations: their keys also end in `.theta`, but they
        // belong to the PairwiseRotation children installed by
        // patchMoESwitchGLULayers, not to a Linear awaiting patching.
        if leafModules[prefix] is PairwiseRotation { continue }

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
///   - tokenizerLoader: Loader for tokenizer.
///   - toolCallFormat: Optional tool-call format for the model configuration.
/// - Returns: A ``ModelContainer`` ready for inference.
public func loadParoQuantModel<T: LanguageModel>(
    from directory: URL,
    typeRegistry: ModelTypeRegistry<T>,
    tokenizerLoader: any TokenizerLoader,
    toolCallFormat: ToolCallFormat? = nil
) async throws -> ModelContainer {
    // Phase wall-clock breakdown, logged once at the end. MLX ops are lazy, so
    // graph-building phases (convert/stack/remap) bill their kernel time to the
    // step-12 `eval(model)` — the phase split separates CPU-side work from that
    // final materialization, it does not attribute kernels to their builders.
    let loadClock = ContinuousClock()
    let loadStart = loadClock.now
    var phaseStart = loadStart
    var phaseTimes: [(String, Double)] = []
    func markPhase(_ name: String) {
        let now = loadClock.now
        phaseTimes.append((name, (now - phaseStart) / .seconds(1)))
        phaseStart = now
    }

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
    if configJSON["text_config"] != nil {
        configJSON = flattenParoQuantTextConfig(configJSON)
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
    markPhase("createModel")

    // 4. EOS token override from generation_config.json
    var eosTokenIds = Set(baseConfig.eosTokenIds?.values ?? [])
    let genConfigURL = directory.appendingPathComponent("generation_config.json")
    let genConfig: GenerationConfigFile? =
        if let genData = try? Data(contentsOf: genConfigURL) {
            try? JSONDecoder().decode(GenerationConfigFile.self, from: genData)
        } else {
            nil
        }
    if let genEos = genConfig?.eosTokenIds?.values {
        eosTokenIds = Set(genEos)
    }

    var config = ModelConfiguration(
        directory: directory, stopStrings: genConfig?.stopStrings,
        toolCallFormat: toolCallFormat)
    config.eosTokenIds = eosTokenIds

    // 5. Load weights. A fresh Prepared Checkpoint (the once-converted
    //    MLX-native form, see `ParoQuantPreparedCheckpoint`) skips steps
    //    6–6c entirely; otherwise load raw safetensors (top-level only; do
    //    not recurse into subdirectories, otherwise nested artefacts like an
    //    HF snapshot cache under the checkpoint dir would be pulled in) and
    //    convert.
    let preparedManifest = try ParoQuantPreparedCheckpoint.currentManifest(
        directory: directory)
    var usedPreparedCheckpoint = false
    var weights = [String: MLXArray]()
    if let prepared = ParoQuantPreparedCheckpoint.load(
        directory: directory, manifest: preparedManifest)
    {
        weights = prepared
        usedPreparedCheckpoint = true
        logger.info("Loaded \(weights.count) weight keys from the Prepared Checkpoint")
        markPhase("preparedLoad")
    } else {
        for url in try ParoQuantPreparedCheckpoint.sourceURLs(in: directory) {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }

        logger.info("Loaded \(weights.count) weight keys from safetensors")
        markPhase("rawLoad")

        // 6. Convert AutoAWQ format → MLX format (BEFORE sanitize)
        if weights.keys.contains(where: { $0.hasSuffix(".qweight") }) {
            convertAutoAWQ(&weights, groupSize: paroConfig.groupSize)
            logger.info("Converted AutoAWQ weights to MLX format")
        }
        markPhase("convertGraph")

        // 6b. Split fused Mamba `in_proj_ba` into `in_proj_b` / `in_proj_a`.
        //     PARO-specific layout detail, so it lives here rather than in the
        //     generic Qwen35 sanitize. Runs before `model.sanitize` so downstream
        //     layers see the already-split keys.
        splitFusedMambaProjections(&weights)

        // 6c. MoE checkpoints: stack the per-expert converted AWQ tensors into
        //     the 3-D switch_mlp layout and remap the shared per-layer rotation
        //     keys onto the nested PairwiseRotation children. Both are key-driven
        //     no-ops for dense checkpoints. The stack stays lazy (graph nodes over
        //     the mmap'd source arrays) until eval — same whole-dict pattern as
        //     the Python reference.
        stackMoEExpertWeights(&weights)
        remapSharedMoERotations(&weights)
        markPhase("moePasses")
    }

    // The Prepared Checkpoint persists this pre-sanitize snapshot: it is the
    // last container-agnostic point (sanitize is model-class-specific, so one
    // artifact serves both the LLM and VLM containers), and everything
    // downstream — patching, quantize passes, update, rotation-state
    // derivation — stays live code on every load.
    let preSanitizeWeights = weights

    // 7. Model-specific sanitization
    weights = model.sanitize(weights: weights)
    markPhase("sanitize")

    // 8a. Swap SwitchGLU → RotateSwitchGLU where shared rotation keys exist.
    //     Must precede the dense `.theta` scan in patchRotationLayers (the
    //     rotation keys also end in `.theta`). The fresh RotateSwitchGLU's
    //     SwitchLinear children are converted to QuantizedSwitchLinear by the
    //     step-9 quantize pass — their stacked checkpoint `.scales` mark them
    //     as checkpoint-quantized, the same mechanism the dense path uses.
    try patchMoESwitchGLULayers(
        model: model, weights: weights, groupSize: paroConfig.groupSize)

    // 8b. Patch dense rotation layers
    try patchRotationLayers(
        model: model, weights: weights,
        bits: paroConfig.bits, groupSize: paroConfig.groupSize
    )
    markPhase("patch")

    // 9. Quantize non-rotation layers in MLX quantized form
    quantize(model: model) { path, module in
        guard module is Quantizable else { return nil }
        guard isCheckpointQuantizedLayer(path: path, weights: weights) else {
            return nil
        }
        return (paroConfig.groupSize, paroConfig.bits, .affine)
    }
    markPhase("quantize")

    // 10. Load checkpoint weights into the patched model
    let parameters = ModuleParameters.unflattened(weights)
    let verify: Module.VerifyUpdate = [.allModelKeysSet, .shapeMismatch]
    try model.update(parameters: parameters, verify: verify)
    markPhase("update")

    // 10b. Finalize rotation-derived state. Must run *after* the checkpoint
    //      update (so theta / pairs / channel_scales hold real values) and
    //      *before* any forward pass. The derived arrays live in underscore-
    //      prefixed fields that Module reflection — and therefore step 12's
    //      eval(model) — skips, so they are materialized here: one batched
    //      eval across all rotation modules (dense RotateQuantizedLinear and
    //      the MoE shared PairwiseRotation leaves) instead of a GPU
    //      round-trip per module. See `RotationStatePreparing`.
    var derivedRotationArrays: [MLXArray] = []
    for (_, layer) in rotationLeafModules(model: model) {
        if let rotation = layer as? RotationStatePreparing {
            derivedRotationArrays.append(contentsOf: rotation.prepareDerivedRotationState())
        }
    }
    eval(derivedRotationArrays)
    markPhase("rotState")

    // 11. Quantize IO embedding path from FP16 weights
    quantize(model: model) { path, module in
        guard isParoQuantIOLayer(path: path, module: module) else {
            return nil
        }
        return (paroConfig.groupSize, paroConfig.bits, .affine)
    }
    markPhase("quantizeIO")

    // 12. Prepare generic inference-only state, then materialize the model.
    // This must follow the final IO-layer topology update above, just like the
    // standard checkpoint-loading path.
    materializeModelForInference(model)
    markPhase("eval")
    logger.info("ParoQuant model loaded and evaluated")
    let totalSeconds = (loadClock.now - loadStart) / .seconds(1)
    let breakdown =
        phaseTimes
        .map { String(format: "%@=%.2fs", $0.0, $0.1) }
        .joined(separator: " ")
    logger.notice(
        "ParoQuant load phases: \(breakdown, privacy: .public) total=\(String(format: "%.2f", totalSeconds), privacy: .public)s"
    )

    // 12b. Persist the Prepared Checkpoint in the background when this load
    //      had to convert. Runs after the materialization above so the
    //      model-bound tensors are already realized (the dict shares them
    //      by reference); readiness never waits on the write, and a failed
    //      or interrupted write only means the next load converts again.
    if !usedPreparedCheckpoint {
        // `[String: MLXArray]` is not Sendable; the boxed dict holds the
        // immutable, already-evaluated checkpoint tensors shared with the
        // live model, read-only on both sides.
        let weightsBox = SendableBox(preSanitizeWeights)
        Task.detached(priority: .utility) {
            ParoQuantPreparedCheckpoint.write(
                weights: weightsBox.consume(), manifest: preparedManifest,
                directory: directory)
        }
    }

    // 13. Load tokenizer
    let tokenizer = try await tokenizerLoader.load(from: directory)

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
