// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

package let modelConversionSidecarPatterns = [
    "*.json", "*.jsonl", "*.jinja", "*.txt", "*.model", "*.tiktoken", "*.py",
]
package let modelConversionDownloadPatterns = ["*.safetensors"] + modelConversionSidecarPatterns

/// How the quantization grid is derived from the source weights.
///
/// This selects a *derivation*, not a storage format: every calibration produces ordinary
/// quantized layers that load and run without any decode-side support.
public enum ModelConversionQuantizationCalibration: Sendable, Equatable {
    /// Derive the grid from the weights, as MLX does by default.
    case standard

    /// Reproduce the `Q4_0` grid: symmetric, 32-element groups, scaled by the signed
    /// element of largest magnitude.
    ///
    /// Use this for checkpoints whose weights were already trained against that grid.
    /// Re-deriving a grid from such weights discards the alignment they were trained for;
    /// reproducing it keeps them on their intended lattice.
    ///
    /// Requires 4 bits, a group size of 32, and `QuantizationMode.affine`. Supplying a
    /// conflicting value throws ``ModelConversionError/incompatibleCalibration(_:_:)``
    /// rather than silently overriding it.
    ///
    /// Only `Linear` layers are calibrated. Other quantizable layers are still quantized to
    /// the same 4-bit/group-32 affine grid, but derive it the standard way, because
    /// `QuantizedEmbedding` has no initializer accepting precomputed scales and biases.
    /// The stored format is identical either way, so this affects accuracy, not loadability.
    case q4Zero
}

/// Quantization settings for one conversion pass or layer.
public struct ModelConversionQuantization: Sendable, Equatable {
    /// Quantization bit depth.
    ///
    /// `nil` lets MLX use the default for the selected quantization mode.
    public var bits: Int?

    /// Quantization group size.
    ///
    /// `nil` lets MLX use the default for the selected quantization mode.
    public var groupSize: Int?

    /// Quantization mode.
    public var mode: QuantizationMode

    /// How the quantization grid is derived. Defaults to ``ModelConversionQuantizationCalibration/standard``.
    public var calibration: ModelConversionQuantizationCalibration

    public init(
        bits: Int? = nil, groupSize: Int? = nil, mode: QuantizationMode = .affine,
        calibration: ModelConversionQuantizationCalibration = .standard
    ) {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
        self.calibration = calibration
    }
}

/// Per-layer quantization decision.
public enum ModelConversionQuantizationDecision: Sendable, Equatable {
    /// Do not quantize this layer.
    case skip

    /// Quantize this layer, using the global conversion settings when the associated value is nil.
    case quantize(ModelConversionQuantization? = nil)
}

/// Predicate used to customize layer quantization.
public typealias ModelConversionQuantizationPredicate =
    @Sendable (
        _ path: String, _ module: Module
    ) -> ModelConversionQuantizationDecision

/// Options controlling model conversion.
public struct ModelConversionOptions: Sendable {
    /// Default quantization settings.
    public var quantization: ModelConversionQuantization

    /// Maximum safetensors shard size in bytes.
    public var maxShardSize: Int64

    /// Remove and recreate the output directory if it already exists.
    public var overwriteExistingOutput: Bool

    /// Optional per-layer quantization customization.
    public var quantizationPredicate: ModelConversionQuantizationPredicate?

    public var bits: Int? {
        get { quantization.bits }
        set { quantization.bits = newValue }
    }

    public var groupSize: Int? {
        get { quantization.groupSize }
        set { quantization.groupSize = newValue }
    }

    public var mode: QuantizationMode {
        get { quantization.mode }
        set { quantization.mode = newValue }
    }

    public var calibration: ModelConversionQuantizationCalibration {
        get { quantization.calibration }
        set { quantization.calibration = newValue }
    }

    public init(
        bits: Int? = nil,
        groupSize: Int? = nil,
        mode: QuantizationMode = .affine,
        calibration: ModelConversionQuantizationCalibration = .standard,
        maxShardSize: Int64 = 5 * 1024 * 1024 * 1024,
        overwriteExistingOutput: Bool = false,
        quantizationPredicate: ModelConversionQuantizationPredicate? = nil
    ) {
        self.quantization = .init(
            bits: bits, groupSize: groupSize, mode: mode, calibration: calibration)
        self.maxShardSize = maxShardSize
        self.overwriteExistingOutput = overwriteExistingOutput
        self.quantizationPredicate = quantizationPredicate
    }
}

/// Coarse conversion stage for progress callbacks.
public enum ModelConversionStage: String, Sendable {
    case downloading
    case copyingFiles
    case loadingWeights
    case quantizing
    case savingWeights
    case updatingConfiguration
}

/// Progress update emitted by the Swift conversion pipeline.
///
/// This intentionally stays coarse-grained so applications can map it onto their own UI without
/// inheriting implementation details. Download progress is forwarded as a fractional value when
/// the downloader provides one.
public struct ModelConversionProgress: Sendable {
    public let stage: ModelConversionStage
    public let fractionCompleted: Double?
    public let message: String?

    public init(
        stage: ModelConversionStage,
        fractionCompleted: Double? = nil,
        message: String? = nil
    ) {
        self.stage = stage
        self.fractionCompleted = fractionCompleted
        self.message = message
    }
}

/// Result returned after converting and saving a model.
public struct ModelConversionResult: Sendable {
    /// Directory containing the converted model files.
    public let outputDirectory: URL

    /// URL of the first written safetensors weights file.
    public let weightsURL: URL

    /// URLs of the written safetensors shard files.
    public let weightsURLs: [URL]

    public init(outputDirectory: URL, weightsURL: URL) {
        self.init(outputDirectory: outputDirectory, weightsURLs: [weightsURL])
    }

    public init(outputDirectory: URL, weightsURLs: [URL]) {
        precondition(!weightsURLs.isEmpty)
        self.outputDirectory = outputDirectory
        self.weightsURL = weightsURLs[0]
        self.weightsURLs = weightsURLs
    }
}

/// Errors thrown by model conversion helpers.
public enum ModelConversionError: LocalizedError, Equatable {
    case outputDirectoryExists(URL)
    case outputDirectoryMatchesSource(URL)
    case noSafetensorsFiles(URL)
    case unsupportedPyTorchWeights(URL)
    case sourceAlreadyQuantized(URL)
    case invalidShardSize(Int64)
    case incompatibleCalibration(
        ModelConversionQuantizationCalibration, ModelConversionQuantization)

    public var errorDescription: String? {
        switch self {
        case .outputDirectoryExists(let directory):
            "Cannot save to \(directory.path) because it already exists. Delete it, choose a new path, or set overwriteExistingOutput."
        case .outputDirectoryMatchesSource(let directory):
            "Cannot convert in place at \(directory.path). Choose an output directory separate from the source model and tokenizer directories."
        case .noSafetensorsFiles(let directory):
            "No safetensors weights were found in \(directory.path)."
        case .unsupportedPyTorchWeights(let directory):
            "PyTorch .bin weights in \(directory.path) are not supported. Use a safetensors model."
        case .sourceAlreadyQuantized(let directory):
            "The source model in \(directory.path) is already quantized. Re-quantizing is not supported by this conversion helper."
        case .invalidShardSize(let shardSize):
            "Shard size must be greater than zero, got \(shardSize)."
        case .incompatibleCalibration(let calibration, let quantization):
            "\(calibration) calibration requires bits 4, group size 32, and affine mode, but got bits \(quantization.bits.map(String.init) ?? "default"), group size \(quantization.groupSize.map(String.init) ?? "default"), mode \(quantization.mode)."
        }
    }
}

/// Quantize and save an already-instantiated model from a compatible safetensors directory.
///
/// The function loads safetensors weights, runs the model's sanitizer through
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:weightFileSelection:)``, applies quantization,
/// writes safetensors shards and an index, copies tokenizer/config sidecar files, and updates
/// `config.json` with the requested quantization block.
public func convert(
    modelDirectory: URL,
    tokenizerDirectory: URL? = nil,
    model: BaseLanguageModel,
    to outputDirectory: URL,
    options: ModelConversionOptions = .init(),
    progressHandler: @Sendable (ModelConversionProgress) -> Void = { _ in },
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws -> ModelConversionResult {
    try validateConvertibleWeights(in: modelDirectory)
    try validateSourceConfigurationForModelConversion(in: modelDirectory)
    if perLayerQuantization != nil {
        throw ModelConversionError.sourceAlreadyQuantized(modelDirectory)
    }
    try validateModelConversionCalibration(options.quantization)
    // Resolved once, before any filesystem mutation, and reused below. Re-running the
    // predicate after the output exists would let a stateful one pass preflight and then
    // fail with a half-written model on disk.
    let quantizationDecisions = try modelConversionQuantizationDecisions(
        model: model, options: options)
    try validateModelConversionOutputDirectory(
        outputDirectory,
        modelDirectory: modelDirectory,
        tokenizerDirectory: tokenizerDirectory)

    try prepareOutputDirectoryForModelConversion(
        outputDirectory, overwriteExistingOutput: options.overwriteExistingOutput)

    progressHandler(.init(stage: .copyingFiles))
    try copyModelConversionFiles(
        modelDirectory: modelDirectory,
        tokenizerDirectory: tokenizerDirectory,
        to: outputDirectory)

    progressHandler(.init(stage: .loadingWeights))
    try loadWeights(
        modelDirectory: modelDirectory,
        model: model,
        perLayerQuantization: perLayerQuantization)

    progressHandler(.init(stage: .quantizing))
    let quantizationResult = quantizeForModelConversion(
        model: model, options: options, decisions: quantizationDecisions)
    eval(model)

    progressHandler(.init(stage: .savingWeights))
    let weightsURLs = try saveModelConversionWeights(
        model.parameters().flattened(),
        to: outputDirectory,
        maxShardSize: options.maxShardSize,
        metadata: modelConversionSafetensorsMetadata(for: model))

    progressHandler(.init(stage: .updatingConfiguration))
    try updateModelConfigWithQuantization(
        at: outputDirectory,
        quantization: quantizationResult.defaultQuantization,
        layerQuantization: quantizationResult.layerQuantization)

    return ModelConversionResult(outputDirectory: outputDirectory, weightsURLs: weightsURLs)
}

package func prepareOutputDirectoryForModelConversion(
    _ outputDirectory: URL,
    overwriteExistingOutput: Bool
) throws {
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: outputDirectory.path) {
        guard overwriteExistingOutput else {
            throw ModelConversionError.outputDirectoryExists(outputDirectory)
        }
        try fileManager.removeItem(at: outputDirectory)
    }

    try fileManager.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
}

package func validateModelConversionOutputDirectory(
    _ outputDirectory: URL,
    modelDirectory: URL,
    tokenizerDirectory: URL?
) throws {
    let output = canonicalModelConversionDirectory(outputDirectory)
    let model = canonicalModelConversionDirectory(modelDirectory)
    if modelConversionDirectoriesOverlap(output, model) {
        throw ModelConversionError.outputDirectoryMatchesSource(outputDirectory)
    }

    if let tokenizerDirectory {
        let tokenizer = canonicalModelConversionDirectory(tokenizerDirectory)
        if modelConversionDirectoriesOverlap(output, tokenizer) {
            throw ModelConversionError.outputDirectoryMatchesSource(outputDirectory)
        }
    }
}

private func canonicalModelConversionDirectory(_ directory: URL) -> URL {
    directory.resolvingSymlinksInPath().standardizedFileURL
}

private func modelConversionDirectoriesOverlap(_ first: URL, _ second: URL) -> Bool {
    let firstComponents = first.pathComponents
    let secondComponents = second.pathComponents

    return firstComponents.starts(with: secondComponents)
        || secondComponents.starts(with: firstComponents)
}

package func removeExistingModelWeights(in outputDirectory: URL) throws {
    try removeModelConversionWeights(in: outputDirectory)
}

/// Quantize and save an already-instantiated model from a compatible safetensors directory.
public func convert(
    modelDirectory: URL,
    tokenizerDirectory: URL? = nil,
    model: BaseLanguageModel,
    to outputDirectory: URL,
    bits: Int? = nil,
    groupSize: Int? = nil,
    mode: QuantizationMode = .affine,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws -> ModelConversionResult {
    try convert(
        modelDirectory: modelDirectory,
        tokenizerDirectory: tokenizerDirectory,
        model: model,
        to: outputDirectory,
        options: .init(bits: bits, groupSize: groupSize, mode: mode),
        perLayerQuantization: perLayerQuantization)
}

package func resolveForModelConversion(
    configuration: ModelConfiguration,
    from downloader: any Downloader,
    useLatest: Bool,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> ResolvedModelConfiguration {
    let modelDirectory: URL
    switch configuration.id {
    case .id(let id, let revision):
        modelDirectory = try await downloader.download(
            id: id,
            revision: revision,
            matching: modelConversionDownloadPatterns,
            useLatest: useLatest,
            progressHandler: progressHandler)
    case .directory(let directory):
        modelDirectory = directory
    }

    let tokenizerDirectory: URL
    switch configuration.tokenizerSource {
    case .id(let id, let revision):
        tokenizerDirectory = try await downloader.download(
            id: id,
            revision: revision,
            matching: modelConversionSidecarPatterns,
            useLatest: useLatest,
            progressHandler: { _ in })
    case .directory(let directory):
        tokenizerDirectory = directory
    case nil:
        tokenizerDirectory = modelDirectory
    }

    return configuration.resolved(
        modelDirectory: modelDirectory,
        tokenizerDirectory: tokenizerDirectory)
}

package func validateSourceConfigurationForModelConversion(in directory: URL) throws {
    let configURL = directory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL) else {
        return
    }
    if sourceConfigurationIsQuantized(data) {
        throw ModelConversionError.sourceAlreadyQuantized(directory)
    }
}

package func validateConvertibleWeights(in directory: URL) throws {
    let fileManager = FileManager.default
    let files = try fileManager.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])

    let safetensors = files.contains { $0.pathExtension == "safetensors" }
    if safetensors {
        if try safetensorsContainQuantizedWeights(files) {
            throw ModelConversionError.sourceAlreadyQuantized(directory)
        }
        return
    }

    let pytorchWeights = files.contains { $0.pathExtension == "bin" }
    if pytorchWeights {
        throw ModelConversionError.unsupportedPyTorchWeights(directory)
    }

    throw ModelConversionError.noSafetensorsFiles(directory)
}

private func safetensorsContainQuantizedWeights(_ files: [URL]) throws -> Bool {
    for file in files where file.pathExtension == "safetensors" {
        if try safetensorsFileContainsQuantizedWeights(file) {
            return true
        }
    }
    return false
}

package func safetensorsFileContainsQuantizedWeights(_ url: URL) throws -> Bool {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    guard let headerSizeData = try handle.read(upToCount: 8),
        headerSizeData.count == 8
    else {
        return false
    }

    let headerSize = headerSizeData.enumerated().reduce(UInt64(0)) { result, byte in
        result | (UInt64(byte.element) << UInt64(byte.offset * 8))
    }
    guard headerSize <= 100 * 1024 * 1024,
        let headerData = try handle.read(upToCount: Int(headerSize)),
        headerData.count == Int(headerSize),
        let json = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any]
    else {
        return false
    }

    return json.keys.contains { key in
        key.hasSuffix(".scales")
    }
}

package func copyModelConversionFiles(
    modelDirectory: URL,
    tokenizerDirectory: URL?,
    to outputDirectory: URL
) throws {
    try copyModelConversionSidecars(from: modelDirectory, to: outputDirectory, copiesConfig: true)

    if let tokenizerDirectory,
        tokenizerDirectory.standardizedFileURL != modelDirectory.standardizedFileURL
    {
        try copyModelConversionSidecars(
            from: tokenizerDirectory, to: outputDirectory, copiesConfig: false)
    }
}

private func copyModelConversionSidecars(
    from sourceDirectory: URL,
    to outputDirectory: URL,
    copiesConfig: Bool
) throws {
    let fileManager = FileManager.default
    let resourceKeys: [URLResourceKey] = [.isDirectoryKey]
    let files = try fileManager.contentsOfDirectory(
        at: sourceDirectory,
        includingPropertiesForKeys: resourceKeys,
        options: [.skipsHiddenFiles])

    let copiedExtensions: Set<String> = [
        "json", "jsonl", "jinja", "txt", "model", "tiktoken", "py",
    ]

    for sourceURL in files {
        let resourceValues = try sourceURL.resourceValues(forKeys: Set(resourceKeys))
        if resourceValues.isDirectory == true {
            continue
        }

        let filename = sourceURL.lastPathComponent
        let lowercasedName = filename.lowercased()
        if !copiesConfig && lowercasedName == "config.json" {
            continue
        }
        if lowercasedName.hasSuffix(".safetensors.index.json") {
            continue
        }
        if sourceURL.pathExtension == "safetensors" {
            continue
        }
        if !copiedExtensions.contains(sourceURL.pathExtension.lowercased()) {
            continue
        }

        let destinationURL = outputDirectory.appendingPathComponent(filename)
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }
        try fileManager.copyItem(at: sourceURL, to: destinationURL)
    }
}

package func updateModelConfigWithQuantization(
    at outputDirectory: URL,
    bits: Int?,
    groupSize: Int?,
    mode: QuantizationMode
) throws {
    try updateModelConfigWithQuantization(
        at: outputDirectory,
        quantization: .init(bits: bits, groupSize: groupSize, mode: mode),
        layerQuantization: [:])
}

package func updateModelConfigWithQuantization(
    at outputDirectory: URL,
    quantization: ModelConversionQuantization,
    layerQuantization: [String: ModelConversionQuantizationDecision]
) throws {
    let configURL = outputDirectory.appendingPathComponent("config.json")
    guard FileManager.default.fileExists(atPath: configURL.path) else {
        return
    }

    let data = try Data(contentsOf: configURL)
    var json = try JSONDecoder.json5().decode([String: JSONValue].self, from: data)

    let effectiveQuantization = effectiveModelConversionQuantization(quantization)
    var quantizationJSON: [String: JSONValue] = [
        "bits": .int(effectiveQuantization.bits),
        "group_size": .int(effectiveQuantization.groupSize),
        "mode": .string(quantizationModeName(quantization.mode)),
    ]
    for (path, decision) in layerQuantization.sorted(by: { $0.key < $1.key }) {
        switch decision {
        case .skip:
            quantizationJSON[path] = .bool(false)
        case .quantize(let value):
            let value = value ?? quantization
            let effectiveValue = effectiveModelConversionQuantization(value)
            quantizationJSON[path] = .object([
                "bits": .int(effectiveValue.bits),
                "group_size": .int(effectiveValue.groupSize),
                "mode": .string(quantizationModeName(value.mode)),
            ])
        }
    }
    let quantizationValue = JSONValue.object(quantizationJSON)
    json["quantization"] = quantizationValue
    json["quantization_config"] = quantizationValue

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let updatedData = try encoder.encode(json)
    try updatedData.write(to: configURL)
}

package func sourceConfigurationIsQuantized(_ data: Data) -> Bool {
    guard let json = try? JSONDecoder.json5().decode([String: JSONValue].self, from: data) else {
        return false
    }
    return json["quantization"] != nil || json["quantization_config"] != nil
}

private func removeModelConversionWeights(in outputDirectory: URL) throws {
    guard FileManager.default.fileExists(atPath: outputDirectory.path) else {
        return
    }

    let files = try FileManager.default.contentsOfDirectory(
        at: outputDirectory,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles])

    for file in files {
        let resourceValues = try file.resourceValues(forKeys: [.isDirectoryKey])
        if resourceValues.isDirectory == true {
            continue
        }

        if file.pathExtension == "safetensors"
            || file.lastPathComponent.lowercased().hasSuffix(".safetensors.index.json")
        {
            try FileManager.default.removeItem(at: file)
        }
    }
}

private func quantizeForModelConversion(
    model: BaseLanguageModel,
    options: ModelConversionOptions,
    decisions: [(String, Module, ModelConversionQuantizationDecision)]
) -> ModelConversionQuantizationResult {
    var layerQuantization = [String: ModelConversionQuantizationDecision]()
    let defaultQuantization = options.quantization
    var effectiveDefaultQuantization = effectiveModelConversionQuantization(defaultQuantization)

    let updates =
        decisions
        .compactMap { path, module, decision -> (String, Module)? in
            let quantization: ModelConversionQuantization
            let usesDefaultQuantization: Bool
            switch decision {
            case .skip:
                layerQuantization[path] = .skip
                return nil
            case .quantize(let override):
                quantization = override ?? defaultQuantization
                usesDefaultQuantization = override == nil
            }

            let effectiveQuantization = effectiveModelConversionQuantization(quantization)
            guard
                isConvertibleQuantizationTarget(
                    module, groupSize: effectiveQuantization.groupSize)
            else {
                return nil
            }

            guard
                let result = quantizeLayerForModelConversion(
                    layer: module, quantization: quantization)
            else {
                return nil
            }

            if usesDefaultQuantization && defaultQuantization.hasUnresolvedDefaults {
                effectiveDefaultQuantization = result.quantization
            }
            if result.quantization != effectiveDefaultQuantization {
                layerQuantization[path] = .quantize(
                    result.quantization.asModelConversionQuantization)
            }
            return (path, result.module)
        }

    model.update(modules: ModuleChildren.unflattened(updates))

    return .init(
        defaultQuantization: effectiveDefaultQuantization.asModelConversionQuantization,
        layerQuantization: layerQuantization)
}

/// Bits, group size, and mode that ``ModelConversionQuantizationCalibration/q4Zero`` requires.
private let q4ZeroBits = 4
private let q4ZeroGroupSize = 32

/// Magnitude of the most-negative representable code. The grid spans `[-8, 7]`, so the scale
/// is `extremum / -8` and the affine bias that recentres it is `-8 * scale`.
private let q4ZeroNegativeExtent: Float = 8

/// Validates that a quantization request is consistent with its calibration.
///
/// Called before any output directory is created or removed so an unusable request fails
/// without touching the filesystem.
func validateModelConversionCalibration(_ quantization: ModelConversionQuantization) throws {
    switch quantization.calibration {
    case .standard:
        return
    case .q4Zero:
        let compatible =
            (quantization.bits ?? q4ZeroBits) == q4ZeroBits
            && (quantization.groupSize ?? q4ZeroGroupSize) == q4ZeroGroupSize
            && quantization.mode == .affine
        if !compatible {
            throw ModelConversionError.incompatibleCalibration(.q4Zero, quantization)
        }
    }
}

/// Resolves every layer's quantization decision and validates it.
///
/// Runs before the output directory is created or removed, so a conflicting per-layer
/// override fails without leaving a half-written model behind.
func modelConversionQuantizationDecisions(
    model: BaseLanguageModel,
    options: ModelConversionOptions
) throws -> [(String, Module, ModelConversionQuantizationDecision)] {
    var resolved = [(String, Module, ModelConversionQuantizationDecision)]()
    for (path, module) in model.leafModules().flattened() {
        let decision = options.quantizationPredicate?(path, module) ?? .quantize()
        if case .quantize(let override) = decision, let override {
            try validateModelConversionCalibration(override)
        }
        resolved.append((path, module, decision))
    }
    return resolved
}

/// Reproduces the `Q4_0` grid for one weight tensor.
///
/// Per 32-element group along the input axis the scale comes from the **signed** element of
/// largest magnitude, `d = extremum / -8`. Using `-max(abs(w))/8` instead flips the sign on
/// every group whose extremum is negative — close to half of them — which silently produces a
/// different grid rather than an error. `codeSignedExtremumIsNotAbsMax` pins that apart.
///
/// Returns MLX's affine triplet: packed 4-bit codes, `scales = d`, and `biases = -8 * d`.
func q4ZeroQuantized(_ weight: MLXArray) -> (weight: MLXArray, scales: MLXArray, biases: MLXArray) {
    let rows = weight.dim(0)
    let columns = weight.dim(-1)
    let groups = columns / q4ZeroGroupSize

    let grouped = weight.reshaped(rows * groups, q4ZeroGroupSize).asType(.float32)

    // Signed element of largest magnitude, per group.
    let magnitudes = MLX.abs(grouped)
    let extremumIndex = MLX.argMax(magnitudes, axis: -1, keepDims: true)
    let extremum = MLX.takeAlong(grouped, extremumIndex, axis: -1)

    let scales = extremum / -q4ZeroNegativeExtent
    let isZero = scales .== 0
    let inverse = MLX.where(
        isZero, MLXArray(Float(0)), 1 / MLX.where(isZero, MLXArray(Float(1)), scales))

    // `grouped * inverse` lands in [-8, 8], so adding 8.5 is always positive and floor()
    // equals the reference encoder's trunc(). Rounding half-to-even would disagree on ties.
    let offset = grouped * inverse + (q4ZeroNegativeExtent + 0.5)
    let codes = MLX.clip(MLX.floor(offset), min: 0, max: 15).asType(.uint32)

    // Pack 8 consecutive codes per uint32, low nibble first. The nibbles never overlap, so
    // summing the shifted codes is exactly a bitwise OR.
    let codesPerWord = 32 / q4ZeroBits
    let packedGroups = codes.reshaped(rows, columns / codesPerWord, codesPerWord)
    let shifts = (MLXArray(0 ..< codesPerWord) * q4ZeroBits).asType(.uint32).reshaped(
        1, 1, codesPerWord)
    let packed = MLX.sum(packedGroups << shifts, axis: 2).asType(.uint32)

    let scaleGrid = scales.reshaped(rows, groups).asType(weight.dtype)
    return (packed, scaleGrid, (scaleGrid * -q4ZeroNegativeExtent).asType(weight.dtype))
}

private func isConvertibleQuantizationTarget(_ module: Module, groupSize: Int) -> Bool {
    if module is Quantized {
        return false
    }
    if let linear = module as? Linear {
        return linear.weight.dim(-1) % groupSize == 0
    }
    if let embedding = module as? Embedding {
        return embedding.weight.dim(-1) % groupSize == 0
    }
    return module is Quantizable
}

private func quantizeLayerForModelConversion(
    layer: Module,
    quantization: ModelConversionQuantization
) -> (module: Module, quantization: EffectiveModelConversionQuantization)? {
    if layer is Quantized {
        return nil
    }

    let effectiveQuantization = effectiveModelConversionQuantization(quantization)
    let mode = quantization.mode

    if let linear = layer as? Linear {
        if quantization.calibration == .q4Zero {
            guard linear.weight.dim(-1) % q4ZeroGroupSize == 0 else { return nil }
            let (weight, scales, biases) = q4ZeroQuantized(linear.weight)
            let resolved = EffectiveModelConversionQuantization(
                bits: q4ZeroBits, groupSize: q4ZeroGroupSize, mode: .affine)
            return (
                QuantizedLinear(
                    weight: weight,
                    bias: linear.bias,
                    scales: scales,
                    biases: biases,
                    groupSize: q4ZeroGroupSize,
                    bits: q4ZeroBits,
                    mode: .affine),
                resolved
            )
        }

        let (weight, scales, biases) = MLX.quantized(
            linear.weight, groupSize: quantization.groupSize, bits: quantization.bits, mode: mode)
        let actualQuantization =
            inferModelConversionQuantization(
                originalWeight: linear.weight,
                quantizedWeight: weight,
                scales: scales,
                mode: mode) ?? effectiveQuantization
        return (
            QuantizedLinear(
                weight: weight,
                bias: linear.bias,
                scales: scales,
                biases: biases,
                groupSize: actualQuantization.groupSize,
                bits: actualQuantization.bits,
                mode: mode),
            actualQuantization
        )
    }

    if let embedding = layer as? Embedding {
        return (
            QuantizedEmbedding(
                weight: embedding.weight,
                groupSize: effectiveQuantization.groupSize,
                bits: effectiveQuantization.bits,
                mode: mode),
            effectiveQuantization
        )
    }

    guard
        let quantized = quantizeSingle(
            layer: layer,
            groupSize: effectiveQuantization.groupSize,
            bits: effectiveQuantization.bits,
            mode: mode)
    else {
        return nil
    }
    return (quantized, effectiveQuantization)
}

struct EffectiveModelConversionQuantization: Equatable {
    var bits: Int
    var groupSize: Int
    var mode: QuantizationMode

    var asModelConversionQuantization: ModelConversionQuantization {
        .init(bits: bits, groupSize: groupSize, mode: mode)
    }
}

private struct ModelConversionQuantizationResult {
    var defaultQuantization: ModelConversionQuantization
    var layerQuantization: [String: ModelConversionQuantizationDecision]
}

extension ModelConversionQuantization {
    fileprivate var hasUnresolvedDefaults: Bool {
        guard calibration == .standard else { return false }
        return bits == nil || groupSize == nil
    }
}

func effectiveModelConversionQuantization(
    _ quantization: ModelConversionQuantization
) -> EffectiveModelConversionQuantization {
    // A calibration fixes its own grid geometry. Falling through to the mode defaults here
    // would resolve q4Zero to group size 64 and then skip linears whose input width is a
    // multiple of 32 but not 64.
    if quantization.calibration == .q4Zero {
        return .init(bits: q4ZeroBits, groupSize: q4ZeroGroupSize, mode: .affine)
    }
    return .init(
        bits: quantization.bits ?? defaultModelConversionBits(for: quantization.mode),
        groupSize: quantization.groupSize
            ?? defaultModelConversionGroupSize(for: quantization.mode),
        mode: quantization.mode)
}

private func defaultModelConversionBits(for mode: QuantizationMode) -> Int {
    switch mode {
    case .mxfp8:
        8
    case .affine, .mxfp4, .nvfp4:
        4
    @unknown default:
        4
    }
}

private func defaultModelConversionGroupSize(for mode: QuantizationMode) -> Int {
    switch mode {
    case .mxfp4, .mxfp8, .nvfp4:
        32
    case .affine:
        64
    @unknown default:
        64
    }
}

private func inferModelConversionQuantization(
    originalWeight: MLXArray,
    quantizedWeight: MLXArray,
    scales: MLXArray,
    mode: QuantizationMode
) -> EffectiveModelConversionQuantization? {
    let inputDimensions = originalWeight.dim(-1)
    let scaleGroups = scales.dim(-1)
    guard inputDimensions > 0, scaleGroups > 0 else {
        return nil
    }

    let quantizedInputDimensions = quantizedWeight.dim(-1)
    let groupSize = inputDimensions / scaleGroups
    let bits = quantizedInputDimensions * 32 / inputDimensions
    guard groupSize > 0, bits > 0 else {
        return nil
    }

    return .init(bits: bits, groupSize: groupSize, mode: mode)
}

package func makeModelConversionShards(
    _ weights: [(String, MLXArray)],
    maxShardSize: Int64
) throws -> [[(String, MLXArray)]] {
    guard maxShardSize > 0 else {
        throw ModelConversionError.invalidShardSize(maxShardSize)
    }

    var shards = [[(String, MLXArray)]]()
    var currentShard = [(String, MLXArray)]()
    var currentShardSize: Int64 = 0

    for weight in weights {
        let itemSize = Int64(weight.1.nbytes)
        if !currentShard.isEmpty && currentShardSize + itemSize > maxShardSize {
            shards.append(currentShard)
            currentShard = []
            currentShardSize = 0
        }
        currentShard.append(weight)
        currentShardSize += itemSize
    }

    if !currentShard.isEmpty {
        shards.append(currentShard)
    }

    return shards
}

package func saveModelConversionWeights(
    _ weights: [(String, MLXArray)],
    to outputDirectory: URL,
    maxShardSize: Int64,
    metadata: [String: String] = [:]
) throws -> [URL] {
    let shards = try makeModelConversionShards(weights, maxShardSize: maxShardSize)
    let totalShards = shards.count
    var weightMap = [String: String]()
    let totalSize = weights.reduce(Int64(0)) { $0 + Int64($1.1.nbytes) }
    var weightsURLs = [URL]()

    for (index, shard) in shards.enumerated() {
        let filename =
            totalShards == 1
            ? "model.safetensors"
            : String(format: "model-%05d-of-%05d.safetensors", index + 1, totalShards)
        let url = outputDirectory.appendingPathComponent(filename)
        let arrays = Dictionary(uniqueKeysWithValues: shard)
        eval(Array(arrays.values))
        try save(arrays: arrays, metadata: modelConversionSafetensorsMetadata(metadata), url: url)

        for (key, _) in shard {
            weightMap[key] = filename
        }
        weightsURLs.append(url)
    }

    try saveModelConversionIndex(weightMap: weightMap, totalSize: totalSize, to: outputDirectory)
    return weightsURLs
}

package func modelConversionSafetensorsMetadata(for model: BaseLanguageModel) -> [String: String] {
    if let provider = model as? ModelConversionMetadataProvider {
        return modelConversionSafetensorsMetadata(provider.modelConversionMetadata)
    }
    return modelConversionSafetensorsMetadata([:])
}

package func modelConversionSafetensorsMetadata(_ metadata: [String: String]) -> [String: String] {
    var result = metadata
    result["format"] = "mlx"
    return result
}

package func saveModelConversionIndex(
    weightMap: [String: String],
    totalSize: Int64,
    to outputDirectory: URL
) throws {
    let sortedWeightMap = Dictionary(
        uniqueKeysWithValues: weightMap.sorted(by: { $0.key < $1.key }))
    let indexData: [String: Any] = [
        "metadata": ["total_size": totalSize],
        "weight_map": sortedWeightMap,
    ]
    let data = try JSONSerialization.data(
        withJSONObject: indexData, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: outputDirectory.appendingPathComponent("model.safetensors.index.json"))
}

private func quantizationModeName(_ mode: QuantizationMode) -> String {
    switch mode {
    case .affine:
        "affine"
    case .mxfp4:
        "mxfp4"
    case .mxfp8:
        "mxfp8"
    case .nvfp4:
        "nvfp4"
    @unknown default:
        String(describing: mode)
    }
}
