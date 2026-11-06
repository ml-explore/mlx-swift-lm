// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Rotation

/// A signed, blockwise Walsh–Hadamard rotation `R = H·S / √n` of an array's last axis.
///
/// `H` is the Sylvester Hadamard matrix of order `n`, the block size, and `S` is a diagonal of
/// ±1 signs. The transform runs in float32 and returns the input's dtype, like the reference.
public struct SignedBlockHadamard: Sendable, Equatable {

    /// Elements per Hadamard block along the last axis. A power of two.
    public let blockSize: Int

    private let scale: Float

    /// Creates a rotation over blocks of `blockSize` elements.
    public init(blockSize: Int) {
        precondition(Self.isValidBlockSize(blockSize), "blockSize must be a power of two")
        self.blockSize = blockSize
        self.scale = 1 / Float(blockSize).squareRoot()
    }

    /// Whether `blockSize` is a positive power of two, the only order Sylvester's construction
    /// defines.
    public static func isValidBlockSize(_ blockSize: Int) -> Bool {
        blockSize > 0 && blockSize & (blockSize - 1) == 0
    }

    /// `R·x` along the last axis: flip signs, then transform each block.
    public func forward(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        transform(x.asType(.float32) * signs).asType(x.dtype)
    }

    /// `Rᵀ·x` along the last axis: transform each block, then flip signs.
    public func inverse(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        (transform(x.asType(.float32)) * signs).asType(x.dtype)
    }

    private func transform(_ x: MLXArray) -> MLXArray {
        precondition(x.dim(-1) % blockSize == 0, "last axis must be a multiple of blockSize")
        let shape = x.shape
        return hadamardTransform(x.reshaped(-1, blockSize), scale: scale).reshaped(shape)
    }
}

// MARK: - Layers

/// A `QuantizedLinear` whose weights were quantized in a rotated input basis.
///
/// The forward pass rotates the input with ``SignedBlockHadamard/forward(_:signs:)`` before the
/// packed matmul. `signs` loads from the checkpoint like any other parameter.
open class HadamardQuantizedLinear: QuantizedLinear {

    /// The rotation folded into `weight`.
    public let rotation: SignedBlockHadamard

    /// The rotation's ±1 signs, one per input feature.
    public let signs: MLXArray

    /// Creates the layer from a checkpoint's packed arrays.
    public init(
        weight: MLXArray, bias: MLXArray? = nil, scales: MLXArray, biases: MLXArray?,
        signs: MLXArray, groupSize: Int, bits: Int, mode: QuantizationMode = .affine,
        rotation: SignedBlockHadamard
    ) {
        self.rotation = rotation
        self.signs = signs
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases, groupSize: groupSize,
            bits: bits, mode: mode)
        freeze()
    }

    /// A placeholder shaped like the checkpoint's arrays, which the loader fills.
    public convenience init(
        replacing linear: Linear, groupSize: Int, bits: Int, mode: QuantizationMode = .affine,
        rotation: SignedBlockHadamard
    ) {
        let (outputs, inputs) = linear.shape
        let groups = inputs / groupSize
        self.init(
            weight: MLXArray.zeros([outputs, inputs * bits / 32], dtype: .uint32),
            bias: linear.bias,
            scales: MLXArray.zeros([outputs, groups], dtype: .float16),
            biases: mode == .affine ? MLXArray.zeros([outputs, groups], dtype: .float16) : nil,
            signs: MLXArray.ones([inputs], dtype: .float32),
            groupSize: groupSize, bits: bits, mode: mode, rotation: rotation)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(rotation.forward(x, signs: signs))
    }
}

/// A `QuantizedEmbedding` whose rows were quantized in a rotated basis.
///
/// Lookup un-rotates the dequantized rows. ``asLinear(_:)`` rotates its input before the packed
/// matmul, so a tied head shares the table as it would an unrotated `QuantizedEmbedding`.
open class HadamardQuantizedEmbedding: QuantizedEmbedding {

    /// The rotation's ±1 signs, one per embedding dimension.
    public let signs: MLXArray

    /// The rotation folded into `weight`.
    public let rotation: SignedBlockHadamard

    /// A placeholder shaped like the checkpoint's arrays, which the loader fills.
    ///
    /// The table quantizes lazily and is never evaluated, as in the loader's own quantize pass.
    public init(
        replacing embedding: Embedding, groupSize: Int, bits: Int,
        mode: QuantizationMode = .affine, rotation: SignedBlockHadamard
    ) {
        self.signs = MLXArray.ones([embedding.shape.1], dtype: .float32)
        self.rotation = rotation
        super.init(weight: embedding.weight, groupSize: groupSize, bits: bits, mode: mode)
        freeze()
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        rotation.inverse(super.callAsFunction(x), signs: signs)
    }

    open override func asLinear(_ x: MLXArray) -> MLXArray {
        super.asLinear(rotation.forward(x, signs: signs))
    }
}

// MARK: - Manifest

/// One module of a rotated checkpoint's manifest.
public struct HadamardQuantizedModule: Codable, Sendable, Equatable {

    /// The module's path relative to the language model, as `config.json` names it.
    public var path: String

    /// The Hadamard block size folded into this module's weights.
    public var block: Int

    /// Whether the module is an embedding table rather than a linear layer.
    public var embedding: Bool

    /// The activation dtype the module was packed for, such as `"float16"`.
    public var dtype: String?

    /// Creates a manifest entry.
    public init(path: String, block: Int, embedding: Bool = false, dtype: String? = nil) {
        self.path = path
        self.block = block
        self.embedding = embedding
        self.dtype = dtype
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        path = try container.decode(String.self, forKey: .path)
        block = try container.decode(Int.self, forKey: .block)
        embedding = try container.decodeIfPresent(Bool.self, forKey: .embedding) ?? false
        dtype = try container.decodeIfPresent(String.self, forKey: .dtype)
    }
}

/// A rotated checkpoint's manifest in `config.json`: the modules that carry a folded rotation
/// and the quantization they share.
///
/// The keys follow the pack format PrismML ships. The base model's configuration sits beside
/// them in the same file.
public struct HadamardQuantizedManifest: Codable, Sendable {

    /// The manifest schema this implementation reads.
    public static let supportedSchemaVersion = 2

    /// The pack's schema version.
    public var schemaVersion: Int

    /// The `model_type` of the architecture the checkpoint runs.
    public var baseModelType: String?

    /// The tensor naming the pack uses, which fixes the module paths.
    public var tensorNamespace: String?

    /// How gated-delta activations are ordered; only `"grouped"` loads without a permutation.
    public var gdnActivationLayout: String?

    /// The modules whose weights carry a folded rotation.
    public var modules: [HadamardQuantizedModule]

    /// The quantization every manifest module shares.
    public var quantization: BaseConfiguration.Quantization

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case baseModelType = "base_model_type"
        case tensorNamespace = "tensor_namespace"
        case gdnActivationLayout = "gdn_activation_layout"
        case modules
        case quantization
    }

    /// Creates a manifest.
    public init(
        schemaVersion: Int = HadamardQuantizedManifest.supportedSchemaVersion,
        baseModelType: String? = nil, tensorNamespace: String? = nil,
        gdnActivationLayout: String? = nil, modules: [HadamardQuantizedModule],
        quantization: BaseConfiguration.Quantization
    ) {
        self.schemaVersion = schemaVersion
        self.baseModelType = baseModelType
        self.tensorNamespace = tensorNamespace
        self.gdnActivationLayout = gdnActivationLayout
        self.modules = modules
        self.quantization = quantization
    }

    func validate(baseModelType expectedType: String, tensorNamespace expectedNamespace: String)
        throws
    {
        guard schemaVersion == Self.supportedSchemaVersion else {
            throw HadamardQuantizedCheckpointError.unsupportedSchemaVersion(schemaVersion)
        }
        guard baseModelType == expectedType else {
            throw HadamardQuantizedCheckpointError.unsupportedBaseModelType(
                baseModelType, expected: expectedType)
        }
        guard tensorNamespace == expectedNamespace else {
            throw HadamardQuantizedCheckpointError.unsupportedTensorNamespace(
                tensorNamespace, expected: expectedNamespace)
        }
        if let layout = gdnActivationLayout, layout != "grouped" {
            throw HadamardQuantizedCheckpointError.unsupportedActivationLayout(layout)
        }
        guard !modules.isEmpty else {
            throw HadamardQuantizedCheckpointError.emptyManifest
        }
        var paths = Set<String>()
        for module in modules where !paths.insert(module.path).inserted {
            throw HadamardQuantizedCheckpointError.duplicateModule(module.path)
        }
        _ = try activationDType()
    }

    /// The one half-precision dtype the declaring entries agree on, or `nil` if none declares one.
    func activationDType() throws -> DType? {
        let declared = Set(modules.compactMap(\.dtype))
        guard let name = declared.first else { return nil }
        guard declared.count == 1 else {
            throw HadamardQuantizedCheckpointError.mixedActivationDTypes(declared.sorted())
        }
        switch name {
        case "float16": return .float16
        case "bfloat16": return .bfloat16
        default: throw HadamardQuantizedCheckpointError.unsupportedActivationDType(name)
        }
    }
}

/// Why a rotated checkpoint cannot be loaded.
public enum HadamardQuantizedCheckpointError: LocalizedError, Equatable {
    case unsupportedSchemaVersion(Int)
    case unsupportedBaseModelType(String?, expected: String)
    case unsupportedTensorNamespace(String?, expected: String)
    case unsupportedActivationLayout(String)
    case emptyManifest
    case duplicateModule(String)
    case mixedActivationDTypes([String])
    case unsupportedActivationDType(String)
    case moduleNotFound(String)
    case moduleNotSubstitutable(String, found: String)
    case unsupportedBlockSize(String, block: Int, width: Int)
    case invalidSigns(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedSchemaVersion(let version):
            "Rotated checkpoint schema_version \(version) is not supported "
                + "(expected \(HadamardQuantizedManifest.supportedSchemaVersion))."
        case .unsupportedBaseModelType(let type, let expected):
            "Rotated checkpoint base_model_type \(type ?? "nil") is not \(expected)."
        case .unsupportedTensorNamespace(let namespace, let expected):
            "Rotated checkpoint tensor_namespace \(namespace ?? "nil") is not \(expected)."
        case .unsupportedActivationLayout(let layout):
            "Rotated checkpoint gdn_activation_layout \(layout) needs a permutation "
                + "this loader does not apply."
        case .emptyManifest:
            "Rotated checkpoint declares no modules."
        case .duplicateModule(let path):
            "Rotated checkpoint lists module \(path) more than once."
        case .mixedActivationDTypes(let names):
            "Rotated checkpoint modules disagree on their activation dtype: "
                + names.joined(separator: ", ") + "."
        case .unsupportedActivationDType(let name):
            "Rotated checkpoint activation dtype \(name) is not supported "
                + "(expected float16 or bfloat16)."
        case .moduleNotFound(let path):
            "Rotated checkpoint names a module the model does not have: \(path)."
        case .moduleNotSubstitutable(let path, let found):
            "Rotated checkpoint module \(path) is a \(found), not a plain Linear or Embedding."
        case .unsupportedBlockSize(let path, let block, let width):
            "Rotated checkpoint module \(path) declares block \(block) for input width \(width); "
                + "a block must be a power of two that divides the width."
        case .invalidSigns(let path):
            "Rotated checkpoint module \(path) has signs other than -1 and +1."
        }
    }
}

// MARK: - Checkpoint policy

/// How a model loads a rotated checkpoint: its validated manifest, applied to the model's layers
/// before the weights load and to the tensors in ``BaseLanguageModel/prepareCheckpoint(_:)``.
///
/// Create the policy before `super.init`, so a manifest the loader cannot honor fails before the
/// model allocates anything. Call ``substituteModules(in:)`` once the base model exists, and pass
/// the base model's prepared checkpoint through ``prepare(_:)``. The `PrismHadamardQwen35`
/// classes in MLXLLM and MLXVLM follow this pattern.
public struct HadamardQuantizedCheckpointPolicy: Sendable {

    /// The checkpoint's manifest.
    public let manifest: HadamardQuantizedManifest

    /// What the model prepends to the manifest's paths, such as `"language_model."`.
    public let pathPrefix: String

    /// The activation dtype the packed modules declare; `nil` leaves unpacked tensors as stored.
    public let activationDType: DType?

    /// Validates `manifest` for a model of `baseModelType` whose module paths follow
    /// `tensorNamespace`.
    ///
    /// - Throws: ``HadamardQuantizedCheckpointError`` when the manifest is not one this loader
    ///   can honor.
    public init(
        manifest: HadamardQuantizedManifest, baseModelType: String, tensorNamespace: String,
        pathPrefix: String = ""
    ) throws {
        try manifest.validate(baseModelType: baseModelType, tensorNamespace: tensorNamespace)
        self.manifest = manifest
        self.pathPrefix = pathPrefix
        self.activationDType = try manifest.activationDType()
    }

    /// Replaces each manifest module of `model` with a rotated placeholder that the loader fills.
    ///
    /// Call it before the weights load. The loader's quantize pass skips the placeholders, which
    /// already conform to `Quantized`.
    ///
    /// - Throws: ``HadamardQuantizedCheckpointError`` when a manifest module is missing, is not a
    ///   plain `Linear` or `Embedding` of the declared kind, or has a block that does not divide
    ///   its input width.
    public func substituteModules(in model: Module) throws {
        let modules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let quantization = manifest.quantization
        var replacements = [(String, Module)]()
        for entry in manifest.modules {
            let path = pathPrefix + entry.path
            guard let module = modules[path] else {
                throw HadamardQuantizedCheckpointError.moduleNotFound(path)
            }
            let replacement: Module
            switch module {
            case let embedding as Embedding
            where entry.embedding && type(of: module) == Embedding.self:
                replacement = HadamardQuantizedEmbedding(
                    replacing: embedding, groupSize: quantization.groupSize,
                    bits: quantization.bits, mode: quantization.mode,
                    rotation: try Self.rotation(entry, at: path, width: embedding.shape.1))
            case let linear as Linear where !entry.embedding && type(of: module) == Linear.self:
                replacement = HadamardQuantizedLinear(
                    replacing: linear, groupSize: quantization.groupSize,
                    bits: quantization.bits, mode: quantization.mode,
                    rotation: try Self.rotation(entry, at: path, width: linear.shape.1))
            default:
                throw HadamardQuantizedCheckpointError.moduleNotSubstitutable(
                    path, found: "\(type(of: module))")
            }
            replacements.append((path, replacement))
        }
        model.update(modules: ModuleChildren.unflattened(replacements))
    }

    /// Checks the checkpoint's sign vectors and casts its unpacked floating-point tensors to the
    /// activation dtype.
    ///
    /// A pack can store norms and small projections in float32 beside float16 packed modules,
    /// which would promote the residual stream to float32. Like mlx-vlm's converter, the cast
    /// skips the packed modules' own tensors and `A_log`, which the recurrence reads in float32.
    ///
    /// - Throws: ``HadamardQuantizedCheckpointError/invalidSigns(_:)`` when a sign vector holds a
    ///   value other than -1 or +1.
    public func prepare(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        let packed = Set(manifest.modules.map { pathPrefix + $0.path })
        try Self.validateSigns(in: checkpoint.weights, packed: packed)
        guard let activationDType else { return checkpoint }

        var prepared = checkpoint
        for (key, value) in checkpoint.weights
        where value.dtype.isFloatingPoint && value.dtype != activationDType
            && !key.hasSuffix("A_log") && !packed.contains(Self.modulePath(of: key))
        {
            prepared.weights[key] = value.asType(activationDType)
        }
        return prepared
    }

    private static func rotation(_ entry: HadamardQuantizedModule, at path: String, width: Int)
        throws -> SignedBlockHadamard
    {
        guard SignedBlockHadamard.isValidBlockSize(entry.block), width % entry.block == 0 else {
            throw HadamardQuantizedCheckpointError.unsupportedBlockSize(
                path, block: entry.block, width: width)
        }
        return SignedBlockHadamard(blockSize: entry.block)
    }

    private static func validateSigns(in weights: [String: MLXArray], packed: Set<String>) throws {
        let signs = packed.sorted().compactMap { path in
            weights[path + ".signs"].map { (path: path, values: $0) }
        }
        guard !signs.isEmpty else { return }

        func isValid(_ values: MLXArray) -> MLXArray { (abs(values) .== 1).all() }
        if stacked(signs.map { isValid($0.values) }).all().item(Bool.self) { return }
        if let invalid = signs.first(where: { !isValid($0.values).item(Bool.self) }) {
            throw HadamardQuantizedCheckpointError.invalidSigns(invalid.path)
        }
    }

    private static func modulePath(of key: String) -> String {
        guard let dot = key.lastIndex(of: ".") else { return key }
        return String(key[..<dot])
    }
}
