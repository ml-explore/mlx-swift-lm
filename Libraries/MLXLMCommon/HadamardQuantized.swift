// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Rotation

/// A signed, blockwise Walsh–Hadamard rotation of an array's last axis.
///
/// Rotated checkpoints fold `R = H·S / √n` into every quantized weight, where `H` is
/// the Sylvester Walsh–Hadamard matrix of order `n` (the block size) and `S` is a
/// diagonal of ±1 signs. Activations take the same rotation before the packed matmul,
/// and embedding rows take the inverse after lookup; a loader that skips either step
/// decodes the checkpoint to plausible garbage rather than failing. The transform runs
/// in float32 and returns the input's dtype, matching the reference runtime.
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

    /// Whether `blockSize` is a positive power of two, the only order Sylvester's
    /// construction defines.
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
/// `weight`, `scales` and `biases` are ordinary affine-quantized arrays. `signs` is the
/// rotation's ±1 vector, one entry per input feature. The forward pass rotates the input
/// with ``SignedBlockHadamard/forward(_:signs:)`` and then runs the packed matmul.
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

    /// A placeholder shaped like the checkpoint's arrays, standing in for `linear` until
    /// the loader updates its parameters.
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
/// Lookup dequantizes the rows and applies ``SignedBlockHadamard/inverse(_:signs:)``;
/// ``asLinear(_:)`` rotates the input forward and runs the packed matmul, so a tied
/// head shares the table the way an unrotated `QuantizedEmbedding` does.
open class HadamardQuantizedEmbedding: QuantizedEmbedding {

    /// The rotation's ±1 signs, one per embedding dimension.
    public let signs: MLXArray

    /// The rotation folded into `weight`.
    public let rotation: SignedBlockHadamard

    /// A placeholder shaped like the checkpoint's arrays, standing in for `embedding`
    /// until the loader updates its parameters. The table is quantized lazily, as the
    /// loader's own quantize pass does for an unrotated embedding, and never evaluated.
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

    /// Whether the module is an embedding table (inverse rotation after lookup) rather
    /// than a linear layer (forward rotation before the matmul).
    public var embedding: Bool

    /// The activation dtype the module was packed for (`"float16"`), when the entry
    /// declares one.
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

/// A rotated checkpoint's declaration in `config.json`: which modules carry a folded
/// rotation and the quantization they share.
///
/// The keys mirror the pack format PrismML ships (`schema_version`, `base_model_type`,
/// `tensor_namespace`, `gdn_activation_layout`, `modules`, `quantization`). The base
/// model's own configuration sits beside them in the same file.
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

    /// Rejects a manifest this implementation cannot load faithfully.
    ///
    /// - Parameters:
    ///   - expectedType: the architecture the caller is about to instantiate.
    ///   - expectedNamespace: the tensor naming the caller's module paths assume.
    public func validate(
        baseModelType expectedType: String, tensorNamespace expectedNamespace: String
    )
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
        _ = try activationDType()
    }

    /// The activation dtype the packed modules declare, or `nil` when no entry declares
    /// one.
    ///
    /// Every declaring entry must agree, and only the half-precision types are accepted:
    /// the rotation is computed in float32 and cast back, and the packed matmul runs in
    /// the activation dtype.
    public func activationDType() throws -> DType? {
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

    /// Casts the checkpoint's unpacked floating-point tensors to `dtype`.
    ///
    /// A pack may store its norms, convolution taps and small projections in float32
    /// while its packed modules run in float16. MLX promotes a float16 activation through
    /// a float32 norm to float32, so left alone the residual stream would silently switch
    /// dtype after the first layer. This applies the cast mlx-vlm's converter applies when
    /// it writes a checkpoint: every floating tensor except the packed modules' own
    /// (`weight`, `scales`, `biases`, `signs`) and `A_log`, which the gated-delta
    /// recurrence reads in float32.
    ///
    /// - Parameters:
    ///   - weights: the checkpoint's tensors, keyed as the model names them.
    ///   - dtype: the activation dtype, normally ``activationDType()``.
    ///   - pathPrefix: what the model prepends to the manifest's paths.
    public func castingUnpackedWeights(
        _ weights: [String: MLXArray], to dtype: DType, pathPrefix: String = ""
    ) -> [String: MLXArray] {
        let packed = Set(modules.map { pathPrefix + $0.path })
        var cast = weights
        for (key, value) in weights
        where value.dtype.isFloatingPoint && value.dtype != dtype && !key.hasSuffix("A_log")
            && !packed.contains(Self.modulePath(of: key))
        {
            cast[key] = value.asType(dtype)
        }
        return cast
    }

    private static func modulePath(of key: String) -> String {
        guard let dot = key.lastIndex(of: ".") else { return key }
        return String(key[..<dot])
    }
}

/// Why a rotated checkpoint cannot be loaded.
public enum HadamardQuantizedCheckpointError: Error, CustomStringConvertible, Equatable {
    case unsupportedSchemaVersion(Int)
    case unsupportedBaseModelType(String?, expected: String)
    case unsupportedTensorNamespace(String?, expected: String)
    case unsupportedActivationLayout(String)
    case emptyManifest
    case mixedActivationDTypes([String])
    case unsupportedActivationDType(String)
    case moduleNotFound(String)
    case moduleNotSubstitutable(String, found: String)
    case unsupportedBlockSize(String, block: Int)

    public var description: String {
        switch self {
        case .unsupportedSchemaVersion(let version):
            "Rotated checkpoint schema_version \(version) is not supported "
                + "(expected \(HadamardQuantizedManifest.supportedSchemaVersion))"
        case .unsupportedBaseModelType(let type, let expected):
            "Rotated checkpoint base_model_type \(type ?? "nil") is not \(expected)"
        case .unsupportedTensorNamespace(let namespace, let expected):
            "Rotated checkpoint tensor_namespace \(namespace ?? "nil") is not \(expected)"
        case .unsupportedActivationLayout(let layout):
            "Rotated checkpoint gdn_activation_layout \(layout) needs a permutation "
                + "this loader does not apply"
        case .unsupportedBlockSize(let path, let block):
            "Rotated checkpoint module \(path) declares block \(block); "
                + "only a positive power of two is supported"
        case .emptyManifest:
            "Rotated checkpoint declares no modules"
        case .mixedActivationDTypes(let names):
            "Rotated checkpoint modules disagree on their activation dtype: "
                + names.joined(separator: ", ")
        case .unsupportedActivationDType(let name):
            "Rotated checkpoint activation dtype \(name) is not supported "
                + "(expected float16 or bfloat16)"
        case .moduleNotFound(let path):
            "Rotated checkpoint names a module the model does not have: \(path)"
        case .moduleNotSubstitutable(let path, let found):
            "Rotated checkpoint module \(path) is a \(found), not a plain Linear or Embedding"
        }
    }
}

// MARK: - Substitution

/// Replaces every manifest module of `model` with its rotated, pre-quantized layer.
///
/// Call this after constructing the model and before loading weights. The replacements
/// are placeholders shaped like the checkpoint's arrays; the loader's quantization pass
/// leaves them alone because they already conform to `Quantized`, and its parameter
/// update fills them in.
///
/// - Parameters:
///   - model: the model whose modules to replace.
///   - manifest: the checkpoint's manifest.
///   - pathPrefix: what to prepend to each manifest path, such as `"language_model."`
///     when the manifest names modules relative to a wrapped language model.
public func substituteHadamardQuantizedModules(
    in model: Module, manifest: HadamardQuantizedManifest, pathPrefix: String = ""
) throws {
    let modulesByPath = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
    let quantization = manifest.quantization
    var updates = [(String, Module)]()
    for entry in manifest.modules {
        let path = pathPrefix + entry.path
        guard let module = modulesByPath[path] else {
            throw HadamardQuantizedCheckpointError.moduleNotFound(path)
        }
        guard SignedBlockHadamard.isValidBlockSize(entry.block) else {
            throw HadamardQuantizedCheckpointError.unsupportedBlockSize(path, block: entry.block)
        }
        let rotation = SignedBlockHadamard(blockSize: entry.block)
        let replacement: Module
        switch module {
        case let embedding as Embedding where entry.embedding && type(of: module) == Embedding.self:
            replacement = HadamardQuantizedEmbedding(
                replacing: embedding, groupSize: quantization.groupSize, bits: quantization.bits,
                mode: quantization.mode, rotation: rotation)
        case let linear as Linear where !entry.embedding && type(of: module) == Linear.self:
            replacement = HadamardQuantizedLinear(
                replacing: linear, groupSize: quantization.groupSize, bits: quantization.bits,
                mode: quantization.mode, rotation: rotation)
        default:
            throw HadamardQuantizedCheckpointError.moduleNotSubstitutable(
                path, found: "\(type(of: module))")
        }
        updates.append((path, replacement))
    }

    model.update(modules: ModuleChildren.unflattened(updates))
}

// MARK: - Checkpoint

/// A rotated checkpoint's manifest, validated against the base model that loads it.
///
/// A model class built from such a checkpoint creates one of these before `super.init`
/// (the manifest is rejected before any weights are allocated), calls
/// ``substituteModules(in:)`` once the base model exists, and routes its `sanitize`
/// through ``sanitize(_:)``. The `PrismHadamardQwen35` classes in MLXLLM and MLXVLM are
/// the pattern.
public struct HadamardQuantizedCheckpoint: Sendable {

    /// The checkpoint's manifest.
    public let manifest: HadamardQuantizedManifest

    /// What the model prepends to the manifest's paths.
    public let pathPrefix: String

    /// The activation dtype the packed modules declare; `nil` leaves the unpacked
    /// tensors as stored.
    public let activationDType: DType?

    /// Validates `manifest` for a model of `baseModelType` whose module paths follow
    /// `tensorNamespace`.
    ///
    /// - Throws: `HadamardQuantizedCheckpointError` when the manifest is not one this
    ///   loader can honor.
    public init(
        manifest: HadamardQuantizedManifest, baseModelType: String, tensorNamespace: String,
        pathPrefix: String = ""
    ) throws {
        try manifest.validate(baseModelType: baseModelType, tensorNamespace: tensorNamespace)
        self.manifest = manifest
        self.pathPrefix = pathPrefix
        self.activationDType = try manifest.activationDType()
    }

    /// Replaces the manifest modules of `model` with rotated placeholders; see
    /// ``substituteHadamardQuantizedModules(in:manifest:pathPrefix:)``.
    public func substituteModules(in model: Module) throws {
        try substituteHadamardQuantizedModules(
            in: model, manifest: manifest, pathPrefix: pathPrefix)
    }

    /// Casts the checkpoint's unpacked floating-point tensors to the activation dtype;
    /// see `HadamardQuantizedManifest.castingUnpackedWeights(_:to:pathPrefix:)`.
    public func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        guard let activationDType else { return weights }
        return manifest.castingUnpackedWeights(weights, to: activationDType, pathPrefix: pathPrefix)
    }
}
