// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

extension DType {
    fileprivate var bytesPerScalar: Int {
        switch self {
        case .bfloat16, .float16: return 2
        case .float32: return 4
        case .int32, .uint32: return 4
        case .int16, .uint16: return 2
        case .int8, .uint8: return 1
        default: return 4
        }
    }
}

/// Lloyd-Max codebook centroids for Beta-distributed coordinates.
public enum TurboQuantCodebook {

    public struct Table {
        public let centroids: [Float]
        public let bounds: [Float]
    }

    /// Lloyd-Max-optimal codebooks for N(0,1) — the post-WHT, group-amax-
    /// normalized element distribution. Decision bounds are the centroid
    /// midpoints.
    public static func table(bits: Int) -> Table {
        switch bits {
        case 2:
            return Table(
                centroids: [-1.5104, -0.4528, 0.4528, 1.5104],
                bounds: [-0.9816, 0.0, 0.9816])
        case 3:
            return Table(
                centroids: [
                    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
                ],
                bounds: [-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748])
        case 4:
            return Table(
                centroids: [
                    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284,
                    0.1284, 0.3880, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
                ],
                bounds: [
                    -2.4008, -1.8435, -1.4371, -1.0993, -0.7996, -0.5224, -0.2582, 0.0,
                    0.2582, 0.5224, 0.7996, 1.0993, 1.4371, 1.8435, 2.4008,
                ])
        default:
            fatalError("TurboQuant supports 2/3/4-bit codebooks (got \(bits))")
        }
    }
}

public enum TurboQuantRotation {

    /// Generate a deterministic random orthogonal rotation matrix via QR decomposition.
    public static func rotationMatrix(dim: Int, seed: UInt64) -> MLXArray {
        let key = MLXRandom.key(seed)
        let gaussian = MLXRandom.normal([dim, dim], key: key)

        let (q, r) = MLXLinalg.qr(gaussian, stream: .cpu)
        let diagR = r.diagonal(stream: .cpu)
        let signs = sign(diagR, stream: .cpu)
        let result = q * expandedDimensions(signs, axis: 0)
        return result
    }

    /// Generate a Hadamard matrix of size dim x dim. Requires dim to be a power of 2.
    public static func hadamardMatrix(dim: Int) -> MLXArray {
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2")
        var h: [[Float]] = [[1.0]]
        var size = 1
        while size < dim {
            var newH = [[Float]](repeating: [Float](repeating: 0, count: size * 2), count: size * 2)
            for i in 0 ..< size {
                for j in 0 ..< size {
                    newH[i][j] = h[i][j]
                    newH[i][j + size] = h[i][j]
                    newH[i + size][j] = h[i][j]
                    newH[i + size][j + size] = -h[i][j]
                }
            }
            h = newH
            size *= 2
        }
        let flat = h.flatMap { $0 }
        let result = MLXArray(flat, [dim, dim])
        return result
    }

    /// Generate random ±1 sign vector for WHT rotation.
    public static func whtSigns(dim: Int, seed: UInt64) -> MLXArray {
        let key = MLXRandom.key(seed)
        let uniform = MLXRandom.uniform(low: 0, high: 1, [dim], key: key)
        let signs = MLX.where(uniform .> Float(0.5), Float(1.0), Float(-1.0))
        return signs
    }

    /// Apply WHT butterfly on the last dimension of x.
    private static func whtButterfly(_ x: MLXArray) -> MLXArray {
        let dim = x.dim(-1)
        let logDim = Int(log2(Double(dim)))
        let origShape = x.shape
        let N = origShape.dropLast().reduce(1, *)
        var y = x.reshaped([N, dim])

        for s in 0 ..< logDim {
            let halfBlock = 1 << s
            let blockSize = halfBlock << 1
            let numBlocks = dim / blockSize
            y = y.reshaped([N, numBlocks, blockSize])
            let a = y[0..., 0..., ..<halfBlock]
            let b = y[0..., 0..., halfBlock...]
            let sumAB = a + b
            let diffAB = a - b
            y = concatenated([sumAB, diffAB], axis: -1)
            y = y.reshaped([N, dim])
        }

        return y.reshaped(origShape)
    }

    /// Apply SRHT forward rotation on the last dimension.
    public static func fwhtForward(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        let dim = x.dim(-1)
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2")
        let signed = x * signs
        let transformed = whtButterfly(signed)
        let invSqrtDim = MLXArray(1.0 / sqrt(Float(dim)), dtype: x.dtype)
        return transformed * invSqrtDim
    }

    /// Apply SRHT inverse rotation on the last dimension.
    public static func fwhtInverse(_ y: MLXArray, signs: MLXArray) -> MLXArray {
        let dim = y.dim(-1)
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2")
        let transformed = whtButterfly(y)
        let invSqrtDim = MLXArray(1.0 / sqrt(Float(dim)), dtype: y.dtype)
        return transformed * invSqrtDim * signs
    }
}

/// Bit packing/unpacking for codebook indices.
public enum TurboQuantPacking {

    /// Number of uint32 words needed to pack `count` values at `bits` each.
    public static func packedWidth(count: Int, bits: Int) -> Int {
        (count * bits + 31) / 32
    }

    /// Pack b-bit indices into uint32 words.
    public static func packLowBit(_ indices: MLXArray, bits: Int) -> MLXArray {
        let count = indices.dim(-1)
        let batchShape = Array(indices.shape.dropLast())
        let rows = batchShape.reduce(1, *)
        let flat = indices.reshaped([rows, count])
        let pw = packedWidth(count: count, bits: bits)
        let mask = UInt32((1 << bits) - 1)

        var wordArrays = [MLXArray]()
        for w in 0 ..< pw {
            var word = MLXArray.zeros([rows], dtype: .uint32)
            for d in 0 ..< count {
                let bitOffset = d * bits
                let wordIdx = bitOffset / 32
                let offset = bitOffset % 32
                let spill = offset + bits - 32

                if wordIdx == w {
                    let shifted =
                        (flat[0..., d].asType(.uint32) & MLXArray(mask)) << MLXArray(UInt32(offset))
                    word = word | shifted
                }
                if spill > 0 && wordIdx + 1 == w {
                    let shifted =
                        (flat[0..., d].asType(.uint32) & MLXArray(mask))
                        >> MLXArray(UInt32(bits - spill))
                    word = word | shifted
                }
            }
            wordArrays.append(expandedDimensions(word, axis: -1))
        }
        let packed = concatenated(wordArrays, axis: -1)
        return packed.reshaped(batchShape + [pw])
    }

    /// Unpack b-bit indices from uint32 words.
    public static func unpackLowBit(_ packed: MLXArray, bits: Int, count: Int) -> MLXArray {
        let shape = packed.shape
        let batchShape = Array(shape.dropLast())
        let rows = batchShape.reduce(1, *)
        let flat = packed.reshaped([rows, -1])
        let mask = UInt32((1 << bits) - 1)

        var dimArrays = [MLXArray]()
        for d in 0 ..< count {
            let bitOffset = d * bits
            let wordIdx = bitOffset / 32
            let offset = bitOffset % 32
            let spill = offset + bits - 32

            var value = (flat[0..., wordIdx] >> MLXArray(UInt32(offset))) & MLXArray(mask)
            if spill > 0 {
                let high =
                    (flat[0..., wordIdx + 1] << MLXArray(UInt32(bits - spill))) & MLXArray(mask)
                value = value | high
            }
            dimArrays.append(expandedDimensions(value, axis: -1))
        }
        let unpacked = concatenated(dimArrays, axis: -1)
        return unpacked.reshaped(batchShape + [count])
    }
}

/// State for MSE-quantized vectors.
public struct MSECodecState {
    /// Per-group matched-norm scales, shape `[rows, dim / groupSize]`.
    public var scales: MLXArray
    public var packedIndices: MLXArray
    public var tokenCount: Int
    public let dim: Int
    public let bits: Int

    public init(scales: MLXArray, packedIndices: MLXArray, tokenCount: Int, dim: Int, bits: Int) {
        self.scales = scales
        self.packedIndices = packedIndices
        self.tokenCount = tokenCount
        self.dim = dim
        self.bits = bits
    }
}

/// Group-scaled Lloyd-Max codec: rotate, normalize each group of 16 by its
/// amax, quantize elements against an N(0,1)-optimal codebook, and store a
/// per-group matched-norm scale (`‖group‖ / ‖centroids‖`) so dequantization
/// preserves each group's L2 norm. Per-group scales are what make the codec
/// robust to the channel outliers of real K/V distributions — a single
/// per-vector norm lets one hot channel destroy the rest of the vector.
public class MSECodec {
    public let dim: Int
    public let bits: Int
    public let seed: UInt64

    /// Elements per scale group.
    public static let groupSize = 16

    public let codebook: MLXArray
    public let boundaries: MLXArray
    public let useWHT: Bool
    public let whtSigns: MLXArray?
    public let rotation: MLXArray
    public let rotationT: MLXArray

    /// Codebook span — amax maps onto ±this before quantization.
    private let codebookMax: Float

    public var numGroups: Int { dim / Self.groupSize }

    public init(dim: Int, bits: Int, seed: UInt64 = 42) {
        precondition(dim % Self.groupSize == 0, "dim must be a multiple of \(Self.groupSize)")
        self.dim = dim
        self.bits = bits
        self.seed = seed
        let table = TurboQuantCodebook.table(bits: bits)
        self.codebook = MLXArray(table.centroids)
        self.boundaries = MLXArray(table.bounds)
        self.codebookMax = table.centroids.last!

        let isPowerOf2 = dim > 0 && (dim & (dim - 1)) == 0
        self.useWHT = isPowerOf2 && dim <= 1024
        if useWHT {
            let signs = TurboQuantRotation.whtSigns(dim: dim, seed: seed)
            self.whtSigns = signs
            let hadamard = TurboQuantRotation.hadamardMatrix(dim: dim)
            let signsDiag = expandedDimensions(signs, axis: 0)
            let whtRot = hadamard * signsDiag / Float(sqrt(Float(dim)))
            // Keep rotation in f32 for precision: bf16 rounding in the
            // rotation matrix compounds across layers and measurably hurts
            // PPL on larger (9B+) models. MLX promotes bf16 inputs through
            // the f32 matmul efficiently.
            eval(whtRot)
            self.rotation = whtRot
            self.rotationT = self.rotation.transposed()
        } else {
            self.whtSigns = nil
            let rot = TurboQuantRotation.rotationMatrix(dim: dim, seed: seed)
            self.rotation = rot
            self.rotationT = self.rotation.transposed()
        }
    }

    /// Encode vectors `[rows, dim]` to packed codebook indices + per-group
    /// matched-norm scales.
    public func encode(_ vectors: MLXArray) -> MSECodecState {
        let rows = vectors.shape.dropLast().reduce(1, *)
        let v = vectors.reshaped([rows, dim]).asType(.float32)
        let rotated = matmul(v, rotationT)

        let g = numGroups
        let grouped = rotated.reshaped([rows, g, Self.groupSize])

        // amax-normalize each group into the codebook's span.
        let amax = abs(grouped).max(axis: -1, keepDims: true)
        let inv = codebookMax / maximum(amax, Float(1e-12))
        let indices = boundaryQuantize(grouped * inv)

        // Matched-norm scale: dequantized group keeps the input's L2 norm.
        // amax fallback on degenerate (all-zero reconstruction) groups.
        let recon = codebook[indices]
        let reconNorm = sqrt((recon * recon).sum(axis: -1))
        let groupNorm = sqrt((grouped * grouped).sum(axis: -1))
        let amaxScale = amax.squeezed(axis: -1) / codebookMax
        let scales = which(reconNorm .> Float(1e-10), groupNorm / reconNorm, amaxScale)

        let packed = TurboQuantPacking.packLowBit(
            indices.reshaped([rows, dim]), bits: bits)
        return MSECodecState(
            scales: scales, packedIndices: packed, tokenCount: rows, dim: dim, bits: bits)
    }

    /// Decode to dense vectors in the ORIGINAL (unrotated) space.
    public func decode(_ state: MSECodecState) -> MLXArray {
        matmul(decodeRotated(state), rotation)
    }

    /// Decode in rotated space without the inverse rotation.
    public func decodeRotated(_ state: MSECodecState) -> MLXArray {
        let rows = state.packedIndices.dim(0)
        let indices = TurboQuantPacking.unpackLowBit(
            state.packedIndices, bits: bits, count: dim)
        let grouped = codebook[indices].reshaped([rows, numGroups, Self.groupSize])
        let scaled =
            grouped * expandedDimensions(state.scales.reshaped([rows, numGroups]), axis: -1)
        return scaled.reshaped([rows, dim])
    }

    /// Pre-rotate queries for scoring against rotated keys.
    public func prepareQueries(_ queries: MLXArray) -> MLXArray {
        matmul(queries, rotationT)
    }

    /// Quantize rotated, group-normalized values to codebook indices.
    func boundaryQuantize(_ normalized: MLXArray) -> MLXArray {
        let ndim = normalized.ndim
        let expanded = expandedDimensions(normalized, axis: -1)
        var bShape = [Int](repeating: 1, count: ndim + 1)
        bShape[ndim] = boundaries.count
        let b = boundaries.reshaped(bShape)
        let greater = (expanded .> b).asType(.uint32)
        return greater.sum(axis: -1).asType(.uint32)
    }
}

/// KV cache with TurboQuant compression. Stores raw K/V during prefill,
/// compresses on first decode call, then encodes new tokens incrementally.
public class TurboQuantKVCache: BaseKVCache {

    public let bits: Int
    public let keyBits: Int
    public let valueBits: Int
    private let seed: UInt64

    /// When true, keys stay FP16 and only values are compressed (keyBits == 0).
    public let rawKeyMode: Bool

    private var keyMSECodec: MSECodec?
    private var valueMSECodec: MSECodec?

    private var rawKeys: MLXArray?
    private var rawValues: MLXArray?
    private var rawAllocSteps = 0

    private var keyPackedMSE: MLXArray?
    private var keyScales: MLXArray?
    private var valPackedMSE: MLXArray?
    private var valScales: MLXArray?
    private var compressedAllocSteps = 0

    public private(set) var isCompressed = false

    private var pendingRawKeys: [MLXArray] = []
    private var pendingRawValues: [MLXArray] = []
    private var uncompressedCount = 0
    private let recompressInterval: Int
    private let step: Int

    public init(
        bits: Int = 4, keyBits: Int? = nil, valueBits: Int? = nil, step: Int = 1024,
        recompressInterval: Int = 64, seed: UInt64 = 42
    ) {
        self.bits = bits
        self.keyBits = keyBits ?? bits
        self.valueBits = valueBits ?? bits
        self.rawKeyMode = (keyBits ?? bits) == 0
        self.seed = seed
        self.step = step
        self.recompressInterval = recompressInterval
        super.init()
    }

    override public var isTrimmable: Bool { true }

    private static let codecLock = NSLock()
    nonisolated(unsafe) private static var sharedCodecs: [String: MSECodec] = [:]

    private static func getOrCreateCodec(dim: Int, bits: Int, seed: UInt64) -> MSECodec {
        let key = "\(dim)_\(bits)_\(seed)"
        codecLock.lock()
        if let cached = sharedCodecs[key] {
            codecLock.unlock()
            return cached
        }
        codecLock.unlock()
        let codec = MSECodec(dim: dim, bits: bits, seed: seed)
        codecLock.lock()
        sharedCodecs[key] = codec
        codecLock.unlock()
        return codec
    }

    /// Initialize codecs if needed, using the shared cache.
    private func ensureCodecs(headDim: Int) {
        guard valueMSECodec == nil else { return }
        if !rawKeyMode {
            keyMSECodec = Self.getOrCreateCodec(dim: headDim, bits: keyBits, seed: seed)
        }
        valueMSECodec = Self.getOrCreateCodec(dim: headDim, bits: valueBits, seed: seed + 1)
    }

    /// Encode rows through the codec, returning packed indices and
    /// per-group scales. (The fused Metal encode kernels predate the
    /// group-scale format; they return in a follow-up re-templated for it.)
    private func fusedEncodeDispatch(
        input: MLXArray, codec: MSECodec, headDim: Int
    ) -> (MLXArray, MLXArray) {
        let state = codec.encode(input)
        return (state.packedIndices, state.scales)
    }

    /// Store raw K/V during prefill. Use ``updateAndDequant`` for decode.
    override public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.rawKeys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.rawKeys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.rawKeys, var currentValues = self.rawValues {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.rawKeys = concatenated([currentKeys, newK], axis: 2)
                self.rawValues = concatenated([currentValues, newV], axis: 2)
            } else {
                self.rawKeys = newK
                self.rawValues = newV
            }
            rawAllocSteps = self.rawKeys!.dim(2)
        }

        self.offset += keys.dim(2)

        self.rawKeys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.rawValues?[.ellipsis, previous ..< self.offset, 0...] = values

        let returnedKeys = self.rawKeys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.rawValues![.ellipsis, ..<self.offset, 0...]

        return (returnedKeys, returnedValues)
    }

    /// Compress the entire raw K/V cache into packed format in one batch.
    private func compressRawCache() {
        guard !isCompressed, let rk = rawKeys, let rv = rawValues, offset > 0 else { return }
        let allKeys = rk[.ellipsis, ..<offset, 0...]
        let allValues = rv[.ellipsis, ..<offset, 0...]
        let headDim = allKeys.dim(-1)
        ensureCodecs(headDim: headDim)
        compressRawCacheInternal(allKeys: allKeys, allValues: allValues, headDim: headDim)
        if rawKeyMode {
            rawValues = nil
        } else {
            rawKeys = nil
            rawValues = nil
            rawAllocSteps = 0
        }
        isCompressed = true
        MLX.Memory.clearCache()
    }

    /// Compress given raw K/V arrays into packed format.
    private func compressRawCacheInternal(allKeys: MLXArray, allValues: MLXArray, headDim: Int) {
        guard let valueMSECodec else { return }

        let B = allKeys.dim(0)
        let H = allKeys.dim(1)
        let tokenCount = allKeys.dim(2)
        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: valueBits)

        let flatVals = allValues.reshaped([B * H * tokenCount, headDim])
        let (valPackedFlat, valNormsFlat) = fusedEncodeDispatch(
            input: flatVals, codec: valueMSECodec, headDim: headDim)

        let allocSteps = ((tokenCount + step - 1) / step) * step
        let vG = valueMSECodec.numGroups
        valPackedMSE = MLXArray.zeros([B, H, allocSteps, vpw], dtype: .uint32)
        valScales = MLXArray.zeros([B, H, allocSteps, vG])

        if !rawKeyMode {
            guard let keyMSECodec else { return }
            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
            let flatKeys = allKeys.reshaped([B * H * tokenCount, headDim])
            let (keyPackedFlat, keyNormsFlat) = fusedEncodeDispatch(
                input: flatKeys, codec: keyMSECodec, headDim: headDim)

            let kG = keyMSECodec.numGroups
            keyPackedMSE = MLXArray.zeros([B, H, allocSteps, kpw], dtype: .uint32)
            keyScales = MLXArray.zeros([B, H, allocSteps, kG])
            keyPackedMSE![.ellipsis, ..<tokenCount, 0...] = keyPackedFlat.reshaped([
                B, H, tokenCount, kpw,
            ])
            keyScales![.ellipsis, ..<tokenCount, 0...] = keyNormsFlat.reshaped([
                B, H, tokenCount, kG,
            ])
        }

        valPackedMSE![.ellipsis, ..<tokenCount, 0...] = valPackedFlat.reshaped([
            B, H, tokenCount, vpw,
        ])
        valScales![.ellipsis, ..<tokenCount, 0...] = valNormsFlat.reshaped([
            B, H, tokenCount, vG,
        ])

        compressedAllocSteps = allocSteps

        if rawKeyMode {
            eval(valPackedMSE!, valScales!)
        } else {
            eval(keyPackedMSE!, keyScales!, valPackedMSE!, valScales!)
        }
    }

    /// Encode new token(s) into compressed storage.
    private func encodeNewToken(keys: MLXArray, values: MLXArray) {
        let headDim = keys.dim(-1)
        let B = keys.dim(0)
        let H = keys.dim(1)
        let numSteps = keys.dim(2)
        let prev = offset

        guard let valueMSECodec else { return }

        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: valueBits)

        let flatVals = values.reshaped([B * H * numSteps, headDim])
        let (valPacked, valNormsNew) = fusedEncodeDispatch(
            input: flatVals, codec: valueMSECodec, headDim: headDim)
        let vG = valueMSECodec.numGroups
        let valPackedShaped = valPacked.reshaped([B, H, numSteps, vpw])
        let valScalesShaped = valNormsNew.reshaped([B, H, numSteps, vG])

        if rawKeyMode {
            if (prev + numSteps) > rawAllocSteps {
                let newAlloc = ((prev + numSteps + step - 1) / step) * step
                let newRK = MLXArray.zeros([B, H, newAlloc, headDim], dtype: keys.dtype)
                if prev > 0, let rk = rawKeys {
                    newRK[.ellipsis, ..<prev, 0...] = rk[.ellipsis, ..<prev, 0...]
                }
                rawKeys = newRK
                rawAllocSteps = newAlloc
            }

            if (prev + numSteps) > compressedAllocSteps {
                let newAlloc = ((prev + numSteps + step - 1) / step) * step
                let newVP = MLXArray.zeros([B, H, newAlloc, vpw], dtype: .uint32)
                let newVN = MLXArray.zeros([B, H, newAlloc, vG])
                if prev > 0 {
                    newVP[.ellipsis, ..<prev, 0...] = valPackedMSE![.ellipsis, ..<prev, 0...]
                    newVN[.ellipsis, ..<prev, 0...] = valScales![.ellipsis, ..<prev, 0...]
                }
                valPackedMSE = newVP
                valScales = newVN
                compressedAllocSteps = newAlloc
            }

            offset = prev + numSteps
            rawKeys![.ellipsis, prev ..< offset, 0...] = keys
            valPackedMSE![.ellipsis, prev ..< offset, 0...] = valPackedShaped
            valScales![.ellipsis, prev ..< offset, 0...] = valScalesShaped
        } else {
            guard let keyMSECodec else { return }

            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
            let flatKeys = keys.reshaped([B * H * numSteps, headDim])
            let (keyPacked, keyNormsNew) = fusedEncodeDispatch(
                input: flatKeys, codec: keyMSECodec, headDim: headDim)
            let kG = keyMSECodec.numGroups
            let keyPackedShaped = keyPacked.reshaped([B, H, numSteps, kpw])
            let keyScalesShaped = keyNormsNew.reshaped([B, H, numSteps, kG])

            if (prev + numSteps) > compressedAllocSteps {
                let newAlloc = ((prev + numSteps + step - 1) / step) * step
                let newKP = MLXArray.zeros([B, H, newAlloc, kpw], dtype: .uint32)
                let newKN = MLXArray.zeros([B, H, newAlloc, kG])
                let newVP = MLXArray.zeros([B, H, newAlloc, vpw], dtype: .uint32)
                let newVN = MLXArray.zeros([B, H, newAlloc, vG])
                if prev > 0 {
                    newKP[.ellipsis, ..<prev, 0...] = keyPackedMSE![.ellipsis, ..<prev, 0...]
                    newKN[.ellipsis, ..<prev, 0...] = keyScales![.ellipsis, ..<prev, 0...]
                    newVP[.ellipsis, ..<prev, 0...] = valPackedMSE![.ellipsis, ..<prev, 0...]
                    newVN[.ellipsis, ..<prev, 0...] = valScales![.ellipsis, ..<prev, 0...]
                }
                keyPackedMSE = newKP
                keyScales = newKN
                valPackedMSE = newVP
                valScales = newVN
                compressedAllocSteps = newAlloc
            }

            offset = prev + numSteps
            keyPackedMSE![.ellipsis, prev ..< offset, 0...] = keyPackedShaped
            keyScales![.ellipsis, prev ..< offset, 0...] = keyScalesShaped
            valPackedMSE![.ellipsis, prev ..< offset, 0...] = valPackedShaped
            valScales![.ellipsis, prev ..< offset, 0...] = valScalesShaped
        }
    }

    /// Batch-encode all pending raw tokens into compressed storage.
    private func flushPendingEncode(headDim: Int) {
        guard !pendingRawKeys.isEmpty else { return }

        let batchKeys = concatenated(pendingRawKeys, axis: 2)
        let batchValues = concatenated(pendingRawValues, axis: 2)
        pendingRawKeys.removeAll(keepingCapacity: true)
        pendingRawValues.removeAll(keepingCapacity: true)

        let savedOffset = offset
        let batchStart = savedOffset - uncompressedCount
        offset = batchStart  // Temporarily rewind so encodeNewToken writes to the right slot
        encodeNewToken(keys: batchKeys, values: batchValues)
        offset = savedOffset  // Restore
        uncompressedCount = 0
    }

    private var dequantKeys: MLXArray?
    private var dequantValues: MLXArray?

    /// Encode new token and return (keys, values) in rotated space for SDPA.
    public func updateAndDequant(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let headDim = newKeys.dim(-1)
        ensureCodecs(headDim: headDim)

        guard let valueMSECodec else {
            return (newKeys, newValues)
        }

        if !isCompressed {
            isCompressed = true
            let tokenCount = offset
            if tokenCount > 0, let rk = rawKeys, let rv = rawValues {
                let rawK = rk[.ellipsis, ..<tokenCount, 0...]
                let rawV = rv[.ellipsis, ..<tokenCount, 0...]

                let rotV = valueMSECodec.prepareQueries(rawV)

                compressRawCacheInternal(allKeys: rawK, allValues: rawV, headDim: headDim)

                if rawKeyMode {
                    let nSteps = (step + tokenCount - 1) / step
                    let kShape = [rawK.dim(0), rawK.dim(1), nSteps * step, headDim]
                    let vShape = [rotV.dim(0), rotV.dim(1), nSteps * step, headDim]
                    dequantKeys = MLXArray.zeros(kShape, dtype: rawK.dtype)
                    dequantValues = MLXArray.zeros(vShape, dtype: rotV.dtype)
                    dequantKeys?[.ellipsis, ..<tokenCount, 0...] = rawK
                    dequantValues?[.ellipsis, ..<tokenCount, 0...] = rotV
                } else {
                    guard let keyMSECodec else { return (newKeys, newValues) }
                    let rotK = keyMSECodec.prepareQueries(rawK)

                    let nSteps = (step + tokenCount - 1) / step
                    let kShape = [rotK.dim(0), rotK.dim(1), nSteps * step, headDim]
                    let vShape = [rotV.dim(0), rotV.dim(1), nSteps * step, headDim]
                    dequantKeys = MLXArray.zeros(kShape, dtype: rotK.dtype)
                    dequantValues = MLXArray.zeros(vShape, dtype: rotV.dtype)
                    dequantKeys?[.ellipsis, ..<tokenCount, 0...] = rotK
                    dequantValues?[.ellipsis, ..<tokenCount, 0...] = rotV
                }
            }
            if !rawKeyMode {
                rawKeys = nil
            }
            rawValues = nil
        }

        let prevOffset = offset
        pendingRawKeys.append(newKeys)
        pendingRawValues.append(newValues)
        uncompressedCount += newKeys.dim(2)
        offset = prevOffset + newKeys.dim(2)

        // Scale compression interval with context length
        let adaptiveInterval = max(recompressInterval, offset / 256)
        if uncompressedCount >= adaptiveInterval {
            flushPendingEncode(headDim: newKeys.dim(-1))
        }

        let dequantNewKeys: MLXArray
        if rawKeyMode {
            dequantNewKeys = newKeys
        } else {
            guard let keyMSECodec else { return (newKeys, newValues) }
            dequantNewKeys = keyMSECodec.prepareQueries(newKeys)
        }
        let rotNewValues = valueMSECodec.prepareQueries(newValues)

        let reset =
            if let dk = self.dequantKeys, prevOffset + newKeys.dim(2) > dk.dim(2) {
                true
            } else {
                self.dequantKeys == nil
            }
        if reset {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let nSteps = (step + newKeys.dim(2) - 1) / step
            let kShape = [B, H, nSteps * step, headDim]
            let newDK = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let newDV = MLXArray.zeros(kShape, dtype: newKeys.dtype)

            if var currentKeys = self.dequantKeys, var currentValues = self.dequantValues {
                if prevOffset % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prevOffset, 0...]
                    currentValues = currentValues[.ellipsis, ..<prevOffset, 0...]
                }
                self.dequantKeys = concatenated([currentKeys, newDK], axis: 2)
                self.dequantValues = concatenated([currentValues, newDV], axis: 2)
            } else {
                self.dequantKeys = newDK
                self.dequantValues = newDV
            }
        }

        self.dequantKeys?[.ellipsis, prevOffset ..< offset, 0...] = dequantNewKeys
        self.dequantValues?[.ellipsis, prevOffset ..< offset, 0...] = rotNewValues

        return (
            self.dequantKeys![.ellipsis, ..<offset, 0...],
            self.dequantValues![.ellipsis, ..<offset, 0...]
        )
    }

    /// Pre-rotate queries to match rotated key space.
    public func prepareQueries(_ queries: MLXArray) -> MLXArray {
        if rawKeyMode { return queries }
        guard let keyMSECodec else { return queries }
        return keyMSECodec.prepareQueries(queries)
    }

    /// Inverse-rotate SDPA output back to original space.
    public func inverseRotateOutput(_ rotatedOutput: MLXArray) -> MLXArray {
        guard let valueMSECodec else { return rotatedOutput }
        return matmul(rotatedOutput, valueMSECodec.rotation)
    }

    /// Attention over the compressed cache for decode steps. Keys are
    /// dequantized in rotated space (raw in `rawKeyMode`), scores computed
    /// against rotated queries, and values reconstructed per group — all in
    /// MLX ops. The rotation cancels in the score dot product, so results
    /// match attention over the dequantized cache exactly.
    public func compressedAttention(
        queries: MLXArray,
        keys newKeys: MLXArray,
        values newValues: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let headDim = newKeys.dim(-1)
        let B = queries.dim(0)
        let nQHeads = queries.dim(1)
        let nKVHeads = newKeys.dim(1)
        let L = queries.dim(2)
        let nRepeats = nQHeads / nKVHeads

        if !isCompressed {
            compressRawCache()
        }

        pendingRawKeys.append(newKeys)
        pendingRawValues.append(newValues)
        uncompressedCount += newKeys.dim(2)
        offset += newKeys.dim(2)
        flushPendingEncode(headDim: newKeys.dim(-1))

        guard let valueMSECodec else {
            return queries
        }

        let tokenCount = offset
        let vG = valueMSECodec.numGroups

        // Keys for scoring: raw (rawKeyMode) or dequantized in rotated space.
        let scoreKeys: MLXArray
        let scoreQueries: MLXArray
        if rawKeyMode {
            guard let rk = rawKeys else { return queries }
            scoreKeys = rk[.ellipsis, ..<tokenCount, 0...].asType(.float32)
            scoreQueries = queries.asType(.float32)
        } else {
            guard let keyMSECodec else { return queries }
            let kG = keyMSECodec.numGroups
            let flatKeyPacked = keyPackedMSE![0..., 0..., ..<tokenCount, 0...]
                .reshaped([B * nKVHeads * tokenCount, -1]).contiguous()
            let flatKeyScales = keyScales![0..., 0..., ..<tokenCount, 0...]
                .reshaped([B * nKVHeads * tokenCount, kG]).contiguous()
            scoreKeys = keyMSECodec.decodeRotated(
                MSECodecState(
                    scales: flatKeyScales, packedIndices: flatKeyPacked,
                    tokenCount: tokenCount, dim: headDim, bits: keyBits)
            ).reshaped([B, nKVHeads, tokenCount, headDim])
            scoreQueries = keyMSECodec.prepareQueries(queries.asType(.float32))
        }

        // GQA: expand kv heads across their query-head group.
        let expandedKeys: MLXArray
        if nRepeats > 1 {
            expandedKeys = MLX.repeated(
                expandedDimensions(scoreKeys, axis: 2), count: nRepeats, axis: 2
            ).reshaped([B, nQHeads, tokenCount, headDim])
        } else {
            expandedKeys = scoreKeys
        }

        var scores = matmul(scoreQueries, expandedKeys.transposed(0, 1, 3, 2)) * scale
        switch mask {
        case .array(let maskArray):
            if maskArray.dtype == .bool {
                scores = MLX.where(
                    maskArray, scores, MLXArray(-Float.infinity, dtype: scores.dtype))
            } else {
                scores = scores + maskArray
            }
        case .causal:
            let queryOffset = tokenCount - L
            let causalMask = MLXArray.tri(L, m: tokenCount, k: queryOffset, type: Bool.self)
            let expandedMask = expandedDimensions(
                expandedDimensions(causalMask, axis: 0), axis: 0)
            scores = MLX.where(
                expandedMask, scores, MLXArray(-Float.infinity, dtype: scores.dtype))
        case .none: break
        default: break
        }

        let attnWeights = softmax(scores.asType(.float32), axis: -1)
        // Materialize before the value pass — unbounded lazy graphs across
        // 28+ layers of unpack/gather chains can wedge evaluation.
        eval(attnWeights)

        // Values: dequantize in rotated space, weighted-sum, un-rotate.
        let flatValPacked = valPackedMSE![0..., 0..., ..<tokenCount, 0...]
            .reshaped([B * nKVHeads * tokenCount, -1]).contiguous()
        let flatValScales = valScales![0..., 0..., ..<tokenCount, 0...]
            .reshaped([B * nKVHeads * tokenCount, vG]).contiguous()
        let valRot = valueMSECodec.decodeRotated(
            MSECodecState(
                scales: flatValScales, packedIndices: flatValPacked,
                tokenCount: tokenCount, dim: headDim, bits: valueBits)
        ).reshaped([B, nKVHeads, tokenCount, headDim])

        let expandedVals: MLXArray
        if nRepeats > 1 {
            expandedVals = MLX.repeated(
                expandedDimensions(valRot, axis: 2), count: nRepeats, axis: 2
            ).reshaped([B, nQHeads, tokenCount, headDim])
        } else {
            expandedVals = valRot
        }

        let rotatedOutput = matmul(attnWeights, expandedVals)
        let output = matmul(rotatedOutput, valueMSECodec.rotation)

        // Return in the query dtype so compressed attention doesn't promote
        // the activation stream.
        return output.asType(queries.dtype)
    }

    /// Approximate memory footprint in bytes (excludes shared codec overhead).
    public var memoryBytes: Int {
        var total = 0
        if let rk = rawKeys { total += rk.shape.reduce(1, *) * rk.dtype.bytesPerScalar }
        if let rv = rawValues { total += rv.shape.reduce(1, *) * rv.dtype.bytesPerScalar }
        if let kp = keyPackedMSE { total += kp.shape.reduce(1, *) * kp.dtype.bytesPerScalar }
        if let kn = keyScales { total += kn.shape.reduce(1, *) * kn.dtype.bytesPerScalar }
        if let vp = valPackedMSE { total += vp.shape.reduce(1, *) * vp.dtype.bytesPerScalar }
        if let vn = valScales { total += vn.shape.reduce(1, *) * vn.dtype.bytesPerScalar }
        if let dk = dequantKeys { total += dk.shape.reduce(1, *) * dk.dtype.bytesPerScalar }
        if let dv = dequantValues { total += dv.shape.reduce(1, *) * dv.dtype.bytesPerScalar }
        return total
    }

    override public var state: [MLXArray] {
        get {
            if isCompressed {
                if rawKeyMode {
                    guard let rk = rawKeys,
                        let vpm = valPackedMSE, let vn = valScales,
                        offset > 0
                    else { return [] }
                    return [
                        rk[0..., 0..., ..<offset, 0...],
                        vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset, 0...],
                    ]
                } else {
                    guard let kpm = keyPackedMSE, let kn = keyScales,
                        let vpm = valPackedMSE, let vn = valScales,
                        offset > 0
                    else { return [] }
                    return [
                        kpm[0..., 0..., ..<offset, 0...], kn[0..., 0..., ..<offset, 0...],
                        vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset, 0...],
                    ]
                }
            } else {
                guard let rk = rawKeys, let rv = rawValues, offset > 0 else { return [] }
                return [rk[0..., 0..., ..<offset, 0...], rv[0..., 0..., ..<offset, 0...]]
            }
        }
        set {
            if rawKeyMode && newValue.count == 3 {
                rawKeys = newValue[0]
                rawAllocSteps = newValue[0].dim(2)
                valPackedMSE = newValue[1]
                valScales = newValue[2]
                offset = newValue[0].dim(2)
                compressedAllocSteps = newValue[1].dim(2)
                isCompressed = true
            } else if newValue.count == 4 {
                keyPackedMSE = newValue[0]
                keyScales = newValue[1]
                valPackedMSE = newValue[2]
                valScales = newValue[3]
                offset = newValue[0].dim(2)
                compressedAllocSteps = offset
                isCompressed = true
            } else if newValue.count == 2 {
                rawKeys = newValue[0]
                rawValues = newValue[1]
                offset = newValue[0].dim(2)
                rawAllocSteps = offset
                isCompressed = false
            }
        }
    }

    override public var metaState: [String] {
        get {
            ["\(offset)", "\(bits)", "\(keyBits)", "\(valueBits)", "\(seed)"]
        }
        set {
            guard newValue.count >= 5,
                let o = Int(newValue[0])
            else { return }
            offset = o
        }
    }

    @discardableResult
    override public func trim(_ n: Int) -> Int {
        guard n > 0, offset > 0 else { return 0 }

        if !pendingRawKeys.isEmpty {
            flushPendingEncode(headDim: pendingRawKeys[0].dim(-1))
        }

        let trimCount = min(n, offset)
        offset -= trimCount
        if offset == 0 {
            rawKeys = nil
            rawValues = nil
            rawAllocSteps = 0
            keyPackedMSE = nil
            keyScales = nil
            valPackedMSE = nil
            valScales = nil
            dequantKeys = nil
            dequantValues = nil
            compressedAllocSteps = 0
            isCompressed = false
            pendingRawKeys.removeAll()
            pendingRawValues.removeAll()
            uncompressedCount = 0
        }
        return trimCount
    }
}

// MARK: - kvScheme routing

/// Resolve a TurboQuant kvScheme string to (keyBits, valueBits).
/// `keyBits == 0` means raw FP16 keys (only values are compressed).
/// Returns nil for non-TurboQuant schemes.
public func resolveTurboScheme(_ scheme: String?) -> (keyBits: Int, valueBits: Int)? {
    switch scheme {
    // Raw-key schemes: keys stay FP16, values are group-quantized. The
    // teacher-forced PPL/KLD gate shows these are near-lossless (see the
    // PR validation table); they are the recommended configurations.
    case "turbo0v4": return (0, 4)
    case "turbo0v2": return (0, 2)
    // NOTE: key-quantizing schemes ("turbo4", "turbo3", "turbo4v2", ...)
    // are not exposed yet — 4-bit grouped key quantization collapses
    // real-text quality (PPL 89 vs 1.18 baseline on Qwen2.5-1.5B) even
    // though every component passes unit parity. Keys need a finer or
    // smarter treatment before those schemes return.
    default: return nil
    }
}

/// Convert eligible `KVCacheSimple` layers to `TurboQuantKVCache` once their
/// offset passes `quantizedKVStart`, transferring the accumulated KV state.
/// Mamba / wrapper caches are left untouched. Called from
/// `maybeQuantizeKVCache` when `kvScheme` names a TurboQuant scheme.
public func maybeTurboQuantizeKVCache(
    cache: inout [KVCache],
    keyBits: Int,
    valueBits: Int,
    quantizedKVStart: Int
) {
    guard
        cache.contains(where: { $0 is KVCacheSimple && $0.offset > quantizedKVStart })
    else { return }

    for i in 0 ..< cache.count {
        guard cache[i] is KVCacheSimple, cache[i].offset > quantizedKVStart,
            !(cache[i] is TurboQuantKVCache)
        else { continue }
        let turbo = TurboQuantKVCache(
            bits: max(keyBits, valueBits), keyBits: keyBits, valueBits: valueBits)
        // Transfer existing KV data, trimmed to the live offset (the simple
        // cache over-allocates in steps).
        let offset = cache[i].offset
        let state = cache[i].innerState()
        if state.count >= 2, offset > 0 {
            let keys = state[0][.ellipsis, ..<offset, 0...]
            let values = state[1][.ellipsis, ..<offset, 0...]
            _ = turbo.update(keys: keys, values: values)
        }
        cache[i] = turbo
    }
}
