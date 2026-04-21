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

    /// Pre-computed centroids for common (dim, bits) pairs.
    nonisolated(unsafe) private static var precomputed: [Int: [Int: [Float]]] = [
        64: [
            2: [-0.18745463, -0.05649366, 0.05649367, 0.18745449],
            3: [
                -0.26375133, -0.16599470, -0.09368263, -0.03040462, 0.03040464, 0.09368261,
                0.16599482, 0.26375186,
            ],
            4: [
                -0.32913971, -0.25096416, -0.19681059, -0.15295772, -0.11478586, -0.08000945,
                -0.04726735, -0.01563822, 0.01563822, 0.04723797, 0.07994876, 0.11472529,
                0.15289739, 0.19675052, 0.25090477, 0.32908401,
            ],
        ],
        128: [
            2: [-0.13302007, -0.03998107, 0.03998102, 0.13302033],
            3: [
                -0.18828832, -0.11801215, -0.06648001, -0.02156330, 0.02156329, 0.06648005,
                0.11801218, 0.18828897,
            ],
            4: [
                -0.23639172, -0.17934021, -0.14023653, -0.10881814, -0.08157559, -0.05678632,
                -0.03350975, -0.01108178, 0.01108178, 0.03350975, 0.05678631, 0.08157560,
                0.10881804, 0.14023650, 0.17934017, 0.23639278,
            ],
        ],
        256: [
            2: [-0.09420358, -0.02827190, 0.02827190, 0.09420330],
            3: [
                -0.13371243, -0.08361249, -0.04704370, -0.01524900, 0.01524901, 0.04704368,
                0.08361248, 0.13371260,
            ],
            4: [
                -0.16852295, -0.12754069, -0.09961203, -0.07719406, -0.05781249, -0.04021866,
                -0.02370371, -0.00783269, 0.00783269, 0.02370371, 0.04021868, 0.05781246,
                0.07719407, 0.09961203, 0.12754090, 0.16852276,
            ],
        ],
    ]

    private static let centroidLock = NSLock()

    /// Non-power-of-2 dims lazily populated on first access (e.g. 80 for Qwen3-4B).
    private static let lazyDims: [Int] = [80, 96]
    private static let lazyBits: [Int] = [2, 3, 4, 8]

    /// Ensure centroids for a given dim are populated.
    private static func ensureCentroidsPopulated(dim: Int) {
        centroidLock.lock()
        let exists = precomputed[dim] != nil
        centroidLock.unlock()
        guard !exists else { return }

        // Generate all bit-widths for this dim
        var dimTable: [Int: [Float]] = [:]
        for bits in lazyBits {
            dimTable[bits] = generateCentroids(dim: dim, bits: bits)
        }

        centroidLock.lock()
        // Double-check after lock (another thread may have populated)
        if precomputed[dim] == nil {
            precomputed[dim] = dimTable
        }
        centroidLock.unlock()
    }

    /// Codebook centroids for the given (dim, bits) pair.
    public static func codebook(dim: Int, bits: Int) -> MLXArray {
        if let dimTable = precomputed[dim], let centroids = dimTable[bits] {
            return MLXArray(centroids)
        }
        // Lazy populate for known model dims
        if lazyDims.contains(dim) {
            ensureCentroidsPopulated(dim: dim)
            if let dimTable = precomputed[dim], let centroids = dimTable[bits] {
                return MLXArray(centroids)
            }
        }
        let centroids = generateCentroids(dim: dim, bits: bits)
        return MLXArray(centroids)
    }

    /// Midpoint boundaries between adjacent centroids.
    public static func boundaries(dim: Int, bits: Int) -> MLXArray {
        let centroids: [Float]
        if let dimTable = precomputed[dim], let cached = dimTable[bits] {
            centroids = cached
        } else if lazyDims.contains(dim) {
            ensureCentroidsPopulated(dim: dim)
            if let dimTable = precomputed[dim], let cached = dimTable[bits] {
                centroids = cached
            } else {
                centroids = generateCentroids(dim: dim, bits: bits)
            }
        } else {
            centroids = generateCentroids(dim: dim, bits: bits)
        }
        var bounds = [Float]()
        for i in 0 ..< centroids.count - 1 {
            bounds.append((centroids[i] + centroids[i + 1]) / 2.0)
        }
        return MLXArray(bounds)
    }

    /// Generate codebook centroids via weighted k-means on Beta distribution.
    static func generateCentroids(dim: Int, bits: Int) -> [Float] {
        let levels = 1 << bits
        let gridSize = 32768
        let sigma = 1.0 / sqrt(Float(dim))

        var grid = [Float](repeating: 0, count: gridSize)
        var weights = [Float](repeating: 0, count: gridSize)
        for i in 0 ..< gridSize {
            let x = -1.0 + 2.0 * Float(i) / Float(gridSize - 1)
            grid[i] = x
            let exponent = Float(dim - 3) / 2.0
            let w = pow(max(1.0 - x * x, 1e-30), exponent)
            weights[i] = w
        }

        let totalW = weights.reduce(0, +)
        var centroids = [Float](repeating: 0, count: levels)
        var cumW: Float = 0
        var ci = 0
        for i in 0 ..< gridSize {
            cumW += weights[i]
            let target = (Float(ci) + 0.5) / Float(levels) * totalW
            if cumW >= target && ci < levels {
                centroids[ci] = grid[i]
                ci += 1
            }
        }
        // Fill remaining
        while ci < levels {
            centroids[ci] = centroids[ci - 1] + sigma
            ci += 1
        }

        for _ in 0 ..< 100 {
            var sums = [Float](repeating: 0, count: levels)
            var counts = [Float](repeating: 0, count: levels)
            for i in 0 ..< gridSize {
                var bestJ = 0
                var bestDist = Float.infinity
                for j in 0 ..< levels {
                    let d = abs(grid[i] - centroids[j])
                    if d < bestDist {
                        bestDist = d
                        bestJ = j
                    }
                }
                sums[bestJ] += grid[i] * weights[i]
                counts[bestJ] += weights[i]
            }
            for j in 0 ..< levels {
                if counts[j] > 0 { centroids[j] = sums[j] / counts[j] }
            }
        }

        return centroids.sorted()
    }
}

/// Random orthogonal rotation matrix generation.
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
    public var norms: MLXArray
    public var packedIndices: MLXArray
    public var tokenCount: Int
    public let dim: Int
    public let bits: Int

    public init(norms: MLXArray, packedIndices: MLXArray, tokenCount: Int, dim: Int, bits: Int) {
        self.norms = norms
        self.packedIndices = packedIndices
        self.tokenCount = tokenCount
        self.dim = dim
        self.bits = bits
    }
}

/// MSE-optimal codec: rotate, quantize to codebook indices, pack bits.
public class MSECodec {
    public let dim: Int
    public let bits: Int
    public let seed: UInt64

    public let codebook: MLXArray
    public let boundaries: MLXArray
    public let useWHT: Bool
    public let whtSigns: MLXArray?
    public let rotation: MLXArray
    public let rotationT: MLXArray

    public init(dim: Int, bits: Int, seed: UInt64 = 42) {
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.codebook = TurboQuantCodebook.codebook(dim: dim, bits: bits)
        self.boundaries = TurboQuantCodebook.boundaries(dim: dim, bits: bits)

        let isPowerOf2 = dim > 0 && (dim & (dim - 1)) == 0
        self.useWHT = isPowerOf2 && dim <= 1024
        if useWHT {
            let signs = TurboQuantRotation.whtSigns(dim: dim, seed: seed)
            self.whtSigns = signs
            let hadamard = TurboQuantRotation.hadamardMatrix(dim: dim)
            let signsDiag = expandedDimensions(signs, axis: 0)
            let whtRot = hadamard * signsDiag / Float(sqrt(Float(dim)))
            // bf16 to match model dtype and avoid promoting inputs through matmul
            self.rotation = whtRot.asType(.bfloat16)
            self.rotationT = self.rotation.transposed()
        } else {
            self.whtSigns = nil
            let rot = TurboQuantRotation.rotationMatrix(dim: dim, seed: seed)
            self.rotation = rot.asType(.bfloat16)
            self.rotationT = self.rotation.transposed()
        }
    }

    /// Encode vectors to packed codebook indices with norm extraction.
    public func encode(_ vectors: MLXArray) -> MSECodecState {
        let norms = sqrt((vectors * vectors).sum(axis: -1))
        let safeNorms = maximum(norms, Float(1e-8))
        let unit = vectors / expandedDimensions(safeNorms, axis: -1)

        let rotated = matmul(unit, rotationT)

        let indices = boundaryQuantize(rotated)

        let storedNorms: MLXArray
        if useWHT {
            // WHT is orthogonal — norms preserved, no correction needed
            storedNorms = norms
        } else {
            // Norm correction compensates for quantization error in dense rotation
            let reconstructed = codebook[indices]
            let reconNormSq = (reconstructed * reconstructed).sum(axis: -1)
            let reconNorms = sqrt(maximum(reconNormSq, Float(1e-16)))
            storedNorms = norms / reconNorms
        }

        let packed = TurboQuantPacking.packLowBit(indices, bits: bits)

        return MSECodecState(
            norms: storedNorms,
            packedIndices: packed,
            tokenCount: vectors.dim(2),
            dim: dim,
            bits: bits
        )
    }

    /// Decode from compressed state back to dense vectors.
    public func decode(_ state: MSECodecState) -> MLXArray {
        let indices = TurboQuantPacking.unpackLowBit(state.packedIndices, bits: bits, count: dim)

        let approx = codebook[indices]

        let unrotated = matmul(approx, rotation)

        return expandedDimensions(state.norms, axis: -1) * unrotated
    }

    /// Decode in rotated space without inverse rotation.
    public func decodeRotated(_ state: MSECodecState) -> MLXArray {
        let indices = TurboQuantPacking.unpackLowBit(state.packedIndices, bits: bits, count: dim)
        let approx = codebook[indices]
        return expandedDimensions(state.norms, axis: -1) * approx
    }

    /// Pre-rotate queries for compressed-domain scoring.
    public func prepareQueries(_ queries: MLXArray) -> MLXArray {
        return matmul(queries, rotationT)
    }

    /// Quantize via boundary comparison. Returns uint32 codebook indices.
    func boundaryQuantize(_ rotated: MLXArray) -> MLXArray {
        let ndim = rotated.ndim
        let expanded = expandedDimensions(rotated, axis: -1)
        var bShape = [Int](repeating: 1, count: ndim + 1)
        bShape[ndim] = boundaries.count
        let b = boundaries.reshaped(bShape)
        let greater = (expanded .> b).asType(.uint32)
        let indices = greater.sum(axis: -1)
        return indices.asType(.uint32)
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
    private var keyNorms: MLXArray?
    private var valPackedMSE: MLXArray?
    private var valNorms: MLXArray?
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

    /// Dispatch to the appropriate fused encode kernel (WHT or dense).
    private func fusedEncodeDispatch(
        input: MLXArray, codec: MSECodec, headDim: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        if codec.useWHT, let signs = codec.whtSigns {
            return TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: codec.bits, dim: headDim
            )
        } else {
            return TurboQuantKernelOps.fusedEncode(
                input: input, rotation: codec.rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: codec.bits, dim: headDim
            )
        }
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
        valPackedMSE = MLXArray.zeros([B, H, allocSteps, vpw], dtype: .uint32)
        valNorms = MLXArray.zeros([B, H, allocSteps])

        if !rawKeyMode {
            guard let keyMSECodec else { return }
            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
            let flatKeys = allKeys.reshaped([B * H * tokenCount, headDim])
            let (keyPackedFlat, keyNormsFlat) = fusedEncodeDispatch(
                input: flatKeys, codec: keyMSECodec, headDim: headDim)

            keyPackedMSE = MLXArray.zeros([B, H, allocSteps, kpw], dtype: .uint32)
            keyNorms = MLXArray.zeros([B, H, allocSteps])
            keyPackedMSE![.ellipsis, ..<tokenCount, 0...] = keyPackedFlat.reshaped([
                B, H, tokenCount, kpw,
            ])
            keyNorms![.ellipsis, ..<tokenCount] = keyNormsFlat.reshaped([B, H, tokenCount])
        }

        valPackedMSE![.ellipsis, ..<tokenCount, 0...] = valPackedFlat.reshaped([
            B, H, tokenCount, vpw,
        ])
        valNorms![.ellipsis, ..<tokenCount] = valNormsFlat.reshaped([B, H, tokenCount])

        compressedAllocSteps = allocSteps

        if rawKeyMode {
            eval(valPackedMSE!, valNorms!)
        } else {
            eval(keyPackedMSE!, keyNorms!, valPackedMSE!, valNorms!)
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
        let valPackedShaped = valPacked.reshaped([B, H, numSteps, vpw])
        let valNormsShaped = valNormsNew.reshaped([B, H, numSteps])

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
                let newVN = MLXArray.zeros([B, H, newAlloc])
                if prev > 0 {
                    newVP[.ellipsis, ..<prev, 0...] = valPackedMSE![.ellipsis, ..<prev, 0...]
                    newVN[.ellipsis, ..<prev] = valNorms![.ellipsis, ..<prev]
                }
                valPackedMSE = newVP
                valNorms = newVN
                compressedAllocSteps = newAlloc
            }

            offset = prev + numSteps
            rawKeys![.ellipsis, prev ..< offset, 0...] = keys
            valPackedMSE![.ellipsis, prev ..< offset, 0...] = valPackedShaped
            valNorms![.ellipsis, prev ..< offset] = valNormsShaped
        } else {
            guard let keyMSECodec else { return }

            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
            let flatKeys = keys.reshaped([B * H * numSteps, headDim])
            let (keyPacked, keyNormsNew) = fusedEncodeDispatch(
                input: flatKeys, codec: keyMSECodec, headDim: headDim)
            let keyPackedShaped = keyPacked.reshaped([B, H, numSteps, kpw])
            let keyNormsShaped = keyNormsNew.reshaped([B, H, numSteps])

            if (prev + numSteps) > compressedAllocSteps {
                let newAlloc = ((prev + numSteps + step - 1) / step) * step
                let newKP = MLXArray.zeros([B, H, newAlloc, kpw], dtype: .uint32)
                let newKN = MLXArray.zeros([B, H, newAlloc])
                let newVP = MLXArray.zeros([B, H, newAlloc, vpw], dtype: .uint32)
                let newVN = MLXArray.zeros([B, H, newAlloc])
                if prev > 0 {
                    newKP[.ellipsis, ..<prev, 0...] = keyPackedMSE![.ellipsis, ..<prev, 0...]
                    newKN[.ellipsis, ..<prev] = keyNorms![.ellipsis, ..<prev]
                    newVP[.ellipsis, ..<prev, 0...] = valPackedMSE![.ellipsis, ..<prev, 0...]
                    newVN[.ellipsis, ..<prev] = valNorms![.ellipsis, ..<prev]
                }
                keyPackedMSE = newKP
                keyNorms = newKN
                valPackedMSE = newVP
                valNorms = newVN
                compressedAllocSteps = newAlloc
            }

            offset = prev + numSteps
            keyPackedMSE![.ellipsis, prev ..< offset, 0...] = keyPackedShaped
            keyNorms![.ellipsis, prev ..< offset] = keyNormsShaped
            valPackedMSE![.ellipsis, prev ..< offset, 0...] = valPackedShaped
            valNorms![.ellipsis, prev ..< offset] = valNormsShaped
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

    /// Use compressed-domain Metal kernels instead of dequant + SDPA.
    public var useCompressedAttention: Bool = false

    /// Compressed-domain attention via Metal kernels.
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

        let flatValPacked = valPackedMSE![0..., 0..., ..<tokenCount, 0...]
            .reshaped([B * nKVHeads, tokenCount, -1]).contiguous()
        let flatValNorms = valNorms![0..., 0..., ..<tokenCount]
            .reshaped([B * nKVHeads, tokenCount]).contiguous()

        let valRotation = valueMSECodec.rotation
        let output: MLXArray

        if rawKeyMode {
            guard let rk = rawKeys else { return queries }
            let allKeys = rk[.ellipsis, ..<tokenCount, 0...]
            let expandedKeys: MLXArray
            if nRepeats > 1 {
                let expanded = expandedDimensions(allKeys, axis: 2)
                let tiledKeys = MLX.tiled(expanded, repetitions: [1, 1, nRepeats, 1, 1])
                expandedKeys = tiledKeys.reshaped([B, nQHeads, tokenCount, headDim])
            } else {
                expandedKeys = allKeys
            }

            var scores = matmul(queries, expandedKeys.transposed(0, 1, 3, 2)) * scale
            switch mask {
            case .array(let maskArray):
                if maskArray.dtype == .bool {
                    scores = MLX.where(
                        maskArray, scores, MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype)
                    )
                } else {
                    scores = scores + maskArray
                }
            case .causal:
                let queryOffset = tokenCount - L
                let causalMask = MLXArray.tri(L, m: tokenCount, k: queryOffset, type: Bool.self)
                let expandedMask = expandedDimensions(
                    expandedDimensions(causalMask, axis: 0), axis: 0)
                scores = MLX.where(
                    expandedMask, scores, MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype))
            case .none: break
            default: break
            }

            let attnWeights = softmax(scores, axis: -1)
            // Materialize before Metal kernel — prevents lazy graph overflow in rawKeyMode
            eval(attnWeights)

            let flatWeights = attnWeights.reshaped([B * nQHeads * L, tokenCount])
            let rotatedOutput = TurboQuantKernelOps.mseWeightedSum(
                weights: flatWeights, packed: flatValPacked, norms: flatValNorms,
                codebook: valueMSECodec.codebook, tokenCount: tokenCount,
                repeatCount: nRepeats, bits: self.valueBits, dim: headDim
            )

            output = matmul(
                rotatedOutput.reshaped([B, nQHeads, L, headDim]),
                valueMSECodec.rotation
            )
        } else {
            guard let keyMSECodec else { return queries }

            let qRot = keyMSECodec.prepareQueries(queries) * scale
            let flatQ = qRot.reshaped([B * nQHeads * L, headDim])

            let flatKeyPacked = keyPackedMSE![0..., 0..., ..<tokenCount, 0...]
                .reshaped([B * nKVHeads, tokenCount, -1]).contiguous()
            let flatKeyNorms = keyNorms![0..., 0..., ..<tokenCount]
                .reshaped([B * nKVHeads, tokenCount]).contiguous()

            if case .causal = mask {
                let queryOffset = tokenCount - L
                output = TurboQuantKernelOps.turboFlashAttentionCausal(
                    rotatedQueries: flatQ,
                    keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                    keyCodebook: keyMSECodec.codebook,
                    valPacked: flatValPacked, valNorms: flatValNorms,
                    valCodebook: valueMSECodec.codebook,
                    tokenCount: tokenCount, repeatCount: nRepeats,
                    keyBits: self.keyBits, valueBits: self.valueBits, dim: headDim,
                    queryChunkLength: L, queryOffset: queryOffset,
                    valRotation: valRotation
                ).reshaped([B, nQHeads, L, headDim])
            } else {
                let dequantKRot = keyMSECodec.decodeRotated(MSECodecState(
                    norms: flatKeyNorms.reshaped([B * nKVHeads * tokenCount]),
                    packedIndices: flatKeyPacked.reshaped([B * nKVHeads * tokenCount, -1]),
                    tokenCount: tokenCount, dim: headDim, bits: self.keyBits
                )).reshaped([B * nKVHeads, tokenCount, headDim])

                let expandedK: MLXArray
                if nRepeats > 1 {
                    expandedK = MLX.repeated(
                        dequantKRot.reshaped([B * nKVHeads, 1, tokenCount, headDim]),
                        count: nRepeats, axis: 1
                    ).reshaped([B * nQHeads * L, tokenCount, headDim])
                } else {
                    expandedK = dequantKRot
                }

                var scores = matmul(
                    flatQ.reshaped([B * nQHeads * L, 1, headDim]),
                    expandedK.transposed(0, 2, 1)
                ).squeezed(axis: 1).reshaped([B, nQHeads, L, tokenCount])

                switch mask {
                case .array(let maskArray):
                    if maskArray.dtype == .bool {
                        scores = MLX.where(
                            maskArray, scores,
                            MLXArray(Float.leastNormalMagnitude, dtype: scores.dtype))
                    } else {
                        scores = scores + maskArray
                    }
                case .none: break
                default: break
                }

                let attnWeights = softmax(scores, axis: -1)
                // Materialize before dequant+matmul chain — prevents lazy graph overflow
                eval(attnWeights)

                let flatWeights = attnWeights.reshaped([B * nQHeads * L, tokenCount]).contiguous()

                let dequantV = valueMSECodec.decodeRotated(MSECodecState(
                    norms: flatValNorms.reshaped([B * nKVHeads * tokenCount]),
                    packedIndices: flatValPacked.reshaped([B * nKVHeads * tokenCount, -1]),
                    tokenCount: tokenCount, dim: headDim, bits: self.valueBits
                )).reshaped([B * nKVHeads, tokenCount, headDim])

                let expandedV: MLXArray
                if nRepeats > 1 {
                    expandedV = MLX.repeated(
                        dequantV.reshaped([B * nKVHeads, 1, tokenCount, headDim]),
                        count: nRepeats, axis: 1
                    ).reshaped([B * nQHeads * L, tokenCount, headDim])
                } else {
                    expandedV = dequantV
                }

                let rotatedOutput = matmul(
                    flatWeights.expandedDimensions(axis: 1),
                    expandedV
                ).squeezed(axis: 1)

                output = matmul(
                    rotatedOutput.reshaped([B, nQHeads, L, headDim]),
                    valueMSECodec.rotation
                )
            }
        }

        return output
    }

    /// Approximate memory footprint in bytes (excludes shared codec overhead).
    public var memoryBytes: Int {
        var total = 0
        if let rk = rawKeys { total += rk.shape.reduce(1, *) * rk.dtype.bytesPerScalar }
        if let rv = rawValues { total += rv.shape.reduce(1, *) * rv.dtype.bytesPerScalar }
        if let kp = keyPackedMSE { total += kp.shape.reduce(1, *) * kp.dtype.bytesPerScalar }
        if let kn = keyNorms { total += kn.shape.reduce(1, *) * kn.dtype.bytesPerScalar }
        if let vp = valPackedMSE { total += vp.shape.reduce(1, *) * vp.dtype.bytesPerScalar }
        if let vn = valNorms { total += vn.shape.reduce(1, *) * vn.dtype.bytesPerScalar }
        if let dk = dequantKeys { total += dk.shape.reduce(1, *) * dk.dtype.bytesPerScalar }
        if let dv = dequantValues { total += dv.shape.reduce(1, *) * dv.dtype.bytesPerScalar }
        return total
    }

    override public var state: [MLXArray] {
        get {
            if isCompressed {
                if rawKeyMode {
                    guard let rk = rawKeys,
                        let vpm = valPackedMSE, let vn = valNorms,
                        offset > 0
                    else { return [] }
                    return [
                        rk[0..., 0..., ..<offset, 0...],
                        vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset],
                    ]
                } else {
                    guard let kpm = keyPackedMSE, let kn = keyNorms,
                        let vpm = valPackedMSE, let vn = valNorms,
                        offset > 0
                    else { return [] }
                    return [
                        kpm[0..., 0..., ..<offset, 0...], kn[0..., 0..., ..<offset],
                        vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset],
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
                valNorms = newValue[2]
                offset = newValue[0].dim(2)
                compressedAllocSteps = newValue[1].dim(2)
                isCompressed = true
            } else if newValue.count == 4 {
                keyPackedMSE = newValue[0]
                keyNorms = newValue[1]
                valPackedMSE = newValue[2]
                valNorms = newValue[3]
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
            keyNorms = nil
            valPackedMSE = nil
            valNorms = nil
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
