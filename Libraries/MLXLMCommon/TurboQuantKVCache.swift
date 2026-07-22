// Copyright © 2026 Apple Inc.
//
// Implements TurboQuant Algorithm 1 (MSE-optimal, arXiv:2504.19874) for KV cache:
//   rotation Π + optimal Lloyd-Max scalar codebook quantization on Beta distribution.
//
// Both keys and values use Algorithm 1 (MSE-only at b bits). QJL (Algorithm 2)
// is omitted, Tom Turney's research shows no quality benefit on Apple Silicon,
// and at 4-bit the MSE bias is negligible (paper Section 3.2: bias = 2/π,
// diminishing with bit-width).
//
// Enhancements beyond paper:
//   - Norm extraction/restoration: paper assumes ||x||=1; we store norms for arbitrary vectors
//   - Norm correction: store ||x|| / ||ỹ|| for dense rotation path (WHT skips, orthogonal preserves norms)
//   - WHT rotation option: O(d log d) butterfly in Metal kernel for power-of-2 dims
//   - Two-phase architecture: raw prefill → batch compress → compressed decode
//   - Pre-rotated queries: q' = Π·q computed once, reused for all cached keys
//
// References:
//   - TurboQuant: https://arxiv.org/abs/2504.19874
//   - QJL: https://arxiv.org/abs/2406.03482
//   - PolarQuant: https://arxiv.org/abs/2502.02617

import Foundation
import MLX
import MLXNN

extension DType {
    fileprivate var bytesPerElement: Int {
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

// MARK: - Codebook Generation

/// Optimal Lloyd-Max codebook centroids for Beta-distributed coordinates.
///
/// After random orthogonal rotation, each coordinate of a unit-sphere vector
/// follows Beta distribution f_X(x) ∝ (1-x²)^((d-3)/2) on [-1,1].
/// For large d, this converges to N(0, 1/d).
enum TurboQuantCodebook {

    // MARK: - Pre-computed Centroids

    /// Pre-computed Lloyd-Max centroids for common (dim, bits) pairs.
    /// Generated offline via 100-iteration weighted k-means on 32K-point Beta PDF grid.
    /// Avoids ~50ms runtime codebook generation per codec.
    ///
    /// Additional dims (80, 96) are lazily generated on first access to support
    /// Qwen3-4B (head_dim=80) and similar models without startup cost for unused dims.
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

    /// Lock for thread-safe lazy population of precomputed centroids.
    private static let centroidLock = NSLock()

    /// Dims that should be lazily pre-populated (non-power-of-2 dims used by real models).
    /// These fall back to dense rotation path since WHT requires power-of-2, but still
    /// benefit from cached centroids to avoid ~50ms runtime k-means per codec init.
    ///
    /// - 80: Qwen3-4B (head_dim=80)
    /// - 96: Various smaller models
    private static let lazyDims: [Int] = [80, 96]
    private static let lazyBits: [Int] = [2, 3, 4]

    /// Ensure centroids for a given dim are populated. Thread-safe, generates once.
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

    // MARK: - Public API

    /// Codebook centroids for (dim, bits). Uses pre-computed table for common configs,
    /// lazily generates and caches for known model dims (80, 96), falls back to
    /// runtime generation for truly uncommon ones.
    ///
    static func codebook(dim: Int, bits: Int) -> MLXArray {
        if let dimTable = precomputed[dim], let centroids = dimTable[bits] {
            return MLXArray(centroids)
        }
        // Lazy populate for known model dims (Qwen3-4B dim=80, etc.)
        if lazyDims.contains(dim) {
            ensureCentroidsPopulated(dim: dim)
            if let dimTable = precomputed[dim], let centroids = dimTable[bits] {
                return MLXArray(centroids)
            }
        }
        let centroids = generateCentroids(dim: dim, bits: bits)
        return MLXArray(centroids)
    }

    /// Codebook boundaries (midpoints between adjacent centroids).
    static func boundaries(dim: Int, bits: Int) -> MLXArray {
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
    /// Used as fallback for uncommon (dim, bits) pairs not in the pre-computed table.
    static func generateCentroids(dim: Int, bits: Int) -> [Float] {
        let levels = 1 << bits
        let gridSize = 32768
        let sigma = 1.0 / sqrt(Float(dim))

        // Generate grid points and PDF weights
        var grid = [Float](repeating: 0, count: gridSize)
        var weights = [Float](repeating: 0, count: gridSize)
        for i in 0 ..< gridSize {
            let x = -1.0 + 2.0 * Float(i) / Float(gridSize - 1)
            grid[i] = x
            // Beta PDF ∝ (1 - x²)^((d-3)/2), approximated by Gaussian for large d
            let exponent = Float(dim - 3) / 2.0
            let w = pow(max(1.0 - x * x, 1e-30), exponent)
            weights[i] = w
        }

        // Initialize centroids via quantiles
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

        // K-means iterations
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

// MARK: - Rotation Matrix

/// Random orthogonal rotation matrix generation.
///
/// TurboQuant Algorithm 1 line 2: Π ∈ ℝ^(d×d) via QR decomposition
/// on random Gaussian matrix. Sign-corrected for determinism.
enum TurboQuantRotation {

    /// Generate a deterministic random orthogonal rotation matrix (dense, d×d).
    /// Uses QR decomposition on CPU (not yet GPU-supported in MLX).
    static func rotationMatrix(dim: Int, seed: UInt64) -> MLXArray {
        let key = MLXRandom.key(seed)
        let gaussian = MLXRandom.normal([dim, dim], key: key)

        // QR on CPU (MLX GPU QR not supported yet). MLX's QR runs in f32
        // regardless of input dtype, so orthogonality is good to ~1e-3, // fine for the dense fallback path, whose matched-norm scales absorb
        // the residual. Real models (pow2 head dims) take the exact WHT path.
        let (q, r) = MLXLinalg.qr(gaussian, stream: .cpu)
        let diagR = r.diagonal(stream: .cpu)
        let signs = sign(diagR, stream: .cpu)
        let result = q * expandedDimensions(signs, axis: 0)
        eval(result)
        return result
    }

    /// Generate a Hadamard matrix of size dim × dim via recursive Kronecker product.
    /// Requires dim to be a power of 2. The resulting matrix H satisfies H·H = dim·I.
    static func hadamardMatrix(dim: Int) -> MLXArray {
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2")
        // Build recursively: H_1 = [[1]], H_2n = [[H_n, H_n], [H_n, -H_n]]
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
        eval(result)
        return result
    }

    /// Generate WHT sign vector: random ±1 per dimension, length d.
    /// Used with Walsh-Hadamard Transform for O(d log d) rotation.
    static func whtSigns(dim: Int, seed: UInt64) -> MLXArray {
        let key = MLXRandom.key(seed)
        // Random bits → ±1
        // Generate random ±1 signs using uniform random
        let uniform = MLXRandom.uniform(low: 0, high: 1, [dim], key: key)
        let signs = MLX.where(
            uniform .> MLXArray(Float(0.5)), MLXArray(Float(1.0)), MLXArray(Float(-1.0)))
        eval(signs)
        return signs
    }

    /// Apply WHT butterfly on the last dimension of x. Shape-preserving.
    /// Computes unnormalized Walsh-Hadamard transform: H * x along last dim.
    private static func whtButterfly(_ x: MLXArray) -> MLXArray {
        let dim = x.dim(-1)
        let logDim = Int(log2(Double(dim)))
        let origShape = x.shape
        // Flatten leading dims: [N, dim]
        let N = origShape.dropLast().reduce(1, *)
        var y = x.reshaped([N, dim])

        for s in 0 ..< logDim {
            let halfBlock = 1 << s
            let blockSize = halfBlock << 1
            let numBlocks = dim / blockSize
            // Reshape to [N, numBlocks, 2, halfBlock]
            y = y.reshaped([N, numBlocks, blockSize])
            let a = y[0..., 0..., ..<halfBlock]  // [N, numBlocks, halfBlock]
            let b = y[0..., 0..., halfBlock...]  // [N, numBlocks, halfBlock]
            let sumAB = a + b
            let diffAB = a - b
            y = concatenated([sumAB, diffAB], axis: -1)  // [N, numBlocks, blockSize]
            y = y.reshaped([N, dim])
        }

        return y.reshaped(origShape)
    }

    /// Apply SRHT forward rotation: y = H * diag(signs) * x / sqrt(dim)
    /// Works on the last dimension of any-shaped input (e.g. [B, H, T, D]).
    /// Uses butterfly pattern, O(d log d) vs O(d²) for dense matmul.
    static func fwhtForward(_ x: MLXArray, signs: MLXArray) -> MLXArray {
        let dim = x.dim(-1)
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2")
        let signed = x * signs
        let transformed = whtButterfly(signed)
        return transformed * MLXArray(Float(1.0 / sqrt(Float(dim))))
    }

    /// Apply SRHT inverse rotation: x = diag(signs) * H * y / sqrt(dim)
    /// WHT is self-inverse up to scale. Inverse of (H·D/√d) is (D·H/√d).
    static func fwhtInverse(_ y: MLXArray, signs: MLXArray) -> MLXArray {
        let dim = y.dim(-1)
        precondition(dim > 0 && (dim & (dim - 1)) == 0, "dim must be power of 2")
        let transformed = whtButterfly(y)
        return transformed * MLXArray(Float(1.0 / sqrt(Float(dim)))) * signs
    }
}

// MARK: - Bit Packing

/// Efficient bit packing/unpacking for codebook indices.
enum TurboQuantPacking {

    /// Number of uint32 words needed to pack `count` values at `bits` each.
    static func packedWidth(count: Int, bits: Int) -> Int {
        (count * bits + 31) / 32
    }

    /// Pack b-bit indices into uint32 words.
    /// Input: [rows, count] as uint32 (values 0..2^bits-1)
    /// Output: [rows, packedWidth] as uint32
    static func packLowBit(_ indices: MLXArray, bits: Int) -> MLXArray {
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
        let packed = concatenated(wordArrays, axis: -1)  // [rows, pw]
        return packed.reshaped(batchShape + [pw])
    }

    /// Unpack b-bit indices from uint32 words.
    /// Input: [rows, packedWidth] as uint32
    /// Output: [rows, count] as uint32
    static func unpackLowBit(_ packed: MLXArray, bits: Int, count: Int) -> MLXArray {
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
        let unpacked = concatenated(dimArrays, axis: -1)  // [rows, count]
        return unpacked.reshaped(batchShape + [count])
    }
}

// MARK: - MSE Codec (TurboQuant Algorithm 1)

/// State for MSE-quantized vectors.
struct MSECodecState {
    var norms: MLXArray  // [B, H, T], original vector L2 norms
    var packedIndices: MLXArray  // [B, H, T, PackedWidth], packed codebook indices
    var tokenCount: Int
    let dim: Int
    let bits: Int
}

/// MSE-optimal codec per TurboQuant Algorithm 1.
///
/// QUANT: y ← Π·x, idx_j ← argmin|y_j - c_k|
/// DEQUANT: ỹ_j ← c_{idx_j}, x̃ ← Π^T · ỹ
class MSECodec {
    let dim: Int
    let bits: Int
    let seed: UInt64

    /// Codebook centroids [2^bits]
    let codebook: MLXArray
    /// Codebook boundaries for fast quantization [2^bits - 1]
    let boundaries: MLXArray

    /// Whether to use WHT (power-of-2 dim) or dense rotation
    let useWHT: Bool
    /// WHT sign vector [dim], for O(d log d) Metal encode kernel (power-of-2 dims only)
    let whtSigns: MLXArray?
    /// Dense rotation matrix Π [dim, dim], used for decode/query rotation (single matmul, fast)
    let rotation: MLXArray
    /// Π^T, for forward rotation
    let rotationT: MLXArray

    init(dim: Int, bits: Int, seed: UInt64 = 42) {
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.codebook = TurboQuantCodebook.codebook(dim: dim, bits: bits)
        self.boundaries = TurboQuantCodebook.boundaries(dim: dim, bits: bits)

        // Use WHT for power-of-2 dims (O(d log d) Metal encode kernel)
        let isPowerOf2 = dim > 0 && (dim & (dim - 1)) == 0
        self.useWHT = isPowerOf2 && dim <= 1024
        if useWHT {
            let signs = TurboQuantRotation.whtSigns(dim: dim, seed: seed)
            self.whtSigns = signs
            // Build dense WHT rotation matrix for decode/query path (single matmul is faster
            // than FWHT butterfly via MLX ops due to graph overhead)
            let hadamard = TurboQuantRotation.hadamardMatrix(dim: dim)
            let signsDiag = expandedDimensions(signs, axis: 0)
            let whtRot = hadamard * signsDiag / MLXArray(Float(sqrt(Float(dim))))
            eval(whtRot)
            self.rotation = whtRot
            self.rotationT = whtRot.transposed()
        } else {
            self.whtSigns = nil
            self.rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: seed)
            self.rotationT = self.rotation.transposed()
        }
    }

    /// Encode vectors (Algorithm 1 QUANT) with optional norm correction and optional
    /// per-dimension key calibration.
    /// Input: [B, H, T, D]
    /// Returns MSECodecState with norms and packed indices.
    ///
    /// WHT path: stores raw norms directly. WHT is an orthogonal transform that preserves
    /// norms, so reconstruction_norm ≈ original_norm (within floating point error).
    /// Skipping norm correction saves one codebook lookup, one norm computation, and one
    /// division per encoded vector.
    ///
    /// Dense rotation path: stores `original_norm / reconstruction_norm` (norm correction).
    /// This compensates for quantization error in the non-orthogonal rotation case.
    ///
    /// `scale`, when non-nil, is `[dim]` and divides the rotated vector elementwise
    /// before quantization: quantize(rotate(k) / scale). This is TurboQuant key
    /// calibration (equalize post-rotation per-dimension variance on
    /// quantization-sensitive families, see TurboQuantKVCache.computeKeyCalibrationScale).
    /// Dividing by scale breaks the orthogonal norm-preservation the WHT raw-norm
    /// shortcut relies on, so norm correction always runs when scale is supplied,
    /// even on the WHT path.
    /// Rotation front half of `encode`: unit-normalize and rotate.
    ///
    /// Exposed separately so the calibrated fused-kernel dispatch quantizes
    /// exactly these values. Boundary quantization is sensitive to the fp
    /// reduce order of the rotation; real caches contain degenerate rows
    /// (attention sinks) whose rotated coordinates sit near quantization
    /// boundaries, so an in-kernel rotation with a different reduce order
    /// flips many dimensions of exactly the rows softmax weights most.
    /// Sharing these ops makes kernel and MLX encode indices identical.
    func rotatedUnit(_ vectors: MLXArray) -> (rotated: MLXArray, norms: MLXArray) {
        // Extract norms and normalize (paper assumes unit sphere; we store norms separately)
        let norms = sqrt((vectors * vectors).sum(axis: -1))
        let safeNorms = maximum(norms, MLXArray(Float(1e-8)))
        let unit = vectors / expandedDimensions(safeNorms, axis: -1)

        // Rotate: y ← Π · x (Algorithm 1 line 5)
        return (matmul(unit, rotationT), norms)
    }

    func encode(_ vectors: MLXArray, scale: MLXArray? = nil) -> MSECodecState {
        let (rotated, norms) = rotatedUnit(vectors)
        let calibrated = scale.map { rotated / $0 } ?? rotated

        // Quantize via boundary comparison (fast, no broadcast)
        let indices = boundaryQuantize(calibrated)

        let storedNorms: MLXArray
        if let scale {
            // Calibrated path: undo the /scale divide before measuring the
            // reconstruction, per-vector norm is computed AFTER scaling, which
            // matches what the flash kernels actually reconstruct (query is
            // pre-multiplied by scale, so the dot product resolves in the true
            // rotated space, see TurboQuantKVCache.compressedAttention).
            let reconstructed = codebook[indices] * scale
            let reconNormSq = (reconstructed * reconstructed).sum(axis: -1)
            let reconNorms = sqrt(maximum(reconNormSq, MLXArray(Float(1e-16))))
            storedNorms = norms / reconNorms
        } else if useWHT {
            // WHT fast path: orthogonal transform preserves norms, skip correction
            storedNorms = norms
        } else {
            // Dense rotation path: norm correction compensates for quantization error
            let reconstructed = codebook[indices]  // [B,H,T,D], quantized approximation in rotated space
            let reconNormSq = (reconstructed * reconstructed).sum(axis: -1)
            let reconNorms = sqrt(maximum(reconNormSq, MLXArray(Float(1e-16))))
            storedNorms = norms / reconNorms  // original_norm / reconstruction_norm
        }

        // Pack indices
        let packed = TurboQuantPacking.packLowBit(indices, bits: bits)

        return MSECodecState(
            norms: storedNorms,
            packedIndices: packed,
            tokenCount: vectors.dim(2),
            dim: dim,
            bits: bits
        )
    }

    /// Decode from state (Algorithm 1 DEQUANT).
    /// `scale`, when non-nil, undoes the calibration divide applied at
    /// `encode(_:scale:)`: multiply the codebook lookup elementwise by scale
    /// before inverse rotation. Must match the scale passed to encode.
    /// Returns: [B, H, T, D]
    func decode(_ state: MSECodecState, scale: MLXArray? = nil) -> MLXArray {
        // Unpack indices
        let indices = TurboQuantPacking.unpackLowBit(state.packedIndices, bits: bits, count: dim)

        // Codebook lookup: ỹ_j ← c_{idx_j} (Algorithm 1 line 9)
        var approx = codebook[indices]
        if let scale {
            approx = approx * scale
        }

        // Inverse rotate: x̃ ← Π^T · ỹ (Algorithm 1 line 10)
        let unrotated = matmul(approx, rotation)

        // Rescale by stored norms
        return expandedDimensions(state.norms, axis: -1) * unrotated
    }

    /// Decode in rotated space (skip inverse rotation).
    /// Returns centroid values scaled by norm, still in Π-rotated coordinate space.
    /// Used with pre-rotated queries for dequant-first SDPA.
    func decodeRotated(_ state: MSECodecState) -> MLXArray {
        let indices = TurboQuantPacking.unpackLowBit(state.packedIndices, bits: bits, count: dim)
        let approx = codebook[indices]
        return expandedDimensions(state.norms, axis: -1) * approx
    }

    /// Pre-rotate queries for compressed-domain scoring.
    /// q' ← Π · q (once per query, reused for all cached keys)
    func prepareQueries(_ queries: MLXArray) -> MLXArray {
        return matmul(queries, rotationT)
    }

    /// Fast quantization via boundary comparison instead of argmin broadcast.
    /// boundaries = sorted midpoints between adjacent centroids.
    /// Returns uint32 indices in [0, 2^bits - 1].
    func boundaryQuantize(_ rotated: MLXArray) -> MLXArray {
        // For each coordinate, count how many boundaries it exceeds
        // This gives the codebook index directly
        let ndim = rotated.ndim
        let expanded = expandedDimensions(rotated, axis: -1)  // [..., D, 1]
        // Reshape boundaries to broadcast: [1, 1, ..., 1, numBoundaries]
        var bShape = [Int](repeating: 1, count: ndim + 1)
        bShape[ndim] = boundaries.count
        let b = boundaries.reshaped(bShape)
        let greater = (expanded .> b).asType(.uint32)  // compare against all boundaries
        let indices = greater.sum(axis: -1)  // count exceeded = index
        return indices.asType(.uint32)
    }
}

// MARK: - TurboQuantKVCache

/// KV cache using TurboQuant compression with two-phase architecture:
///
/// **Phase 1, Prefill** (L>1): Store raw K/V like KVCacheSimple. Zero overhead.
/// **Transition**: On first decode call, compress entire raw cache in one batch.
/// **Phase 2, Decode** (L=1): Encode 1 new token. Metal kernel scores against
///   all compressed tokens. Zero dequantization.
///
/// Both keys and values: Algorithm 1 (MSE at b bits, no QJL)
public class TurboQuantKVCache: BaseKVCache {

    public let bits: Int  // Legacy: used when keyBits == valueBits
    public let keyBits: Int  // Bit-width for key compression (0 = raw FP16, no compression)
    public let valueBits: Int  // Bit-width for value compression (can be lower, V compression is nearly free)
    private let seed: UInt64

    /// Affine-K mode: keys stored as 8-bit affine-quantized (upstream QuantizedKVCache
    /// layout, groupSize 64) while values are TurboQuant compressed. The efficient K per
    /// the asymmetric-kv paper: near-lossless like FP16 keys at half the bytes.
    /// Enabled when keyBits == 8.
    public let affineKeyMode: Bool

    /// Affine-K quantization group size. Defaults to 64, but callers that
    /// already know the head dimension (e.g. `maybeTurboQuantizeKVCache`)
    /// can pass a value pre-resolved via `resolvedKVQuantizationGroupSize`
    /// so kernels and quantization agree. Resolved (and possibly adjusted)
    /// against the actual head dimension the first time an affine-K encode
    /// runs, see `resolveAffineKeyGroupSize(headDim:)`.
    public private(set) var keyGroupSize: Int

    // Affine-K storage (wq/scales/biases triplet, grown in steps)
    private var affKeyW: MLXArray?
    private var affKeyScales: MLXArray?
    private var affKeyBiases: MLXArray?

    /// Raw-K mode: keys stay at FP16 (uncompressed) while only values are TurboQuant compressed.
    /// This is the single biggest quality finding from TurboQuant+, K precision dominates
    /// quality via softmax amplification, V compression is nearly free (linear averaging).
    /// Enabled when keyBits == 0.
    public let rawKeyMode: Bool

    // Codecs (lazy init)
    private var keyMSECodec: MSECodec?  // keyBits for keys
    private var valueMSECodec: MSECodec?  // valueBits for values

    // Phase 1: Raw K/V storage (like KVCacheSimple), used during prefill
    private var rawKeys: MLXArray?  // [B, H, allocSteps, D]
    private var rawValues: MLXArray?  // [B, H, allocSteps, D]
    private var rawAllocSteps = 0

    // Phase 2: Compressed storage, used during decode
    // MSE-only: packed indices + norms (no QJL, simpler, same quality)
    private var keyPackedMSE: MLXArray?
    private var keyNorms: MLXArray?
    private var valPackedMSE: MLXArray?
    private var valNorms: MLXArray?
    private var compressedAllocSteps = 0

    /// Per-dimension key calibration scale s ∈ ℝ^dim, computed once at
    /// compressRawCache time from the full prefill K cache and reused for
    /// every subsequent encodeNewToken call. `nil` means identity (no
    /// calibration): the fast fused Metal encode kernels stay in use for
    /// keys. Only the standard (both-K-and-V-quantized) symmetric schemes
    /// ever set this, rawKeyMode/affineKeyMode keys are untouched.
    private var keyCalibScale: MLXArray?

    /// Whether we've transitioned from raw → compressed
    public private(set) var isCompressed = false

    private let step = 256

    public init(
        bits: Int = 4, keyBits: Int? = nil, valueBits: Int? = nil, seed: UInt64 = 42,
        keyGroupSize: Int = 64
    ) {
        self.bits = bits
        self.keyBits = keyBits ?? bits
        self.valueBits = valueBits ?? bits
        self.rawKeyMode = (keyBits ?? bits) == 0
        self.affineKeyMode = (keyBits ?? bits) == 8
        self.seed = seed
        self.keyGroupSize = keyGroupSize
        super.init()
    }

    /// Resolves (and, if needed, adjusts) `keyGroupSize` against the actual
    /// key head dimension the first time an affine-K encode runs. Callers
    /// that pre-resolve the group size (`maybeTurboQuantizeKVCache`) hit a
    /// no-op here; a directly constructed cache with an incompatible head
    /// dimension gets an actionable crash instead of an unhelpful failure
    /// deep inside MLX's `quantized(...)`.
    private func resolveAffineKeyGroupSize(headDim: Int) {
        guard
            let resolved = resolvedKVQuantizationGroupSize(
                requested: keyGroupSize, keyHeadDim: headDim, valueHeadDim: headDim)
        else {
            fatalError(
                "TurboQuant affine key quantization requires head dimensions divisible by one of the supported group sizes (32, 64, 128). Requested group size: \(keyGroupSize). Key head dim: \(headDim)."
            )
        }
        keyGroupSize = resolved
    }

    override public var isTrimmable: Bool { true }

    // MARK: - Shared Codec Cache

    /// Shared codec cache: all layers with the same (dim, bits, seed) reuse the same codec.
    /// Eliminates 56 redundant [128,128] rotation matrices (~7 MB) across 28 layers.
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

    /// Initialize codecs if needed. Uses shared cache to avoid duplicating rotation matrices.
    /// In rawKeyMode, key codec is nil, keys stay at FP16, no rotation/quantization needed.
    private func ensureCodecs(headDim: Int) {
        guard valueMSECodec == nil else { return }
        if !rawKeyMode && !affineKeyMode {
            keyMSECodec = Self.getOrCreateCodec(dim: headDim, bits: keyBits, seed: seed)
        }
        valueMSECodec = Self.getOrCreateCodec(dim: headDim, bits: valueBits, seed: seed + 1)
    }

    /// Dispatch to WHT or dense fused encode kernel based on codec configuration.
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

    /// Calibrated key encode (per-dimension key calibration). The rotation
    /// runs as MLX ops through `MSECodec.rotatedUnit`, the same ops
    /// `MSECodec.encode(_:scale:)` uses, so the kernel quantizes
    /// bit-identical rotated values; only the boundary compare, packing, and
    /// norm correction run in the fused kernel. Rotating in-kernel instead
    /// flips quantization ties on degenerate rows (attention sinks) whose
    /// coordinates sit near boundaries, which measurably distorts attention.
    private func fusedEncodeDispatchScaled(
        input: MLXArray, scale: MLXArray, codec: MSECodec, headDim: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        let (rotated, norms) = codec.rotatedUnit(input.asType(.float32))
        return TurboQuantKernelOps.fusedQuantizePackScaled(
            rotated: rotated, rawNorms: norms, scale: scale,
            boundaries: codec.boundaries, codebook: codec.codebook,
            bits: codec.bits, dim: headDim
        )
    }

    // MARK: - Phase 1: Raw Prefill

    /// Prefill update: store raw K/V, return raw. Zero encoding overhead.
    /// Uses KVCacheSimple-style allocation with concatenated growth.
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

    /// Compute per-dimension key calibration scale from the rotated raw K cache.
    ///
    /// Hypothesis: post-rotation per-dimension variance is not flat for some
    /// model families (qk-norm Qwens, small models); equalizing it before
    /// scalar quantization reduces key error on the dimensions that actually
    /// carry signal. s_d = std(rotate(K)[..., d]) pooled over every cached
    /// (batch, head, token), normalized to mean 1 and clamped to [0.25, 4.0]
    /// so no single dimension dominates or collapses.
    ///
    /// Score correctness: with keys encoded as quantize(rotate(k) / s), folding
    /// s into the query side (prepareQueries, see compressedAttention) restores
    /// the exact dot product, rotate(q)*s · rotate(k)/s = rotate(q)·rotate(k),
    /// so the flash/score kernels stay completely untouched.
    ///
    /// Returns nil (identity, skip calibration) when the sample is too small
    /// to estimate a stable std, or when the resulting scale already sits
    /// within 10% of flat everywhere. In that case calibration would only add
    /// MLX-ops overhead on the key encode path (it forces off the fused Metal
    /// kernel) for no quality gain.
    private func computeKeyCalibrationScale(allKeys: MLXArray, codec: MSECodec) -> MLXArray? {
        let dim = allKeys.dim(-1)
        let sampleCount = allKeys.dim(0) * allKeys.dim(1) * allKeys.dim(2)
        guard sampleCount >= 32 else { return nil }

        let keysF32 = allKeys.asType(.float32)
        let norms = sqrt((keysF32 * keysF32).sum(axis: -1))
        let safeNorms = maximum(norms, MLXArray(Float(1e-8)))
        let unit = keysF32 / expandedDimensions(safeNorms, axis: -1)
        let rotated = matmul(unit, codec.rotationT).reshaped([-1, dim])

        let perDimStd = std(rotated, axis: 0)
        let meanStd = maximum(perDimStd.mean(), MLXArray(Float(1e-8)))
        let clamped = clip(perDimStd / meanStd, min: Float(0.25), max: Float(4.0))
        eval(clamped)

        let maxDeviation = (clamped - MLXArray(Float(1.0))).abs().max().item(Float.self)
        guard maxDeviation >= 0.1 else { return nil }
        return clamped
    }

    // MARK: - Transition: Compress Raw Cache

    /// Compress the entire raw K/V cache into packed format in one batch.
    /// Called once when transitioning from prefill to decode.
    ///
    /// In rawKeyMode: only compress values. Keys stay as raw FP16 in rawKeys buffer.
    /// This is the highest-quality TurboQuant+ mode, K precision dominates quality.
    private func compressRawCache() {
        guard !isCompressed, let rk = rawKeys, let rv = rawValues, offset > 0 else { return }
        let allKeys = rk[.ellipsis, ..<offset, 0...]
        let allValues = rv[.ellipsis, ..<offset, 0...]
        let headDim = allKeys.dim(-1)
        ensureCodecs(headDim: headDim)
        compressRawCacheInternal(allKeys: allKeys, allValues: allValues, headDim: headDim)
        if rawKeyMode {
            // Keep rawKeys alive, they're our FP16 key storage going forward
            // Only free rawValues since those are now compressed
            rawValues = nil
        } else if affineKeyMode {
            // Keys now live in the affine triplet
            rawKeys = nil
            rawValues = nil
            rawAllocSteps = 0
        } else {
            rawKeys = nil
            rawValues = nil
            rawAllocSteps = 0
        }
        isCompressed = true
        MLX.Memory.clearCache()
    }

    /// Compress given raw K/V arrays into packed format.
    ///
    /// In rawKeyMode: only compress values. Keys are not encoded, they stay as raw FP16
    /// in the rawKeys buffer. keyPackedMSE and keyNorms remain nil.
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

        if affineKeyMode {
            resolveAffineKeyGroupSize(headDim: headDim)
            let quantK = quantized(
                allKeys, groupSize: keyGroupSize, bits: 8)
            let (kw, ks) = (quantK.wq, quantK.scales)
            let kb = quantK.biases ?? MLXArray.zeros(ks.shape, dtype: ks.dtype)
            affKeyW = MLXArray.zeros(
                [B, H, allocSteps, kw.dim(-1)], dtype: kw.dtype)
            affKeyScales = MLXArray.zeros(
                [B, H, allocSteps, ks.dim(-1)], dtype: ks.dtype)
            affKeyBiases = MLXArray.zeros(
                [B, H, allocSteps, kb.dim(-1)], dtype: kb.dtype)
            affKeyW![.ellipsis, ..<tokenCount, 0...] = kw
            affKeyScales![.ellipsis, ..<tokenCount, 0...] = ks
            affKeyBiases![.ellipsis, ..<tokenCount, 0...] = kb
        } else if !rawKeyMode {
            // Compress keys too (standard TurboQuant path)
            guard let keyMSECodec else { return }
            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)

            // Key calibration: computed once from the full prefill K cache,
            // reused for every subsequent encodeNewToken call on this cache.
            keyCalibScale = computeKeyCalibrationScale(allKeys: allKeys, codec: keyMSECodec)

            let keyPackedShaped: MLXArray
            let keyNormsShaped: MLXArray
            let flatKeys = allKeys.reshaped([B * H * tokenCount, headDim])
            if let keyCalibScale {
                // Calibrated keys route through the scale-aware fused Metal
                // kernel (mirrors MSECodec.encode(scale:)'s math), keeping
                // the fast path instead of falling back to MLX ops.
                let (keyPackedFlat, keyNormsFlat) = fusedEncodeDispatchScaled(
                    input: flatKeys, scale: keyCalibScale, codec: keyMSECodec, headDim: headDim)
                keyPackedShaped = keyPackedFlat.reshaped([B, H, tokenCount, kpw])
                keyNormsShaped = keyNormsFlat.reshaped([B, H, tokenCount])
            } else {
                let (keyPackedFlat, keyNormsFlat) = fusedEncodeDispatch(
                    input: flatKeys, codec: keyMSECodec, headDim: headDim)
                keyPackedShaped = keyPackedFlat.reshaped([B, H, tokenCount, kpw])
                keyNormsShaped = keyNormsFlat.reshaped([B, H, tokenCount])
            }

            keyPackedMSE = MLXArray.zeros([B, H, allocSteps, kpw], dtype: .uint32)
            keyNorms = MLXArray.zeros([B, H, allocSteps])
            keyPackedMSE![.ellipsis, ..<tokenCount, 0...] = keyPackedShaped
            keyNorms![.ellipsis, ..<tokenCount] = keyNormsShaped
        }

        valPackedMSE![.ellipsis, ..<tokenCount, 0...] = valPackedFlat.reshaped([
            B, H, tokenCount, vpw,
        ])
        valNorms![.ellipsis, ..<tokenCount] = valNormsFlat.reshaped([B, H, tokenCount])

        compressedAllocSteps = allocSteps

        if rawKeyMode {
            eval(valPackedMSE!, valNorms!)
        } else if affineKeyMode {
            eval(affKeyW!, affKeyScales!, affKeyBiases!, valPackedMSE!, valNorms!)
        } else {
            eval(keyPackedMSE!, keyNorms!, valPackedMSE!, valNorms!)
        }
    }

    // MARK: - Phase 2: Compressed Decode

    /// Encode a single new token into compressed storage using fused Metal kernel.
    ///
    /// In rawKeyMode: keys are appended to rawKeys buffer as raw FP16 (no encoding).
    /// Only values are encoded via the fused Metal kernel.
    private func encodeNewToken(keys: MLXArray, values: MLXArray) {
        let headDim = keys.dim(-1)
        let B = keys.dim(0)
        let H = keys.dim(1)
        let numSteps = keys.dim(2)
        let prev = offset

        // A cache restored from a prompt-cache file arrives compressed but
        // with lazy codecs uninitialized, without this, the guard below
        // silently drops every new token after restore.
        ensureCodecs(headDim: headDim)
        guard let valueMSECodec else { return }

        let vpw = TurboQuantPacking.packedWidth(count: headDim, bits: valueBits)

        // Encode values via fused Metal kernel
        let flatVals = values.reshaped([B * H * numSteps, headDim])
        let (valPacked, valNormsNew) = fusedEncodeDispatch(
            input: flatVals, codec: valueMSECodec, headDim: headDim)
        let valPackedShaped = valPacked.reshaped([B, H, numSteps, vpw])
        let valNormsShaped = valNormsNew.reshaped([B, H, numSteps])

        if affineKeyMode {
            // Affine-K mode: quantize the new key chunk and append to the triplet
            resolveAffineKeyGroupSize(headDim: headDim)
            let quantK = quantized(keys, groupSize: keyGroupSize, bits: 8)
            let (kw, ks) = (quantK.wq, quantK.scales)
            let kb = quantK.biases ?? MLXArray.zeros(ks.shape, dtype: ks.dtype)

            if (prev + numSteps) > compressedAllocSteps {
                let newAlloc = ((prev + numSteps + step - 1) / step) * step
                let newKW = MLXArray.zeros([B, H, newAlloc, kw.dim(-1)], dtype: kw.dtype)
                let newKS = MLXArray.zeros([B, H, newAlloc, ks.dim(-1)], dtype: ks.dtype)
                let newKB = MLXArray.zeros([B, H, newAlloc, kb.dim(-1)], dtype: kb.dtype)
                let newVP = MLXArray.zeros([B, H, newAlloc, vpw], dtype: .uint32)
                let newVN = MLXArray.zeros([B, H, newAlloc])
                if prev > 0 {
                    newKW[.ellipsis, ..<prev, 0...] = affKeyW![.ellipsis, ..<prev, 0...]
                    newKS[.ellipsis, ..<prev, 0...] = affKeyScales![.ellipsis, ..<prev, 0...]
                    newKB[.ellipsis, ..<prev, 0...] = affKeyBiases![.ellipsis, ..<prev, 0...]
                    newVP[.ellipsis, ..<prev, 0...] = valPackedMSE![.ellipsis, ..<prev, 0...]
                    newVN[.ellipsis, ..<prev] = valNorms![.ellipsis, ..<prev]
                }
                affKeyW = newKW
                affKeyScales = newKS
                affKeyBiases = newKB
                valPackedMSE = newVP
                valNorms = newVN
                compressedAllocSteps = newAlloc
            }

            offset = prev + numSteps
            affKeyW![.ellipsis, prev ..< offset, 0...] = kw
            affKeyScales![.ellipsis, prev ..< offset, 0...] = ks
            affKeyBiases![.ellipsis, prev ..< offset, 0...] = kb
            valPackedMSE![.ellipsis, prev ..< offset, 0...] = valPackedShaped
            valNorms![.ellipsis, prev ..< offset] = valNormsShaped
        } else if rawKeyMode {
            // Raw-K mode: append keys to rawKeys buffer as FP16
            // Grow rawKeys buffer if needed
            if (prev + numSteps) > rawAllocSteps {
                let newAlloc = ((prev + numSteps + step - 1) / step) * step
                let newRK = MLXArray.zeros([B, H, newAlloc, headDim], dtype: keys.dtype)
                if prev > 0, let rk = rawKeys {
                    newRK[.ellipsis, ..<prev, 0...] = rk[.ellipsis, ..<prev, 0...]
                }
                rawKeys = newRK
                rawAllocSteps = newAlloc
            }

            // Grow compressed (value) storage
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
            // Standard TurboQuant: encode both K and V
            guard let keyMSECodec else { return }

            let kpw = TurboQuantPacking.packedWidth(count: headDim, bits: keyBits)
            let keyPackedShaped: MLXArray
            let keyNormsShaped: MLXArray
            let flatKeys = keys.reshaped([B * H * numSteps, headDim])
            if let keyCalibScale {
                // Reuse the scale computed once at compressRawCache time,
                // routed through the scale-aware fused Metal kernel to keep
                // the fast path instead of falling back to MLX ops.
                let (keyPacked, keyNormsNew) = fusedEncodeDispatchScaled(
                    input: flatKeys, scale: keyCalibScale, codec: keyMSECodec, headDim: headDim)
                keyPackedShaped = keyPacked.reshaped([B, H, numSteps, kpw])
                keyNormsShaped = keyNormsNew.reshaped([B, H, numSteps])
            } else {
                let (keyPacked, keyNormsNew) = fusedEncodeDispatch(
                    input: flatKeys, codec: keyMSECodec, headDim: headDim)
                keyPackedShaped = keyPacked.reshaped([B, H, numSteps, kpw])
                keyNormsShaped = keyNormsNew.reshaped([B, H, numSteps])
            }

            // Grow compressed storage using concatenated growth
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

    /// Compressed-domain attention via Metal kernels.
    ///
    /// On first call: compresses raw prefill cache in one batch.
    /// Then: encode 1 new token → fused attention kernel → inverse rotation.
    ///
    /// For L=1 (decode): uses TurboFlashAttention, a single Metal dispatch that fuses
    /// Q×K scoring + online softmax + Attn×V aggregation. No intermediate score or weight
    /// arrays are materialized. This reduces 3 dispatches (score + softmax + value) to 1.
    ///
    /// For L>1 (prefill chunks): falls back to separate score → softmax → value kernels
    /// since causal masking across multiple query positions requires the full score matrix.
    ///
    /// In rawKeyMode: uses standard matmul for Q*K scoring (raw FP16 keys, no rotation),
    /// then compressed-domain Metal kernel for Attn*V (TurboQuant compressed values).
    /// TurboFlash is NOT used, it assumes both K and V are packed.
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

        // Transition: compress raw cache on first decode call
        if !isCompressed {
            compressRawCache()
        }

        // Phase A: Encode new token
        encodeNewToken(keys: newKeys, values: newValues)

        guard let valueMSECodec else {
            return queries
        }

        let tokenCount = offset

        // Shared V slicing (used by all paths)
        let flatValPacked = valPackedMSE![0..., 0..., ..<tokenCount, 0...]
            .reshaped([B * nKVHeads, tokenCount, -1])
        let flatValNorms = valNorms![0..., 0..., ..<tokenCount]
            .reshaped([B * nKVHeads, tokenCount])

        let valRotation = valueMSECodec.rotation
        let output: MLXArray

        if rawKeyMode || affineKeyMode {
            // ═══ Raw-K / Affine-K + Compressed-V path ═══
            // Fast path (decode, L=1, no array mask): the SIMD-parallel flash
            // kernel that scores the symmetric schemes, with the K side reading
            // raw fp16 or inline-dequantized 8-bit affine instead of unpacking
            // turbo K. K is scored in its stored basis (queries not
            // pre-rotated); V decodes rotated and pass 2 applies the inverse
            // value rotation. This avoids the separated path's per-step
            // full-cache f32 cast and GQA materialization.
            var hasArrayMaskAsym = false
            switch mask {
            case .array, .arrays: hasArrayMaskAsym = true
            default: break
            }
            if L == 1, !hasArrayMaskAsym {
                let flatQ = (queries * MLXArray(scale)).reshaped([B * nQHeads, headDim])
                let rotated: MLXArray
                if affineKeyMode {
                    guard let kw = affKeyW, let ks = affKeyScales, let kb = affKeyBiases else {
                        return queries
                    }
                    rotated = TurboQuantKernelOps.turboFlashAffineK(
                        rotatedQueries: flatQ,
                        kWeights: kw[0..., 0..., ..<tokenCount, 0...]
                            .reshaped([B * nKVHeads, tokenCount, -1]),
                        kScales: ks[0..., 0..., ..<tokenCount, 0...]
                            .reshaped([B * nKVHeads, tokenCount, -1]),
                        kBiases: kb[0..., 0..., ..<tokenCount, 0...]
                            .reshaped([B * nKVHeads, tokenCount, -1]),
                        valPacked: flatValPacked, valNorms: flatValNorms,
                        valCodebook: valueMSECodec.codebook,
                        tokenCount: tokenCount, repeatCount: nRepeats,
                        valueBits: valueBits, dim: headDim,
                        kGroup: keyGroupSize, valRotation: valRotation)
                } else {
                    guard let rk = rawKeys else { return queries }
                    rotated = TurboQuantKernelOps.turboFlashRawK(
                        rotatedQueries: flatQ,
                        rawKeys: rk[0..., 0..., ..<tokenCount, 0...]
                            .reshaped([B * nKVHeads, tokenCount, headDim]),
                        valPacked: flatValPacked, valNorms: flatValNorms,
                        valCodebook: valueMSECodec.codebook,
                        tokenCount: tokenCount, repeatCount: nRepeats,
                        valueBits: valueBits, dim: headDim, valRotation: valRotation)
                }
                let out = rotated.reshaped([B, nQHeads, 1, headDim])
                return out.dtype == queries.dtype ? out : out.asType(queries.dtype)
            }

            var scores: MLXArray
            if affineKeyMode {
                guard let kw = affKeyW, let ks = affKeyScales, let kb = affKeyBiases else {
                    return queries
                }
                // Score in f32: f16 activations (Qwen2.5-style, K amax in the
                // hundreds) overflow the f16 dot product and NaN the softmax.
                var q = queries.asType(.float32) * MLXArray(scale)
                var kwS = kw[.ellipsis, ..<tokenCount, 0...]
                var ksS = ks[.ellipsis, ..<tokenCount, 0...]
                var kbS = kb[.ellipsis, ..<tokenCount, 0...]
                if nRepeats > 1 {
                    q = q.reshaped([B, nKVHeads, nRepeats, L, headDim])
                    kwS = expandedDimensions(kwS, axis: -3)
                    ksS = expandedDimensions(ksS, axis: -3)
                    kbS = expandedDimensions(kbS, axis: -3)
                }
                scores = quantizedMM(
                    q, kwS, scales: ksS, biases: kbS,
                    transpose: true, groupSize: keyGroupSize, bits: 8)
                if nRepeats > 1 {
                    scores = scores.reshaped([B, nQHeads, L, tokenCount])
                }
            } else {
                // Q*K scoring: standard matmul with raw FP16 keys
                guard let rk = rawKeys else { return queries }
                let allKeys = rk[.ellipsis, ..<tokenCount, 0...]  // [B, nKVHeads, T, D]

                // GQA: expand keys to match query heads
                let expandedKeys: MLXArray
                if nRepeats > 1 {
                    let expanded = expandedDimensions(allKeys, axis: 2)
                    let tiledKeys = MLX.tiled(expanded, repetitions: [1, 1, nRepeats, 1, 1])
                    expandedKeys = tiledKeys.reshaped([B, nQHeads, tokenCount, headDim])
                } else {
                    expandedKeys = allKeys
                }

                // scores = Q * K^T * scale  → [B, nQHeads, L, T], f32: the f16
                // dot product overflows on outlier-heavy families (Qwen2.5
                // layer-0/last K amax ~200-400 → q·k beyond f16 max).
                scores =
                    matmul(
                        queries.asType(.float32),
                        expandedKeys.asType(.float32).transposed(0, 1, 3, 2)) * MLXArray(scale)
            }

            // Mask + softmax
            switch mask {
            case .array(let maskArray):
                if maskArray.dtype == .bool {
                    scores = MLX.where(maskArray, scores, MLXArray(-Float.greatestFiniteMagnitude))
                } else {
                    scores = scores + maskArray
                }
            case .causal:
                // Build causal mask manually
                let queryOffset = tokenCount - L
                let causalMask = MLXArray.tri(L, m: tokenCount, k: queryOffset, type: Bool.self)
                let expandedMask = expandedDimensions(
                    expandedDimensions(causalMask, axis: 0), axis: 0)
                scores = MLX.where(expandedMask, scores, MLXArray(-Float.greatestFiniteMagnitude))
            case .arrays(let maskArrays):
                if let maskArray = maskArrays.first {
                    if maskArray.dtype == .bool {
                        scores = MLX.where(
                            maskArray, scores, MLXArray(-Float.greatestFiniteMagnitude))
                    } else {
                        scores = scores + maskArray
                    }
                }
            case .none: break
            }

            let attnWeights = softmax(scores, axis: -1)

            // Attn*V: compressed-domain Metal kernel for weighted sum of TurboQuant values
            let flatWeights = attnWeights.reshaped([B * nQHeads * L, tokenCount])
            let rotatedOutput = TurboQuantKernelOps.mseWeightedSum(
                weights: flatWeights, packed: flatValPacked, norms: flatValNorms,
                codebook: valueMSECodec.codebook, tokenCount: tokenCount,
                repeatCount: nRepeats, bits: self.valueBits, dim: headDim, queryChunkLength: L
            )

            // Inverse value rotation (V was encoded in rotated space)
            output = matmul(
                rotatedOutput.reshaped([B, nQHeads, L, headDim]),
                valueMSECodec.rotation
            )
        } else {
            // ═══ Standard TurboQuant path: both K and V compressed ═══
            guard let keyMSECodec else { return queries }

            // Pre-rotate query for compressed-domain K scoring. When key
            // calibration is active, fold s into the query side (elementwise,
            // post-rotation): score = rotate(q)*s · rotate(k)/s = rotate(q)·rotate(k).
            // The flash/score kernels stay untouched, they only ever see a
            // plain dot product between rotatedQueries and the packed K codebook.
            var qRot = keyMSECodec.prepareQueries(queries) * MLXArray(scale)
            if let keyCalibScale {
                qRot = qRot * keyCalibScale
            }
            let flatQ = qRot.reshaped([B * nQHeads * L, headDim])

            // K slicing
            let flatKeyPacked = keyPackedMSE![0..., 0..., ..<tokenCount, 0...]
                .reshaped([B * nKVHeads, tokenCount, -1])
            let flatKeyNorms = keyNorms![0..., 0..., ..<tokenCount]
                .reshaped([B * nKVHeads, tokenCount])

            let hasArrayMask: Bool
            switch mask {
            case .array, .arrays: hasArrayMask = true
            default: hasArrayMask = false
            }

            if L == 1 && !hasArrayMask {
                // TurboFlashAttention path (decode, L=1; causal is a no-op at
                // L=1, and array masks take the separated path below so they
                // are actually applied)
                output = TurboQuantKernelOps.turboFlashAttention(
                    rotatedQueries: flatQ,
                    keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                    keyCodebook: keyMSECodec.codebook,
                    valPacked: flatValPacked, valNorms: flatValNorms,
                    valCodebook: valueMSECodec.codebook,
                    tokenCount: tokenCount, repeatCount: nRepeats,
                    keyBits: self.keyBits, valueBits: self.valueBits, dim: headDim,
                    valRotation: valRotation
                ).reshaped([B, nQHeads, L, headDim])
            } else if case .causal = mask {
                // Causal TurboFlashAttention path (prefill, L>1)
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
                // Separated path (L>1, non-causal masks)
                var scores = TurboQuantKernelOps.mseScore(
                    rotatedQueries: flatQ, packed: flatKeyPacked, norms: flatKeyNorms,
                    codebook: keyMSECodec.codebook, tokenCount: tokenCount,
                    repeatCount: nRepeats, bits: self.keyBits, dim: headDim, queryChunkLength: L
                ).reshaped([B, nQHeads, L, tokenCount])

                // Mask + softmax
                switch mask {
                case .array(let maskArray):
                    if maskArray.dtype == .bool {
                        scores = MLX.where(
                            maskArray, scores, MLXArray(-Float.greatestFiniteMagnitude))
                    } else {
                        scores = scores + maskArray
                    }
                case .arrays(let maskArrays):
                    if let maskArray = maskArrays.first {
                        if maskArray.dtype == .bool {
                            scores = MLX.where(
                                maskArray, scores, MLXArray(-Float.greatestFiniteMagnitude))
                        } else {
                            scores = scores + maskArray
                        }
                    }
                case .none: break
                default: break
                }

                let attnWeights = softmax(scores, axis: -1)

                // Metal value kernel
                let flatWeights = attnWeights.reshaped([B * nQHeads * L, tokenCount])
                let rotatedOutput = TurboQuantKernelOps.mseWeightedSum(
                    weights: flatWeights, packed: flatValPacked, norms: flatValNorms,
                    codebook: valueMSECodec.codebook, tokenCount: tokenCount,
                    repeatCount: nRepeats, bits: self.valueBits, dim: headDim, queryChunkLength: L
                )

                // Inverse rotation
                output = matmul(
                    rotatedOutput.reshaped([B, nQHeads, L, headDim]),
                    valueMSECodec.rotation
                )
            }
        }

        // Kernels emit f32; return in the activation dtype so a turbo layer
        // does not promote the whole downstream stream (and every later
        // layer's KV cache) to f32.
        return output.dtype == queries.dtype ? output : output.asType(queries.dtype)
    }

    // MARK: - Memory Reporting

    /// Actual memory footprint: compressed storage (packed indices + norms) for K and V,
    /// plus any raw FP16 buffers if still in prefill phase or rawKeyMode.
    /// Does NOT include codec overhead (rotation matrices, codebooks) which is shared across layers.
    /// In rawKeyMode: rawKeys is always present (FP16 keys), no keyPackedMSE/keyNorms.
    public var memoryBytes: Int {
        var total = 0
        // Raw FP16 buffers (always present in rawKeyMode for keys, or during prefill)
        if let rk = rawKeys { total += rk.shape.reduce(1, *) * rk.dtype.bytesPerElement }
        if let rv = rawValues { total += rv.shape.reduce(1, *) * rv.dtype.bytesPerElement }
        // Compressed storage (K only present when NOT rawKeyMode)
        if let kw = affKeyW { total += kw.shape.reduce(1, *) * kw.dtype.bytesPerElement }
        if let ks = affKeyScales { total += ks.shape.reduce(1, *) * ks.dtype.bytesPerElement }
        if let kb = affKeyBiases { total += kb.shape.reduce(1, *) * kb.dtype.bytesPerElement }
        if let kp = keyPackedMSE { total += kp.shape.reduce(1, *) * kp.dtype.bytesPerElement }
        if let kn = keyNorms { total += kn.shape.reduce(1, *) * kn.dtype.bytesPerElement }
        if let vp = valPackedMSE { total += vp.shape.reduce(1, *) * vp.dtype.bytesPerElement }
        if let vn = valNorms { total += vn.shape.reduce(1, *) * vn.dtype.bytesPerElement }
        if let kcs = keyCalibScale { total += kcs.shape.reduce(1, *) * kcs.dtype.bytesPerElement }
        return total
    }

    // MARK: - State / Trim

    override public var state: [MLXArray] {
        get {
            if isCompressed {
                if affineKeyMode {
                    // Affine-K compressed: [kW, kScales, kBiases, valPacked, valNorms]
                    guard let kw = affKeyW, let ks = affKeyScales, let kb = affKeyBiases,
                        let vpm = valPackedMSE, let vn = valNorms,
                        offset > 0
                    else { return [] }
                    return [
                        kw[0..., 0..., ..<offset, 0...],
                        ks[0..., 0..., ..<offset, 0...],
                        kb[0..., 0..., ..<offset, 0...],
                        vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset],
                    ]
                } else if rawKeyMode {
                    // Raw-K mode compressed: [rawKeys, valPacked, valNorms]
                    guard let rk = rawKeys,
                        let vpm = valPackedMSE, let vn = valNorms,
                        offset > 0
                    else { return [] }
                    return [
                        rk[0..., 0..., ..<offset, 0...],
                        vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset],
                    ]
                } else {
                    // Standard compressed: [keyPacked, keyNorms, valPacked, valNorms]
                    // (+ keyCalibScale [dim] when key calibration is active)
                    guard let kpm = keyPackedMSE, let kn = keyNorms,
                        let vpm = valPackedMSE, let vn = valNorms,
                        offset > 0
                    else { return [] }
                    var arrays = [
                        kpm[0..., 0..., ..<offset, 0...], kn[0..., 0..., ..<offset],
                        vpm[0..., 0..., ..<offset, 0...], vn[0..., 0..., ..<offset],
                    ]
                    if let keyCalibScale {
                        arrays.append(keyCalibScale)
                    }
                    return arrays
                }
            } else {
                guard let rk = rawKeys, let rv = rawValues, offset > 0 else { return [] }
                return [rk[0..., 0..., ..<offset, 0...], rv[0..., 0..., ..<offset, 0...]]
            }
        }
        set {
            if affineKeyMode && newValue.count == 5 {
                // Affine-K compressed state
                affKeyW = newValue[0]
                affKeyScales = newValue[1]
                affKeyBiases = newValue[2]
                valPackedMSE = newValue[3]
                valNorms = newValue[4]
                offset = newValue[0].dim(2)
                compressedAllocSteps = offset
                isCompressed = true
            } else if rawKeyMode && newValue.count == 3 {
                // Raw-K mode compressed state: [rawKeys, valPacked, valNorms]
                rawKeys = newValue[0]
                rawAllocSteps = newValue[0].dim(2)
                valPackedMSE = newValue[1]
                valNorms = newValue[2]
                offset = newValue[0].dim(2)
                compressedAllocSteps = newValue[1].dim(2)
                isCompressed = true
            } else if !affineKeyMode && !rawKeyMode && newValue.count == 5 {
                // Standard compressed state with key calibration:
                // [keyPacked, keyNorms, valPacked, valNorms, keyCalibScale]
                keyPackedMSE = newValue[0]
                keyNorms = newValue[1]
                valPackedMSE = newValue[2]
                valNorms = newValue[3]
                keyCalibScale = newValue[4]
                offset = newValue[0].dim(2)
                compressedAllocSteps = offset
                isCompressed = true
            } else if newValue.count == 4 {
                // Standard compressed state: [keyPacked, keyNorms, valPacked, valNorms]
                keyPackedMSE = newValue[0]
                keyNorms = newValue[1]
                valPackedMSE = newValue[2]
                valNorms = newValue[3]
                offset = newValue[0].dim(2)
                compressedAllocSteps = offset
                isCompressed = true
            } else if newValue.count == 2 {
                // Raw state
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
        let trimCount = min(n, offset)
        offset -= trimCount
        if offset == 0 {
            rawKeys = nil
            rawValues = nil
            rawAllocSteps = 0
            keyPackedMSE = nil
            keyNorms = nil
            affKeyW = nil
            affKeyScales = nil
            affKeyBiases = nil
            valPackedMSE = nil
            valNorms = nil
            keyCalibScale = nil
            compressedAllocSteps = 0
            isCompressed = false
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
    case "turbo0v3": return (0, 3)
    case "turbo0v2": return (0, 2)
    // Affine-K asymmetric family (keyBits == 8 → 8-bit affine keys, turbo
    // values): the recommended ladder from the asymmetric-kv paper. q8-K is
    // near-lossless at half the key bytes; f16-K buys nothing further.
    case "turbo8v4": return (8, 4)
    case "turbo8v3": return (8, 3)
    case "turbo8v2": return (8, 2)
    // Key-quantizing (symmetric) schemes: maximum compression. K sensitivity
    // varies by model family, validate on your model; the asym family above
    // is the recommended starting point.
    case "turbo4": return (4, 4)
    case "turbo4v2": return (4, 2)
    case "turbo3": return (3, 3)
    case "turbo2": return (2, 2)

    default: return nil
    }
}

/// Namespace holding the dedup state for the "rotating layers kept fp16"
/// notice. Mirrors the `nonisolated(unsafe) static var` + `NSLock` idiom
/// used elsewhere in this file (e.g. `TurboQuantCodebook.precomputed`).
private enum RotatingSkipNotice {
    /// Once per process: the notice is informational and repeating it per
    /// generation adds noise without new signal. Tracking instances instead
    /// would grow without bound, `newCache(parameters:)` allocates fresh
    /// caches for every generation.
    nonisolated(unsafe) static var logged = false
    static let lock = NSLock()
}

/// Log, at most once per set of `RotatingKVCache` instances, that a
/// requested TurboQuant strategy is leaving sliding-window layers at fp16.
///
/// TurboQuant only compresses `KVCacheSimple` layers (see
/// `maybeTurboQuantizeKVCache` below); `RotatingKVCache` (Gemma-style
/// sliding-window) layers are already memory-bounded by their window size,
/// and their rotating/eviction storage layout is not compatible with the
/// sequential-append compression path, so they are intentionally left
/// untouched. The notice tells a user which layers kept fp16 rotating
/// caches so a partially engaged scheme is visible.
private func logRotatingKVCacheSkipOnce(leaves: [KVCacheLeaf]) {
    let paths = leaves.compactMap { leaf -> [Int]? in
        guard case .rotating = leaf.kind else { return nil }
        return leaf.path
    }
    guard !paths.isEmpty else { return }

    RotatingSkipNotice.lock.lock()
    defer { RotatingSkipNotice.lock.unlock() }

    guard !RotatingSkipNotice.logged else { return }
    RotatingSkipNotice.logged = true

    let indexList = paths.map { $0.map(String.init).joined(separator: ".") }
        .joined(separator: ", ")
    print(
        "[TurboQuant] KV compression was requested, but layer(s) at index \(indexList) "
            + "use RotatingKVCache (sliding-window) and will stay fp16. TurboQuant only "
            + "compresses non-rotating (global) KV cache layers."
    )
}

/// Convert eligible `KVCacheSimple` layers to `TurboQuantKVCache` once their
/// offset passes `quantizedKVStart`, transferring the accumulated KV state.
/// Mamba / wrapper caches are left untouched. `RotatingKVCache` layers
/// (Gemma-style sliding-window) are also left untouched -- see
/// `logRotatingKVCacheSkipOnce` -- since they are already memory-bounded by
/// their window size and their rotating storage layout does not fit the
/// sequential-append compression path. Called from `maybeQuantizeKVCache`
/// when `kvScheme` names a TurboQuant scheme.
@discardableResult
public func maybeTurboQuantizeKVCache(
    cache: inout [KVCache],
    keyBits: Int,
    valueBits: Int,
    quantizedKVStart: Int
) -> Bool {
    let leaves = KVCacheTree.leaves(in: cache)
    logRotatingKVCacheSkipOnce(leaves: leaves)

    // Boundary layer protection: the first and last attention layers are
    // disproportionately sensitive to KV quantization, keeping 2 on each
    // end at FP16 recovers 37-91% of the quality gap at minimal
    // compression cost. Non-attention layers (Mamba/wrapper caches) are
    // excluded from the rank count so hybrids protect the right layers.
    let protectedPaths = KVCacheTree.turboQuantProtectedPaths(
        in: leaves, keyBits: keyBits, valueBits: valueBits)

    var awaitsCompressionStart = false
    KVCacheTree.rewrite(&cache) { leaf in
        guard case .simple(let simple) = leaf.kind else { return leaf.cache }
        guard simple.offset > quantizedKVStart else {
            awaitsCompressionStart = true
            return simple
        }

        let state = simple.innerState()
        let headDims: (key: Int, value: Int)? =
            state.count >= 2 ? (state[0].dim(3), state[1].dim(3)) : nil

        if protectedPaths.contains(leaf.path) {
            // Boundary layers use 8-bit affine instead of the turbo scheme:
            // near-lossless protection for the quantization-sensitive first
            // and last layers at a quarter of the fp16 cost. Skip the
            // conversion (leave fp16) rather than crash when neither head
            // dimension divides one of the supported affine group sizes.
            guard
                let headDims,
                resolvedKVQuantizationGroupSize(
                    requested: 64, keyHeadDim: headDims.key, valueHeadDim: headDims.value) != nil
            else { return simple }
            return simple.toQuantized(groupSize: 64, bits: 8)
        }

        // Affine-K mode (keyBits == 8) quantizes keys in groups, resolve the
        // group size against this layer's head dimension up front so the
        // TurboQuantKVCache instance never has to fall back at first encode.
        // Skip layers whose head dimension divides none of the supported
        // sizes rather than constructing a cache that would only crash on
        // its first compress. Other modes (raw / symmetric turbo) don't
        // group-quantize keys, so they carry the unused default unchanged.
        var resolvedKeyGroupSize = 64
        if keyBits == 8 {
            guard
                let headDims,
                let resolved = resolvedKVQuantizationGroupSize(
                    requested: 64, keyHeadDim: headDims.key, valueHeadDim: headDims.value)
            else { return simple }
            resolvedKeyGroupSize = resolved
        }

        let turbo = TurboQuantKVCache(
            bits: max(keyBits, valueBits), keyBits: keyBits, valueBits: valueBits,
            keyGroupSize: resolvedKeyGroupSize)
        // Transfer existing KV data, trimmed to the live offset (the simple
        // cache over-allocates in steps).
        let offset = simple.offset
        if state.count >= 2, offset > 0 {
            let keys = state[0][.ellipsis, ..<offset, 0...]
            let values = state[1][.ellipsis, ..<offset, 0...]
            _ = turbo.update(keys: keys, values: values)
        }
        return turbo
    }
    return !awaitsCompressionStart
}
