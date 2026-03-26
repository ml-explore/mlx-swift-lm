import Foundation
import MLX

public enum KVQuantizationScheme: String, Sendable, Codable {
    case uniform
    case turboQuant = "turboquant"
}

private let turboQuantEpsilon: Float = 1e-6
private let turboQuantDefaultSeed = 0
private let turboQuantCodebookGridSize = 32_768
private let turboQuantCodebookIterations = 100

public func turboquantEnabled(
    bits: Float?,
    scheme: KVQuantizationScheme = .uniform
) -> Bool {
    guard let bits else { return false }
    if scheme == .turboQuant {
        return true
    }
    return !isIntegerBitWidth(bits)
}

private func isIntegerBitWidth(_ bits: Float) -> Bool {
    abs(bits - round(bits)) <= turboQuantEpsilon
}

private func validateTurboQuantBits(_ bits: Float) -> Float {
    let rounded = (bits * 2).rounded() / 2
    precondition(rounded >= 1, "TurboQuant requires kvBits >= 1.")
    precondition(
        abs(bits - rounded) <= turboQuantEpsilon,
        "TurboQuant supports integer and .5 bit widths, got \(bits)."
    )
    return rounded
}

private struct TurboQuantMSEState {
    var norms: MLXArray
    var indices: MLXArray
}

private struct TurboQuantProdState {
    var norms: MLXArray
    var mseIndices: MLXArray
    var residualNorms: MLXArray
    var qjlSigns: MLXArray
}

private struct TurboQuantSplitState {
    var low: TurboQuantState
    var high: TurboQuantState
}

private indirect enum TurboQuantState {
    case mse(TurboQuantMSEState)
    case prod(TurboQuantProdState)
    case split(TurboQuantSplitState)

    var length: Int {
        switch self {
        case .mse(let state):
            return state.norms.dim(2)
        case .prod(let state):
            return state.norms.dim(2)
        case .split(let state):
            return state.low.length
        }
    }

    var nbytes: Int {
        switch self {
        case .mse(let state):
            return state.norms.nbytes + state.indices.nbytes
        case .prod(let state):
            return state.norms.nbytes + state.mseIndices.nbytes + state.residualNorms.nbytes
                + state.qjlSigns.nbytes
        case .split(let state):
            return state.low.nbytes + state.high.nbytes
        }
    }

    func slice(end: Int) -> TurboQuantState {
        switch self {
        case .mse(let state):
            return .mse(
                .init(
                    norms: state.norms[.ellipsis, ..<end],
                    indices: state.indices[.ellipsis, ..<end, 0...]
                ))
        case .prod(let state):
            return .prod(
                .init(
                    norms: state.norms[.ellipsis, ..<end],
                    mseIndices: state.mseIndices[.ellipsis, ..<end, 0...],
                    residualNorms: state.residualNorms[.ellipsis, ..<end],
                    qjlSigns: state.qjlSigns[.ellipsis, ..<end, 0...]
                ))
        case .split(let state):
            return .split(.init(low: state.low.slice(end: end), high: state.high.slice(end: end)))
        }
    }

    func slice(range: Range<Int>) -> TurboQuantState {
        switch self {
        case .mse(let state):
            return .mse(
                .init(
                    norms: state.norms[.ellipsis, range],
                    indices: state.indices[.ellipsis, range, 0...]
                ))
        case .prod(let state):
            return .prod(
                .init(
                    norms: state.norms[.ellipsis, range],
                    mseIndices: state.mseIndices[.ellipsis, range, 0...],
                    residualNorms: state.residualNorms[.ellipsis, range],
                    qjlSigns: state.qjlSigns[.ellipsis, range, 0...]
                ))
        case .split(let state):
            return .split(
                .init(low: state.low.slice(range: range), high: state.high.slice(range: range)))
        }
    }

    func allocateLike(length: Int) -> TurboQuantState {
        switch self {
        case .mse(let state):
            return .mse(
                .init(
                    norms: MLXArray.zeros(
                        [state.norms.dim(0), state.norms.dim(1), length],
                        dtype: state.norms.dtype
                    ),
                    indices: MLXArray.zeros(
                        [state.indices.dim(0), state.indices.dim(1), length, state.indices.dim(3)],
                        dtype: state.indices.dtype
                    )
                ))
        case .prod(let state):
            return .prod(
                .init(
                    norms: MLXArray.zeros(
                        [state.norms.dim(0), state.norms.dim(1), length],
                        dtype: state.norms.dtype
                    ),
                    mseIndices: MLXArray.zeros(
                        [
                            state.mseIndices.dim(0), state.mseIndices.dim(1), length,
                            state.mseIndices.dim(3),
                        ],
                        dtype: state.mseIndices.dtype
                    ),
                    residualNorms: MLXArray.zeros(
                        [state.residualNorms.dim(0), state.residualNorms.dim(1), length],
                        dtype: state.residualNorms.dtype
                    ),
                    qjlSigns: MLXArray.zeros(
                        [
                            state.qjlSigns.dim(0), state.qjlSigns.dim(1), length,
                            state.qjlSigns.dim(3),
                        ],
                        dtype: state.qjlSigns.dtype
                    )
                ))
        case .split(let state):
            return .split(
                .init(
                    low: state.low.allocateLike(length: length),
                    high: state.high.allocateLike(length: length)
                ))
        }
    }

    mutating func write(_ source: TurboQuantState, start: Int) {
        let end = start + source.length
        switch (self, source) {
        case (.mse(var destination), .mse(let sourceState)):
            destination.norms[.ellipsis, start ..< end] = sourceState.norms
            destination.indices[.ellipsis, start ..< end, 0...] = sourceState.indices
            self = .mse(destination)
        case (.prod(var destination), .prod(let sourceState)):
            destination.norms[.ellipsis, start ..< end] = sourceState.norms
            destination.mseIndices[.ellipsis, start ..< end, 0...] = sourceState.mseIndices
            destination.residualNorms[.ellipsis, start ..< end] = sourceState.residualNorms
            destination.qjlSigns[.ellipsis, start ..< end, 0...] = sourceState.qjlSigns
            self = .prod(destination)
        case (.split(var destination), .split(let sourceState)):
            destination.low.write(sourceState.low, start: start)
            destination.high.write(sourceState.high, start: start)
            self = .split(destination)
        default:
            fatalError("TurboQuant state type mismatch during write.")
        }
    }
}

private indirect enum TurboQuantPreparedQueries {
    case array(MLXArray)
    case pair(MLXArray, MLXArray)
    case split(TurboQuantPreparedQueries, TurboQuantPreparedQueries)
}

private final class TurboQuantCodecDescriptor: Codable, @unchecked Sendable {
    enum Kind: String, Codable, Sendable {
        case mse
        case prod
        case split
    }

    let kind: Kind
    let dim: Int
    let bits: Float
    let seed: Int
    let lowIndices: [Int]?
    let highIndices: [Int]?
    let low: TurboQuantCodecDescriptor?
    let high: TurboQuantCodecDescriptor?

    init(
        kind: Kind,
        dim: Int,
        bits: Float,
        seed: Int,
        lowIndices: [Int]?,
        highIndices: [Int]?,
        low: TurboQuantCodecDescriptor?,
        high: TurboQuantCodecDescriptor?
    ) {
        self.kind = kind
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.lowIndices = lowIndices
        self.highIndices = highIndices
        self.low = low
        self.high = high
    }
}

private struct TurboQuantCacheMetadata: Codable, Sendable {
    let offset: Int
    let bits: Float
    let seed: Int
    let keyCodec: TurboQuantCodecDescriptor?
    let valueCodec: TurboQuantCodecDescriptor?
}

private protocol TurboQuantCodec: AnyObject {
    var dim: Int { get }
    var descriptor: TurboQuantCodecDescriptor { get }

    func quantize(_ vectors: MLXArray) -> TurboQuantState
    func dequantize(_ state: TurboQuantState) -> MLXArray
    func prepareQueries(_ queries: MLXArray) -> TurboQuantPreparedQueries
    func scorePrepared(_ preparedQueries: TurboQuantPreparedQueries, state: TurboQuantState)
        -> MLXArray
    func weightedSum(_ weights: MLXArray, state: TurboQuantState) -> MLXArray
    func weightedSumFromScores(_ scores: MLXArray, state: TurboQuantState) -> MLXArray
    func weightedSumStatsFromScores(_ scores: MLXArray, state: TurboQuantState) -> (
        MLXArray, MLXArray, MLXArray
    )
}

private func l2Norm(_ array: MLXArray) -> MLXArray {
    sqrt(sum(square(array), axis: -1))
}

private struct TurboQuantRandom {
    private var state: UInt64
    private var spare: Float?

    init(seed: Int) {
        self.state = UInt64(bitPattern: Int64(seed)) &+ 0x9E37_79B9_7F4A_7C15
        self.spare = nil
    }

    mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func nextUniform() -> Float {
        Float(Double(nextUInt64()) / Double(UInt64.max))
    }

    mutating func nextNormal() -> Float {
        if let spare {
            self.spare = nil
            return spare
        }

        let u1 = max(nextUniform(), 1e-7)
        let u2 = nextUniform()
        let radius = sqrt(-2 * log(u1))
        let angle = 2 * Float.pi * u2
        let z0 = radius * cos(angle)
        let z1 = radius * sin(angle)
        self.spare = z1
        return z0
    }
}

private final class TurboQuantMatrixCache: @unchecked Sendable {
    static let shared = TurboQuantMatrixCache()

    private let lock = NSLock()
    private var rotations: [String: MLXArray] = [:]
    private var projections: [String: MLXArray] = [:]
    private var codebooks: [String: MLXArray] = [:]

    func rotation(dim: Int, seed: Int) -> MLXArray {
        let key = "rotation:\(dim):\(seed)"
        lock.lock()
        if let cached = rotations[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let created = buildRotationMatrix(dim: dim, seed: seed)

        lock.lock()
        rotations[key] = created
        lock.unlock()
        return created
    }

    func projection(dim: Int, seed: Int) -> MLXArray {
        let key = "projection:\(dim):\(seed)"
        lock.lock()
        if let cached = projections[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let created = buildProjectionMatrix(dim: dim, seed: seed)

        lock.lock()
        projections[key] = created
        lock.unlock()
        return created
    }

    func codebook(dim: Int, bits: Int) -> MLXArray {
        let key = "codebook:\(dim):\(bits)"
        lock.lock()
        if let cached = codebooks[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let created = buildCodebook(dim: dim, bits: bits)

        lock.lock()
        codebooks[key] = created
        lock.unlock()
        return created
    }
}

private func buildRotationMatrix(dim: Int, seed: Int) -> MLXArray {
    if dim <= 0 {
        return MLXArray.zeros([0, 0], dtype: .float32)
    }
    if dim == 1 {
        return MLXArray([1 as Float]).reshaped([1, 1]).asType(.float32)
    }

    var rng = TurboQuantRandom(seed: seed + dim * 7_919)
    var matrix = Array(repeating: Array(repeating: Float(0), count: dim), count: dim)
    for row in 0 ..< dim {
        for column in 0 ..< dim {
            matrix[row][column] = rng.nextNormal()
        }
    }

    var orthonormal = Array(repeating: Array(repeating: Float(0), count: dim), count: dim)
    for column in 0 ..< dim {
        var vector = (0 ..< dim).map { matrix[$0][column] }
        for previous in 0 ..< column {
            let dot = zip(vector, (0 ..< dim).map { orthonormal[$0][previous] }).reduce(Float(0)) {
                $0 + $1.0 * $1.1
            }
            for row in 0 ..< dim {
                vector[row] -= dot * orthonormal[row][previous]
            }
        }

        let norm = sqrt(vector.reduce(Float(0)) { $0 + $1 * $1 })
        if norm <= turboQuantEpsilon {
            for row in 0 ..< dim {
                vector[row] = row == column ? 1 : 0
            }
        } else {
            for row in 0 ..< dim {
                vector[row] /= norm
            }
        }

        if vector[column] < 0 {
            for row in 0 ..< dim {
                vector[row] *= -1
            }
        }

        for row in 0 ..< dim {
            orthonormal[row][column] = vector[row]
        }
    }

    let flattened = orthonormal.flatMap { $0 }
    return MLXArray(flattened).reshaped([dim, dim]).asType(.float32)
}

private func buildProjectionMatrix(dim: Int, seed: Int) -> MLXArray {
    if dim <= 0 {
        return MLXArray.zeros([0, 0], dtype: .float32)
    }

    var rng = TurboQuantRandom(seed: seed + dim * 2_971 + 17)
    let values = (0 ..< (dim * dim)).map { _ in rng.nextNormal() }
    return MLXArray(values).reshaped([dim, dim]).asType(.float32)
}

private func betaPDF(_ grid: [Float], dim: Int) -> [Float] {
    if dim <= 1 {
        return Array(repeating: 1 / Float(grid.count), count: grid.count)
    }

    let coefficient =
        Float(
            tgamma(Double(dim) / 2.0)
                / (sqrt(Double.pi) * tgamma(Double(dim - 1) / 2.0)))
    let exponent = Float(dim - 3) / 2
    var weights = grid.map { value in
        coefficient * pow(max(1 - value * value, 0), exponent)
    }
    let total = weights.reduce(Float(0), +)
    guard total > 0 else {
        return Array(repeating: 1 / Float(grid.count), count: grid.count)
    }
    for index in weights.indices {
        weights[index] /= total
    }
    return weights
}

private func interpolate(grid: [Float], cdf: [Float], quantile: Float) -> Float {
    if quantile <= cdf[0] {
        return grid[0]
    }
    if quantile >= cdf[cdf.count - 1] {
        return grid[grid.count - 1]
    }

    var low = 0
    var high = cdf.count - 1
    while low + 1 < high {
        let middle = (low + high) / 2
        if cdf[middle] < quantile {
            low = middle
        } else {
            high = middle
        }
    }

    let cdfLow = cdf[low]
    let cdfHigh = cdf[high]
    if abs(cdfHigh - cdfLow) <= turboQuantEpsilon {
        return grid[high]
    }
    let t = (quantile - cdfLow) / (cdfHigh - cdfLow)
    return grid[low] + t * (grid[high] - grid[low])
}

private func buildCodebook(dim: Int, bits: Int) -> MLXArray {
    if bits <= 0 {
        return MLXArray.zeros([0], dtype: .float32)
    }

    let levels = 1 << bits
    if dim <= 1 {
        let centroids = (0 ..< levels).map { index in
            -1 + (2 * Float(index) / Float(max(levels - 1, 1)))
        }
        return MLXArray(centroids).asType(.float32)
    }

    let denominator = Float(turboQuantCodebookGridSize - 1)
    let grid = (0 ..< turboQuantCodebookGridSize).map { index in
        -1 + (2 * Float(index) / denominator)
    }
    let weights = betaPDF(grid, dim: dim)

    var cdf = Array(repeating: Float(0), count: weights.count)
    var runningTotal: Float = 0
    for index in weights.indices {
        runningTotal += weights[index]
        cdf[index] = runningTotal
    }

    var centroids = (0 ..< levels).map { index in
        interpolate(
            grid: grid,
            cdf: cdf,
            quantile: (Float(index) + 0.5) / Float(levels)
        )
    }

    for _ in 0 ..< turboQuantCodebookIterations {
        var boundaries = Array(repeating: Float(0), count: levels + 1)
        boundaries[0] = -1
        boundaries[levels] = 1
        for index in 1 ..< levels {
            boundaries[index] = 0.5 * (centroids[index - 1] + centroids[index])
        }

        var updated = centroids
        for level in 0 ..< levels {
            var numerator: Float = 0
            var denominator: Float = 0
            for gridIndex in grid.indices {
                let value = grid[gridIndex]
                let upperInclusive =
                    level == levels - 1
                    ? value <= boundaries[level + 1]
                    : value
                        < boundaries[level + 1]
                if value >= boundaries[level] && upperInclusive {
                    numerator += weights[gridIndex] * value
                    denominator += weights[gridIndex]
                }
            }
            if denominator > 0 {
                updated[level] = numerator / denominator
            }
        }

        let delta = zip(updated, centroids).map { abs($0 - $1) }.max() ?? 0
        centroids = updated
        if delta < turboQuantEpsilon {
            break
        }
    }

    return MLXArray(centroids).asType(.float32)
}

private func packedWidth(length: Int, bits: Int) -> Int {
    guard length > 0 && bits > 0 else { return 0 }
    return (length * bits + 31) / 32
}

private func metalAvailable() -> Bool {
    #if canImport(Metal)
        true
    #else
        false
    #endif
}

private func makeMSEScoreKernel() -> MLXFast.MLXFastKernel? {
    guard metalAvailable() else { return nil }

    let source = #"""
            auto lane = thread_position_in_grid.x;
            auto repeat_idx = thread_position_in_grid.y;
            auto n = thread_position_in_grid.z;

            auto token_count = norms_shape[2];
            auto kv_heads = norms_shape[1];
            auto repeat_count = q_rot_shape[2];
            if (repeat_idx >= repeat_count) {
                return;
            }

            auto b = n / (kv_heads * token_count);
            auto rem = n % (kv_heads * token_count);
            auto h = rem / token_count;
            auto t = rem % token_count;

            auto q_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
            auto packed_ptr = packed + ((b * kv_heads + h) * token_count + t) * PackedWidth;

            float acc = 0.0f;
            for (int d = lane; d < Dim; d += 32) {
                int bit_offset = d * Bits;
                int word_idx = bit_offset / 32;
                int offset = bit_offset % 32;
                uint value = packed_ptr[word_idx] >> offset;
                int spill = offset + Bits - 32;
                if (spill > 0) {
                    value |= packed_ptr[word_idx + 1] << (Bits - spill);
                }
                value &= ((1u << Bits) - 1u);
                acc += static_cast<float>(q_ptr[d]) * codebook[value];
            }

            acc = simd_sum(acc);
            if (thread_index_in_simdgroup == 0) {
                out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                    acc * static_cast<float>(norms[(b * kv_heads + h) * token_count + t]);
            }
        """#

    return MLXFast.metalKernel(
        name: "turboquant_mse_score",
        inputNames: ["q_rot", "norms", "packed", "codebook"],
        outputNames: ["out"],
        source: source
    )
}

private func makePackLowBitKernel() -> MLXFast.MLXFastKernel? {
    guard metalAvailable() else { return nil }

    let source = #"""
            auto word = thread_position_in_grid.x;
            auto row = thread_position_in_grid.y;

            if (row >= values_shape[0] || word >= PackedWidth) {
                return;
            }

            auto values_ptr = values + row * Length;
            uint packed_word = 0u;
            int start = max(0, (int(word) * 32 - (Bits - 1)) / Bits);
            int end = min(Length, ((int(word) + 1) * 32 + (Bits - 1)) / Bits);

            for (int idx = start; idx < end; ++idx) {
                int bit_offset = idx * Bits;
                int word_idx = bit_offset / 32;
                int offset = bit_offset % 32;
                uint value = values_ptr[idx] & ((1u << Bits) - 1u);
                if (word_idx == word) {
                    packed_word |= value << offset;
                }
                if (word_idx + 1 == word) {
                    int spill = offset + Bits - 32;
                    if (spill > 0) {
                        packed_word |= value >> (Bits - spill);
                    }
                }
            }

            out[row * PackedWidth + word] = packed_word;
        """#

    return MLXFast.metalKernel(
        name: "turboquant_pack_lowbit",
        inputNames: ["values"],
        outputNames: ["out"],
        source: source
    )
}

private func makeUnpackLowBitKernel() -> MLXFast.MLXFastKernel? {
    guard metalAvailable() else { return nil }

    let source = #"""
            auto idx = thread_position_in_grid.x;
            auto row = thread_position_in_grid.y;

            if (row >= packed_shape[0] || idx >= Length) {
                return;
            }

            auto packed_ptr = packed + row * PackedWidth;
            int bit_offset = idx * Bits;
            int word_idx = bit_offset / 32;
            int offset = bit_offset % 32;
            uint value = packed_ptr[word_idx] >> offset;
            int spill = offset + Bits - 32;
            if (spill > 0) {
                value |= packed_ptr[word_idx + 1] << (Bits - spill);
            }
            out[row * Length + idx] = value & ((1u << Bits) - 1u);
        """#

    return MLXFast.metalKernel(
        name: "turboquant_unpack_lowbit",
        inputNames: ["packed"],
        outputNames: ["out"],
        source: source
    )
}

private func makeQJLScoreKernel() -> MLXFast.MLXFastKernel? {
    guard metalAvailable() else { return nil }

    let source = #"""
            auto lane = thread_position_in_grid.x;
            auto repeat_idx = thread_position_in_grid.y;
            auto n = thread_position_in_grid.z;

            auto token_count = norms_shape[2];
            auto kv_heads = norms_shape[1];
            auto repeat_count = q_proj_shape[2];
            if (repeat_idx >= repeat_count) {
                return;
            }

            auto b = n / (kv_heads * token_count);
            auto rem = n % (kv_heads * token_count);
            auto h = rem / token_count;
            auto t = rem % token_count;

            auto q_ptr = q_proj + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
            auto packed_ptr = signs + ((b * kv_heads + h) * token_count + t) * PackedWidth;

            float acc = 0.0f;
            for (int d = lane; d < Dim; d += 32) {
                int word_idx = d / 32;
                int offset = d % 32;
                uint bit = (packed_ptr[word_idx] >> offset) & 1u;
                float sign = bit ? 1.0f : -1.0f;
                acc += static_cast<float>(q_ptr[d]) * sign;
            }

            acc = simd_sum(acc);
            if (thread_index_in_simdgroup == 0) {
                auto idx = (b * kv_heads + h) * token_count + t;
                out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                    acc
                    * static_cast<float>(norms[idx])
                    * static_cast<float>(residual_norms[idx])
                    * scale[0];
            }
        """#

    return MLXFast.metalKernel(
        name: "turboquant_qjl_score",
        inputNames: ["q_proj", "norms", "residual_norms", "signs", "scale"],
        outputNames: ["out"],
        source: source
    )
}

private func makeProdScoreKernel() -> MLXFast.MLXFastKernel? {
    guard metalAvailable() else { return nil }

    let source = #"""
            auto lane = thread_position_in_grid.x;
            auto repeat_idx = thread_position_in_grid.y;
            auto n = thread_position_in_grid.z;

            auto token_count = norms_shape[2];
            auto kv_heads = norms_shape[1];
            auto repeat_count = q_rot_shape[2];
            if (repeat_idx >= repeat_count) {
                return;
            }

            auto b = n / (kv_heads * token_count);
            auto rem = n % (kv_heads * token_count);
            auto h = rem / token_count;
            auto t = rem % token_count;

            auto q_rot_ptr = q_rot + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
            auto q_proj_ptr = q_proj + ((b * kv_heads + h) * repeat_count + repeat_idx) * Dim;
            auto mse_ptr = mse_packed + ((b * kv_heads + h) * token_count + t) * MsePackedWidth;
            auto sign_ptr = signs + ((b * kv_heads + h) * token_count + t) * SignPackedWidth;

            float mse_acc = 0.0f;
            float qjl_acc = 0.0f;
            for (int d = lane; d < Dim; d += 32) {
                int bit_offset = d * MseBits;
                int word_idx = bit_offset / 32;
                int offset = bit_offset % 32;
                uint value = mse_ptr[word_idx] >> offset;
                int spill = offset + MseBits - 32;
                if (spill > 0) {
                    value |= mse_ptr[word_idx + 1] << (MseBits - spill);
                }
                value &= ((1u << MseBits) - 1u);
                mse_acc += static_cast<float>(q_rot_ptr[d]) * codebook[value];

                int sign_word = d / 32;
                int sign_offset = d % 32;
                uint bit = (sign_ptr[sign_word] >> sign_offset) & 1u;
                float sign = bit ? 1.0f : -1.0f;
                qjl_acc += static_cast<float>(q_proj_ptr[d]) * sign;
            }

            mse_acc = simd_sum(mse_acc);
            qjl_acc = simd_sum(qjl_acc);
            if (thread_index_in_simdgroup == 0) {
                auto idx = (b * kv_heads + h) * token_count + t;
                out[((b * kv_heads + h) * repeat_count + repeat_idx) * token_count + t] =
                    static_cast<float>(norms[idx]) * (
                        mse_acc
                        + scale[0] * static_cast<float>(residual_norms[idx]) * qjl_acc
                    );
            }
        """#

    return MLXFast.metalKernel(
        name: "turboquant_prod_score",
        inputNames: [
            "q_rot", "q_proj", "norms", "residual_norms", "mse_packed", "signs", "codebook",
            "scale",
        ],
        outputNames: ["out"],
        source: source
    )
}

private func makeMSEWeightedRotKernel() -> MLXFast.MLXFastKernel? {
    guard metalAvailable() else { return nil }

    let source = #"""
            auto lane = thread_position_in_grid.x;
            auto dim_idx = thread_position_in_grid.y;
            auto n = thread_position_in_grid.z;

            if (dim_idx >= Dim) {
                return;
            }

            auto token_count = norms_shape[2];
            auto kv_heads = norms_shape[1];
            auto repeat_count = weights_shape[2];
            auto b = n / (kv_heads * repeat_count);
            auto rem = n % (kv_heads * repeat_count);
            auto h = rem / repeat_count;
            auto repeat_idx = rem % repeat_count;

            auto weights_ptr = weights + ((b * kv_heads + h) * repeat_count + repeat_idx) * token_count;
            auto norms_ptr = norms + (b * kv_heads + h) * token_count;
            auto packed_ptr = packed + ((b * kv_heads + h) * token_count) * PackedWidth;

            float acc = 0.0f;
            for (int t = lane; t < token_count; t += 32) {
                auto token_ptr = packed_ptr + t * PackedWidth;
                int bit_offset = dim_idx * Bits;
                int word_idx = bit_offset / 32;
                int offset = bit_offset % 32;
                uint value = token_ptr[word_idx] >> offset;
                int spill = offset + Bits - 32;
                if (spill > 0) {
                    value |= token_ptr[word_idx + 1] << (Bits - spill);
                }
                value &= ((1u << Bits) - 1u);
                acc += static_cast<float>(weights_ptr[t])
                    * static_cast<float>(norms_ptr[t])
                    * codebook[value];
            }

            acc = simd_sum(acc);
            if (thread_index_in_simdgroup == 0) {
                out[((b * kv_heads + h) * repeat_count + repeat_idx) * Dim + dim_idx] = acc;
            }
        """#

    return MLXFast.metalKernel(
        name: "turboquant_mse_weighted_rot",
        inputNames: ["weights", "norms", "packed", "codebook"],
        outputNames: ["out"],
        source: source
    )
}

private final class TurboQuantKernelManager: @unchecked Sendable {
    static let shared = TurboQuantKernelManager()

    let mseScoreKernel = makeMSEScoreKernel()
    let packLowBitKernel = makePackLowBitKernel()
    let unpackLowBitKernel = makeUnpackLowBitKernel()
    let qjlScoreKernel = makeQJLScoreKernel()
    let prodScoreKernel = makeProdScoreKernel()
    let mseWeightedRotKernel = makeMSEWeightedRotKernel()
}

private func packLowBit(_ values: MLXArray, bits: Int) -> MLXArray {
    if bits == 0 {
        return MLXArray.zeros(Array(values.shape.dropLast()) + [0], dtype: .uint32)
    }

    let values = values.asType(.uint32)
    let length = values.dim(-1)
    let width = packedWidth(length: length, bits: bits)
    let flat = values.reshaped([-1, length])

    if let kernel = TurboQuantKernelManager.shared.packLowBitKernel {
        let packed = kernel(
            [flat],
            template: [
                ("Bits", bits),
                ("Length", length),
                ("PackedWidth", width),
            ],
            grid: (width, flat.dim(0), 1),
            threadGroup: (min(32, max(width, 1)), 1, 1),
            outputShapes: [[flat.dim(0), width]],
            outputDTypes: [.uint32]
        )[0]
        return packed.reshaped(Array(values.shape.dropLast()) + [width])
    }

    var packed = MLXArray.zeros([flat.dim(0), width], dtype: .uint32)
    for index in 0 ..< length {
        let bitOffset = index * bits
        let wordIndex = bitOffset / 32
        let offset = bitOffset % 32
        packed[0..., wordIndex] = packed[0..., wordIndex] | (flat[0..., index] << offset)

        let spill = offset + bits - 32
        if spill > 0 {
            packed[0..., wordIndex + 1] =
                packed[0..., wordIndex + 1]
                | (flat[0..., index] >> (bits - spill))
        }
    }

    return packed.reshaped(Array(values.shape.dropLast()) + [width])
}

private func unpackLowBit(_ packed: MLXArray, bits: Int, length: Int) -> MLXArray {
    if bits == 0 {
        return MLXArray.zeros(Array(packed.shape.dropLast()) + [0], dtype: .uint32)
    }

    let packed = packed.asType(.uint32)
    let flat = packed.reshaped([-1, packed.dim(-1)])

    if let kernel = TurboQuantKernelManager.shared.unpackLowBitKernel {
        let unpacked = kernel(
            [flat],
            template: [
                ("Bits", bits),
                ("Length", length),
                ("PackedWidth", flat.dim(-1)),
            ],
            grid: (length, flat.dim(0), 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[flat.dim(0), length]],
            outputDTypes: [.uint32]
        )[0]
        return unpacked.reshaped(Array(packed.shape.dropLast()) + [length])
    }

    var unpacked = MLXArray.zeros([flat.dim(0), length], dtype: .uint32)
    let mask = (1 << bits) - 1
    for index in 0 ..< length {
        let bitOffset = index * bits
        let wordIndex = bitOffset / 32
        let offset = bitOffset % 32

        var value = flat[0..., wordIndex] >> offset
        let spill = offset + bits - 32
        if spill > 0 {
            value = value | (flat[0..., wordIndex + 1] << (bits - spill))
        }
        unpacked[0..., index] = value & MLXArray(mask, dtype: .uint32)
    }

    return unpacked.reshaped(Array(packed.shape.dropLast()) + [length])
}

private func flattenTurboQuantState(_ state: TurboQuantState) -> [MLXArray] {
    switch state {
    case .mse(let state):
        return [state.norms, state.indices]
    case .prod(let state):
        return [state.norms, state.mseIndices, state.residualNorms, state.qjlSigns]
    case .split(let state):
        return flattenTurboQuantState(state.low) + flattenTurboQuantState(state.high)
    }
}

private func unflattenTurboQuantState(
    _ arrays: ArraySlice<MLXArray>,
    descriptor: TurboQuantCodecDescriptor
) -> (TurboQuantState, Int) {
    switch descriptor.kind {
    case .mse:
        let norms = arrays[arrays.startIndex]
        let indices = arrays[arrays.startIndex + 1]
        return (.mse(.init(norms: norms, indices: indices)), 2)
    case .prod:
        let base = arrays.startIndex
        return (
            .prod(
                .init(
                    norms: arrays[base],
                    mseIndices: arrays[base + 1],
                    residualNorms: arrays[base + 2],
                    qjlSigns: arrays[base + 3]
                )), 4
        )
    case .split:
        guard let low = descriptor.low, let high = descriptor.high else {
            fatalError("TurboQuant split descriptor is missing child metadata.")
        }
        let (lowState, lowCount) = unflattenTurboQuantState(arrays, descriptor: low)
        let highStart = arrays.startIndex + lowCount
        let (highState, highCount) = unflattenTurboQuantState(
            arrays[highStart...], descriptor: high)
        return (.split(.init(low: lowState, high: highState)), lowCount + highCount)
    }
}

private func reserveTurboQuantStateCapacity(
    _ state: TurboQuantState,
    used: Int,
    needed: Int,
    step: Int
) -> TurboQuantState {
    if state.length >= needed {
        return state
    }

    var capacity = max(state.length * 2, step)
    capacity = max(capacity, needed)
    capacity = ((capacity + step - 1) / step) * step

    var grown = state.allocateLike(length: capacity)
    if used > 0 {
        grown.write(state.slice(end: used), start: 0)
    }
    return grown
}

private func buildTurboQuantCodec(
    tensor: MLXArray,
    bits: Float,
    mode: TurboQuantCodecDescriptor.Kind,
    seed: Int
) -> any TurboQuantCodec {
    let roundedBits = validateTurboQuantBits(bits)
    if isIntegerBitWidth(roundedBits) {
        let integerBits = Int(round(roundedBits))
        switch mode {
        case .mse:
            return TurboQuantMSECodec(dim: tensor.dim(-1), bits: integerBits, seed: seed)
        case .prod:
            return TurboQuantProdCodec(dim: tensor.dim(-1), bits: integerBits, seed: seed)
        case .split:
            fatalError("Split codec cannot be built directly from integer bits.")
        }
    }
    return TurboQuantSplitCodec(tensor: tensor, bits: roundedBits, mode: mode, seed: seed)
}

private func rebuildTurboQuantCodec(from descriptor: TurboQuantCodecDescriptor)
    -> any TurboQuantCodec
{
    switch descriptor.kind {
    case .mse:
        return TurboQuantMSECodec(
            dim: descriptor.dim, bits: Int(round(descriptor.bits)), seed: descriptor.seed)
    case .prod:
        return TurboQuantProdCodec(
            dim: descriptor.dim, bits: Int(round(descriptor.bits)), seed: descriptor.seed)
    case .split:
        guard let low = descriptor.low, let high = descriptor.high,
            let lowIndices = descriptor.lowIndices, let highIndices = descriptor.highIndices
        else {
            fatalError("TurboQuant split descriptor is incomplete.")
        }
        return TurboQuantSplitCodec(
            bits: descriptor.bits,
            seed: descriptor.seed,
            lowIndices: lowIndices,
            highIndices: highIndices,
            lowCodec: rebuildTurboQuantCodec(from: low),
            highCodec: rebuildTurboQuantCodec(from: high)
        )
    }
}

private final class TurboQuantMSECodec: TurboQuantCodec {
    let dim: Int
    let bits: Int
    let rotation: MLXArray
    let rotationT: MLXArray
    let codebook: MLXArray

    init(dim: Int, bits: Int, seed: Int) {
        self.dim = dim
        self.bits = bits
        self.rotation = TurboQuantMatrixCache.shared.rotation(dim: dim, seed: seed)
        self.rotationT = rotation.transposed()
        self.codebook = TurboQuantMatrixCache.shared.codebook(dim: dim, bits: bits)
        self.descriptor = .init(
            kind: .mse, dim: dim, bits: Float(bits), seed: seed, lowIndices: nil,
            highIndices: nil, low: nil, high: nil)
    }

    let descriptor: TurboQuantCodecDescriptor

    func quantize(_ vectors: MLXArray) -> TurboQuantState {
        let vectors = vectors.asType(.float32)
        let norms = l2Norm(vectors)
        let safeNorms = maximum(norms[.ellipsis, .newAxis], MLXArray(turboQuantEpsilon))
        let unitVectors = MLX.where(
            norms[.ellipsis, .newAxis] .> 0,
            vectors / safeNorms,
            MLXArray.zeros(like: vectors)
        )
        let (indices, _) = quantizeUnitWithEstimate(unitVectors)
        return .mse(.init(norms: norms.asType(vectors.dtype), indices: indices))
    }

    func dequantize(_ state: TurboQuantState) -> MLXArray {
        guard case .mse(let mseState) = state else {
            fatalError("Expected TurboQuant MSE state.")
        }
        let unitVectors = dequantizeUnit(mseState.indices)
        return mseState.norms[.ellipsis, .newAxis].asType(.float32) * unitVectors
    }

    func prepareQueries(_ queries: MLXArray) -> TurboQuantPreparedQueries {
        .array(matmul(queries, rotationT))
    }

    func scorePrepared(_ preparedQueries: TurboQuantPreparedQueries, state: TurboQuantState)
        -> MLXArray
    {
        guard case .array(let preparedQueries) = preparedQueries,
            case .mse(let mseState) = state
        else {
            fatalError("Expected TurboQuant MSE prepared queries and state.")
        }

        if preparedQueries.dim(-2) == 1,
            let fastScores = metalMSEScore(preparedQueries, state: mseState)
        {
            return fastScores
        }

        let indices = unpackLowBit(mseState.indices, bits: bits, length: dim).asType(.int32)
        let rotated = take(codebook, indices, axis: 0)
        let dots = einsum("bhmld,bhtd->bhmlt", preparedQueries, rotated)
        return dots * mseState.norms.asType(.float32)[0..., 0..., .newAxis, .newAxis, 0...]
    }

    func weightedSum(_ weights: MLXArray, state: TurboQuantState) -> MLXArray {
        guard case .mse(let mseState) = state else {
            fatalError("Expected TurboQuant MSE state.")
        }

        if weights.dim(-2) == 1, let fastOutput = metalMSEWeightedSum(weights, state: mseState) {
            return fastOutput
        }

        let indices = unpackLowBit(mseState.indices, bits: bits, length: dim).asType(.int32)
        let rotated = take(codebook, indices, axis: 0)
        let weightedRot = einsum(
            "bhmlt,bht,bhtd->bhmld",
            weights,
            mseState.norms.asType(.float32),
            rotated
        )
        return matmul(weightedRot, rotation)
    }

    func weightedSumFromScores(_ scores: MLXArray, state: TurboQuantState) -> MLXArray {
        weightedSum(softmax(scores, axis: -1), state: state)
    }

    func weightedSumStatsFromScores(_ scores: MLXArray, state: TurboQuantState) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        let maxScores = max(scores, axis: -1)
        let weights = exp(scores - maxScores[.ellipsis, .newAxis])
        let output = weightedSum(weights, state: state)
        let denominator = sum(weights, axis: -1)
        return (output, denominator, maxScores)
    }

    func quantizeUnitWithEstimate(_ unitVectors: MLXArray) -> (MLXArray, MLXArray) {
        if bits == 0 {
            let empty = MLXArray.zeros(Array(unitVectors.shape.dropLast()) + [0], dtype: .uint32)
            return (empty, MLXArray.zeros(like: unitVectors))
        }

        let rotated = matmul(unitVectors, rotationT)
        let distances = abs(rotated[.ellipsis, .newAxis] - codebook)
        let indices = argSort(distances, axis: -1)[.ellipsis, 0].asType(.uint32)
        let packed = packLowBit(indices, bits: bits)
        let estimatedRotated = take(codebook, indices.asType(.int32), axis: 0)
        let estimated = matmul(estimatedRotated, rotation)
        return (packed, estimated)
    }

    func dequantizeUnit(_ packedIndices: MLXArray) -> MLXArray {
        if bits == 0 {
            return MLXArray.zeros(Array(packedIndices.shape.dropLast()) + [dim], dtype: .float32)
        }

        let indices = unpackLowBit(packedIndices, bits: bits, length: dim).asType(.int32)
        let rotated = take(codebook, indices, axis: 0)
        return matmul(rotated, rotation)
    }

    private func metalMSEScore(_ preparedQueries: MLXArray, state: TurboQuantMSEState) -> MLXArray?
    {
        guard let kernel = TurboQuantKernelManager.shared.mseScoreKernel, state.norms.dim(2) > 0
        else {
            return nil
        }

        let squeezed = preparedQueries.squeezed(axis: 3)
        let B = squeezed.dim(0)
        let H = squeezed.dim(1)
        let R = squeezed.dim(2)
        let T = state.norms.dim(2)

        let scores = kernel(
            [squeezed, state.norms, state.indices.asType(.uint32), codebook],
            template: [
                ("Dim", dim),
                ("Bits", bits),
                ("PackedWidth", state.indices.dim(-1)),
            ],
            grid: (32, R, B * H * T),
            threadGroup: (32, 1, 1),
            outputShapes: [[B, H, R, T]],
            outputDTypes: [.float32]
        )[0]

        return expandedDimensions(scores, axis: 3)
    }

    private func metalMSEWeightedSum(_ weights: MLXArray, state: TurboQuantMSEState) -> MLXArray? {
        guard let kernel = TurboQuantKernelManager.shared.mseWeightedRotKernel,
            state.norms.dim(2) > 0
        else {
            return nil
        }

        let weights2D = weights.reshaped([
            weights.dim(0), weights.dim(1), weights.dim(2), weights.dim(4),
        ])
        let B = weights2D.dim(0)
        let H = weights2D.dim(1)
        let R = weights2D.dim(2)

        let weightedRot = kernel(
            [weights2D, state.norms, state.indices.asType(.uint32), codebook],
            template: [
                ("Dim", dim),
                ("Bits", bits),
                ("PackedWidth", state.indices.dim(-1)),
            ],
            grid: (32, dim, B * H * R),
            threadGroup: (32, 1, 1),
            outputShapes: [[B, H, R, dim]],
            outputDTypes: [.float32]
        )[0]

        return expandedDimensions(matmul(weightedRot, rotation), axis: 3)
    }
}

private final class TurboQuantProdCodec: TurboQuantCodec {
    let dim: Int
    let bits: Int
    let mseCodec: TurboQuantMSECodec
    let projection: MLXArray
    let projectionT: MLXArray
    let queryTransformT: MLXArray
    let scale: Float
    let scaleArray: MLXArray
    let descriptor: TurboQuantCodecDescriptor

    init(dim: Int, bits: Int, seed: Int) {
        self.dim = dim
        self.bits = bits
        self.mseCodec = TurboQuantMSECodec(dim: dim, bits: max(bits - 1, 0), seed: seed)
        self.projection = TurboQuantMatrixCache.shared.projection(dim: dim, seed: seed + 1)
        self.projectionT = projection.transposed()
        if dim > 0 {
            self.queryTransformT = concatenated([mseCodec.rotationT, projectionT], axis: -1)
            self.scale = sqrt(Float.pi / 2) / Float(dim)
        } else {
            self.queryTransformT = MLXArray.zeros([0, 0], dtype: .float32)
            self.scale = 0
        }
        self.scaleArray = MLXArray([self.scale]).asType(.float32)
        self.descriptor = .init(
            kind: .prod, dim: dim, bits: Float(bits), seed: seed, lowIndices: nil,
            highIndices: nil, low: nil, high: nil)
    }

    func quantize(_ vectors: MLXArray) -> TurboQuantState {
        let vectors = vectors.asType(.float32)
        let norms = l2Norm(vectors)
        let safeNorms = maximum(norms[.ellipsis, .newAxis], MLXArray(turboQuantEpsilon))
        let unitVectors = MLX.where(
            norms[.ellipsis, .newAxis] .> 0,
            vectors / safeNorms,
            MLXArray.zeros(like: vectors)
        )

        let (mseIndices, mseUnit) = mseCodec.quantizeUnitWithEstimate(unitVectors)
        let residual = unitVectors - mseUnit
        let residualNorms = l2Norm(residual)
        let projected = matmul(residual, projectionT)
        let signs = MLX.where(
            projected .>= 0,
            MLXArray(1, dtype: .uint32),
            MLXArray(0, dtype: .uint32)
        )

        return .prod(
            .init(
                norms: norms.asType(vectors.dtype),
                mseIndices: mseIndices,
                residualNorms: residualNorms.asType(vectors.dtype),
                qjlSigns: packLowBit(signs, bits: 1)
            ))
    }

    func dequantize(_ state: TurboQuantState) -> MLXArray {
        guard case .prod(let prodState) = state else {
            fatalError("Expected TurboQuant product state.")
        }

        let mseUnit = mseCodec.dequantizeUnit(prodState.mseIndices)
        let signBits = unpackLowBit(prodState.qjlSigns, bits: 1, length: dim).asType(.float32)
        let signs = signBits * 2 - 1
        let qjlUnit =
            MLXArray(scale) * prodState.residualNorms[.ellipsis, .newAxis].asType(.float32)
            * matmul(signs, projection)
        return prodState.norms[.ellipsis, .newAxis].asType(.float32) * (mseUnit + qjlUnit)
    }

    func prepareQueries(_ queries: MLXArray) -> TurboQuantPreparedQueries {
        let transformed = matmul(queries, queryTransformT)
        return .pair(
            transformed[.ellipsis, ..<dim],
            transformed[.ellipsis, dim...]
        )
    }

    func scorePrepared(_ preparedQueries: TurboQuantPreparedQueries, state: TurboQuantState)
        -> MLXArray
    {
        guard case .pair(let mseQueries, let projectionQueries) = preparedQueries,
            case .prod(let prodState) = state
        else {
            fatalError("Expected TurboQuant product prepared queries and state.")
        }

        if projectionQueries.dim(-2) == 1,
            let fastScores = metalProdScore(
                mseQueries: mseQueries, projectionQueries: projectionQueries, state: prodState)
        {
            return fastScores
        }

        let mseScore: MLXArray
        if mseCodec.bits > 0 {
            mseScore = mseCodec.scorePrepared(
                .array(mseQueries),
                state: .mse(.init(norms: prodState.norms, indices: prodState.mseIndices))
            )
        } else {
            mseScore = MLXArray.zeros(
                [
                    projectionQueries.dim(0), projectionQueries.dim(1), projectionQueries.dim(2),
                    projectionQueries.dim(3), prodState.norms.dim(2),
                ], dtype: .float32)
        }

        if projectionQueries.dim(-2) == 1,
            let fastQJL = metalQJLScore(projectionQueries: projectionQueries, state: prodState)
        {
            return mseScore + fastQJL
        }

        let signBits = unpackLowBit(prodState.qjlSigns, bits: 1, length: dim).asType(.float32)
        let signs = signBits * 2 - 1
        let qjlScore =
            scale
            * prodState.residualNorms.asType(.float32)[0..., 0..., .newAxis, .newAxis, 0...]
            * einsum("bhmld,bhtd->bhmlt", projectionQueries, signs)
        let norms = prodState.norms.asType(.float32)[0..., 0..., .newAxis, .newAxis, 0...]
        return mseScore + norms * qjlScore
    }

    func weightedSum(_ weights: MLXArray, state: TurboQuantState) -> MLXArray {
        fatalError("TurboQuantProdCodec is not used for weighted sums.")
    }

    func weightedSumFromScores(_ scores: MLXArray, state: TurboQuantState) -> MLXArray {
        fatalError("TurboQuantProdCodec is not used for weighted sums.")
    }

    func weightedSumStatsFromScores(_ scores: MLXArray, state: TurboQuantState) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        fatalError("TurboQuantProdCodec is not used for weighted sums.")
    }

    private func metalQJLScore(projectionQueries: MLXArray, state: TurboQuantProdState)
        -> MLXArray?
    {
        guard let kernel = TurboQuantKernelManager.shared.qjlScoreKernel, state.norms.dim(2) > 0
        else {
            return nil
        }

        let squeezed = projectionQueries.squeezed(axis: 3)
        let B = squeezed.dim(0)
        let H = squeezed.dim(1)
        let R = squeezed.dim(2)
        let T = state.norms.dim(2)

        let scores = kernel(
            [
                squeezed, state.norms, state.residualNorms, state.qjlSigns.asType(.uint32),
                scaleArray,
            ],
            template: [
                ("Dim", dim),
                ("PackedWidth", state.qjlSigns.dim(-1)),
            ],
            grid: (32, R, B * H * T),
            threadGroup: (32, 1, 1),
            outputShapes: [[B, H, R, T]],
            outputDTypes: [.float32]
        )[0]

        return expandedDimensions(scores, axis: 3)
    }

    private func metalProdScore(
        mseQueries: MLXArray,
        projectionQueries: MLXArray,
        state: TurboQuantProdState
    ) -> MLXArray? {
        guard let kernel = TurboQuantKernelManager.shared.prodScoreKernel, state.norms.dim(2) > 0
        else {
            return nil
        }

        let mseQueries = mseQueries.squeezed(axis: 3)
        let projectionQueries = projectionQueries.squeezed(axis: 3)
        let B = mseQueries.dim(0)
        let H = mseQueries.dim(1)
        let R = mseQueries.dim(2)
        let T = state.norms.dim(2)

        let scores = kernel(
            [
                mseQueries,
                projectionQueries,
                state.norms,
                state.residualNorms,
                state.mseIndices.asType(.uint32),
                state.qjlSigns.asType(.uint32),
                mseCodec.codebook,
                scaleArray,
            ],
            template: [
                ("Dim", dim),
                ("MseBits", mseCodec.bits),
                ("MsePackedWidth", state.mseIndices.dim(-1)),
                ("SignPackedWidth", state.qjlSigns.dim(-1)),
            ],
            grid: (32, R, B * H * T),
            threadGroup: (32, 1, 1),
            outputShapes: [[B, H, R, T]],
            outputDTypes: [.float32]
        )[0]

        return expandedDimensions(scores, axis: 3)
    }
}

private func selectOutlierIndices(_ tensor: MLXArray, averageBits: Float) -> ([Int], [Int]) {
    let lowerBits = Int(floor(averageBits))
    let upperBits = Int(ceil(averageBits))
    precondition(lowerBits != upperBits, "Split selection requires a fractional bit width.")

    let dim = tensor.dim(-1)
    let highCount = max(
        1,
        min(
            dim - 1,
            Int(round((averageBits - Float(lowerBits)) * Float(dim) / Float(upperBits - lowerBits)))
        )
    )

    let scores = abs(tensor.asType(.float32)).mean(axes: [0, 1, 2]).asArray(Float.self)
    let sortedIndices = scores.enumerated().sorted { $0.element < $1.element }.map(\.offset)
    let highIndices = Array(sortedIndices.suffix(highCount)).sorted()
    let highSet = Set(highIndices)
    let lowIndices = (0 ..< dim).filter { !highSet.contains($0) }
    return (lowIndices, highIndices)
}

private final class TurboQuantSplitCodec: TurboQuantCodec {
    let dim: Int
    let bits: Float
    let seed: Int
    let lowIndices: MLXArray
    let highIndices: MLXArray
    let restoreOrder: MLXArray
    let lowCodec: any TurboQuantCodec
    let highCodec: any TurboQuantCodec
    let descriptor: TurboQuantCodecDescriptor

    convenience init(
        tensor: MLXArray,
        bits: Float,
        mode: TurboQuantCodecDescriptor.Kind,
        seed: Int
    ) {
        let (lowIndices, highIndices) = selectOutlierIndices(tensor, averageBits: bits)
        let lowTensor = take(tensor, MLXArray(lowIndices).asType(.int32), axis: -1)
        let highTensor = take(tensor, MLXArray(highIndices).asType(.int32), axis: -1)
        let lowCodec = buildTurboQuantCodec(
            tensor: lowTensor, bits: Float(floor(bits)), mode: mode, seed: seed)
        let highCodec = buildTurboQuantCodec(
            tensor: highTensor, bits: Float(ceil(bits)), mode: mode, seed: seed + 97)
        self.init(
            bits: bits,
            seed: seed,
            lowIndices: lowIndices,
            highIndices: highIndices,
            lowCodec: lowCodec,
            highCodec: highCodec
        )
    }

    init(
        bits: Float,
        seed: Int,
        lowIndices: [Int],
        highIndices: [Int],
        lowCodec: any TurboQuantCodec,
        highCodec: any TurboQuantCodec
    ) {
        self.bits = bits
        self.seed = seed
        self.lowIndices = MLXArray(lowIndices).asType(.int32)
        self.highIndices = MLXArray(highIndices).asType(.int32)
        let concatenatedIndices = lowIndices + highIndices
        let restoreOrder = concatenatedIndices.enumerated().sorted { $0.element < $1.element }.map(
            \.offset)
        self.restoreOrder = MLXArray(restoreOrder).asType(.int32)
        self.lowCodec = lowCodec
        self.highCodec = highCodec
        self.dim = lowIndices.count + highIndices.count
        self.descriptor = .init(
            kind: .split,
            dim: dim,
            bits: bits,
            seed: seed,
            lowIndices: lowIndices,
            highIndices: highIndices,
            low: lowCodec.descriptor,
            high: highCodec.descriptor
        )
    }

    func quantize(_ vectors: MLXArray) -> TurboQuantState {
        let lowTensor = take(vectors, lowIndices, axis: -1)
        let highTensor = take(vectors, highIndices, axis: -1)
        return .split(
            .init(
                low: lowCodec.quantize(lowTensor),
                high: highCodec.quantize(highTensor)
            ))
    }

    func dequantize(_ state: TurboQuantState) -> MLXArray {
        guard case .split(let splitState) = state else {
            fatalError("Expected TurboQuant split state.")
        }
        let lowTensor = lowCodec.dequantize(splitState.low)
        let highTensor = highCodec.dequantize(splitState.high)
        let merged = concatenated([lowTensor, highTensor], axis: -1)
        return take(merged, restoreOrder, axis: -1)
    }

    func prepareQueries(_ queries: MLXArray) -> TurboQuantPreparedQueries {
        let lowTensor = take(queries, lowIndices, axis: -1)
        let highTensor = take(queries, highIndices, axis: -1)
        return .split(lowCodec.prepareQueries(lowTensor), highCodec.prepareQueries(highTensor))
    }

    func scorePrepared(_ preparedQueries: TurboQuantPreparedQueries, state: TurboQuantState)
        -> MLXArray
    {
        guard case .split(let lowQueries, let highQueries) = preparedQueries,
            case .split(let splitState) = state
        else {
            fatalError("Expected TurboQuant split prepared queries and state.")
        }
        return lowCodec.scorePrepared(lowQueries, state: splitState.low)
            + highCodec.scorePrepared(highQueries, state: splitState.high)
    }

    func weightedSum(_ weights: MLXArray, state: TurboQuantState) -> MLXArray {
        guard case .split(let splitState) = state else {
            fatalError("Expected TurboQuant split state.")
        }
        let lowTensor = lowCodec.weightedSum(weights, state: splitState.low)
        let highTensor = highCodec.weightedSum(weights, state: splitState.high)
        let merged = concatenated([lowTensor, highTensor], axis: -1)
        return take(merged, restoreOrder, axis: -1)
    }

    func weightedSumFromScores(_ scores: MLXArray, state: TurboQuantState) -> MLXArray {
        guard case .split(let splitState) = state else {
            fatalError("Expected TurboQuant split state.")
        }
        let lowTensor = lowCodec.weightedSumFromScores(scores, state: splitState.low)
        let highTensor = highCodec.weightedSumFromScores(scores, state: splitState.high)
        let merged = concatenated([lowTensor, highTensor], axis: -1)
        return take(merged, restoreOrder, axis: -1)
    }

    func weightedSumStatsFromScores(_ scores: MLXArray, state: TurboQuantState) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        guard case .split(let splitState) = state else {
            fatalError("Expected TurboQuant split state.")
        }
        let (lowTensor, denominator, maxScores) = lowCodec.weightedSumStatsFromScores(
            scores, state: splitState.low)
        let (highTensor, _, _) = highCodec.weightedSumStatsFromScores(
            scores, state: splitState.high)
        let merged = concatenated([lowTensor, highTensor], axis: -1)
        return (take(merged, restoreOrder, axis: -1), denominator, maxScores)
    }
}

public final class TurboQuantKVCache: BaseKVCache {
    private enum Constants {
        static let decodeKeyChunkSize = 65_536
        static let prefillKeyChunkSize = 512
        static let prefillQueryBlockSize = 16
        static let cacheStep = 256
    }

    private var keyState: TurboQuantState?
    private var valueState: TurboQuantState?
    private var keyCodec: (any TurboQuantCodec)?
    private var valueCodec: (any TurboQuantCodec)?

    public private(set) var bits: Float
    public private(set) var seed: Int

    public init(bits: Float = 4.0, seed: Int = 0) {
        self.bits = validateTurboQuantBits(bits)
        self.seed = seed
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    private func updateTurboQuant(keys: MLXArray, values: MLXArray) -> (
        TurboQuantState, TurboQuantState
    ) {
        ensureCodecs(keys: keys, values: values)
        guard let keyCodec, let valueCodec else {
            fatalError("TurboQuant codecs were not initialized.")
        }

        let newKeyState = keyCodec.quantize(keys)
        let newValueState = valueCodec.quantize(values)
        let newEnd = offset + keys.dim(2)

        if let existingKeyState = keyState, let existingValueState = valueState {
            keyState = reserveTurboQuantStateCapacity(
                existingKeyState, used: offset, needed: newEnd, step: Constants.cacheStep)
            valueState = reserveTurboQuantStateCapacity(
                existingValueState, used: offset, needed: newEnd, step: Constants.cacheStep)
        } else {
            keyState = newKeyState.allocateLike(length: newEnd)
            valueState = newValueState.allocateLike(length: newEnd)
        }

        if var keyState {
            keyState.write(newKeyState, start: offset)
            self.keyState = keyState
        }
        if var valueState {
            valueState.write(newValueState, start: offset)
            self.valueState = valueState
        }
        offset = newEnd

        guard let keyState, let valueState else {
            fatalError("TurboQuant cache write failed.")
        }
        return (keyState.slice(end: offset), valueState.slice(end: offset))
    }

    private func dequantize(
        keysState: TurboQuantState? = nil,
        valuesState: TurboQuantState? = nil
    ) -> (MLXArray, MLXArray) {
        let keysState = keysState ?? keyState?.slice(end: offset)
        let valuesState = valuesState ?? valueState?.slice(end: offset)
        guard let keysState, let valuesState, let keyCodec, let valueCodec else {
            return (
                MLXArray.zeros([0], dtype: .float32),
                MLXArray.zeros([0], dtype: .float32)
            )
        }

        let keys = keyCodec.dequantize(keysState).asType(.float32)
        let values = valueCodec.dequantize(valuesState).asType(.float32)
        return (keys, values)
    }

    public func dequantizedState() -> (MLXArray, MLXArray)? {
        guard keyState != nil, valueState != nil else {
            return nil
        }
        return dequantize()
    }

    private func quantizedAttention(
        queries: MLXArray,
        keysState: TurboQuantState? = nil,
        valuesState: TurboQuantState? = nil,
        scale: Float = 1.0,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let keysState = keysState ?? keyState?.slice(end: offset)
        let valuesState = valuesState ?? valueState?.slice(end: offset)
        guard let keysState, let valuesState, let keyCodec, let valueCodec else {
            fatalError("TurboQuant cache is empty.")
        }

        let B = queries.dim(0)
        let qHeads = queries.dim(1)
        let L = queries.dim(2)
        let D = queries.dim(3)
        let kvHeads = inferredKVHeads(from: keysState)
        let repeats = qHeads / kvHeads
        let groupedQueries = (queries * scale).reshaped([B, kvHeads, repeats, L, D])
        let valueDim = valueCodec.dim
        let totalTokens = keysState.length

        var outputs: [MLXArray] = []
        for queryStart in stride(from: 0, to: L, by: Constants.prefillQueryBlockSize) {
            let queryEnd = min(L, queryStart + Constants.prefillQueryBlockSize)
            let queryBlock = groupedQueries[.ellipsis, queryStart ..< queryEnd, 0...]
            let preparedQueries = keyCodec.prepareQueries(queryBlock)

            var output = MLXArray.zeros(
                [B, kvHeads, repeats, queryEnd - queryStart, valueDim],
                dtype: .float32
            )
            var normalizer = MLXArray.zeros(
                [B, kvHeads, repeats, queryEnd - queryStart],
                dtype: .float32
            )
            var maxScore = full(
                [B, kvHeads, repeats, queryEnd - queryStart],
                values: -Float.infinity
            ).asType(.float32)

            for keyStart in stride(from: 0, to: totalTokens, by: Constants.prefillKeyChunkSize) {
                let keyEnd = min(totalTokens, keyStart + Constants.prefillKeyChunkSize)
                let keyChunk = keysState.slice(range: keyStart ..< keyEnd)
                let valueChunk = valuesState.slice(range: keyStart ..< keyEnd)

                var scores = keyCodec.scorePrepared(preparedQueries, state: keyChunk)
                scores = applyAttentionMask(
                    scores,
                    mask: mask,
                    queryStart: queryStart,
                    queryEnd: queryEnd,
                    keyStart: keyStart,
                    keyEnd: keyEnd,
                    totalQueries: L,
                    totalTokens: totalTokens
                )

                let (chunkOutput, chunkDenominator, chunkMax) =
                    valueCodec
                    .weightedSumStatsFromScores(scores, state: valueChunk)
                let newMax = maximum(maxScore, chunkMax)
                let previousScale = exp(maxScore - newMax)
                let chunkScale = exp(chunkMax - newMax)

                output =
                    output * previousScale[.ellipsis, .newAxis]
                    + chunkOutput * chunkScale[.ellipsis, .newAxis]
                normalizer = normalizer * previousScale + chunkDenominator * chunkScale
                maxScore = newMax
                eval(output, normalizer, maxScore)
            }

            let normalized =
                output / maximum(normalizer[.ellipsis, .newAxis], MLXArray(turboQuantEpsilon))
            outputs.append(normalized)
            eval(normalized)
        }

        let output = concatenated(outputs, axis: 3).reshaped([B, qHeads, L, valueDim])
        return output.asType(queries.dtype)
    }

    private func decodeAttention(
        queries: MLXArray,
        keysState: TurboQuantState? = nil,
        valuesState: TurboQuantState? = nil,
        scale: Float = 1.0,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        precondition(
            queries.dim(2) == 1, "TurboQuant decode attention expects a single query token.")

        let keysState = keysState ?? keyState?.slice(end: offset)
        let valuesState = valuesState ?? valueState?.slice(end: offset)
        guard let keysState, let valuesState, let keyCodec, let valueCodec else {
            fatalError("TurboQuant cache is empty.")
        }

        let B = queries.dim(0)
        let qHeads = queries.dim(1)
        let L = queries.dim(2)
        let D = queries.dim(3)
        let kvHeads = inferredKVHeads(from: keysState)
        let repeats = qHeads / kvHeads
        let groupedQueries = (queries * scale).reshaped([B, kvHeads, repeats, L, D])
        let totalTokens = keysState.length

        let preparedQueries = keyCodec.prepareQueries(groupedQueries)

        let noMask: Bool
        if case .none = mask {
            noMask = true
        } else {
            noMask = false
        }

        if totalTokens <= Constants.decodeKeyChunkSize && noMask {
            let scores = keyCodec.scorePrepared(preparedQueries, state: keysState)
            let output = valueCodec.weightedSumFromScores(scores, state: valuesState)
            return output.reshaped([B, qHeads, L, valueCodec.dim]).asType(queries.dtype)
        }

        var output = MLXArray.zeros([B, kvHeads, repeats, L, valueCodec.dim], dtype: .float32)
        var normalizer = MLXArray.zeros([B, kvHeads, repeats, L], dtype: .float32)
        var maxScore = full([B, kvHeads, repeats, L], values: -Float.infinity).asType(.float32)

        for keyStart in stride(from: 0, to: totalTokens, by: Constants.decodeKeyChunkSize) {
            let keyEnd = min(totalTokens, keyStart + Constants.decodeKeyChunkSize)
            let keyChunk = keysState.slice(range: keyStart ..< keyEnd)
            let valueChunk = valuesState.slice(range: keyStart ..< keyEnd)

            var scores = keyCodec.scorePrepared(preparedQueries, state: keyChunk)
            scores = applyAttentionMask(
                scores,
                mask: mask,
                queryStart: 0,
                queryEnd: L,
                keyStart: keyStart,
                keyEnd: keyEnd,
                totalQueries: L,
                totalTokens: totalTokens
            )

            let (chunkOutput, chunkDenominator, chunkMax) = valueCodec.weightedSumStatsFromScores(
                scores, state: valueChunk)
            let newMax = maximum(maxScore, chunkMax)
            let previousScale = exp(maxScore - newMax)
            let chunkScale = exp(chunkMax - newMax)

            output =
                output * previousScale[.ellipsis, .newAxis]
                + chunkOutput * chunkScale[.ellipsis, .newAxis]
            normalizer = normalizer * previousScale + chunkDenominator * chunkScale
            maxScore = newMax
        }

        let normalized =
            output / maximum(normalizer[.ellipsis, .newAxis], MLXArray(turboQuantEpsilon))
        return normalized.reshaped([B, qHeads, L, valueCodec.dim]).asType(queries.dtype)
    }

    func attention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float = 1.0,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let (keysState, valuesState) = updateTurboQuant(keys: keys, values: values)
        if queries.dim(2) == 1 {
            return decodeAttention(
                queries: queries,
                keysState: keysState,
                valuesState: valuesState,
                scale: scale,
                mask: mask
            )
        }

        return quantizedAttention(
            queries: queries,
            keysState: keysState,
            valuesState: valuesState,
            scale: scale,
            mask: mask
        )
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let (keysState, valuesState) = updateTurboQuant(keys: keys, values: values)
        return dequantize(keysState: keysState, valuesState: valuesState)
    }

    public override var state: [MLXArray] {
        get {
            guard let keyState, let valueState else { return [] }
            return flattenTurboQuantState(keyState.slice(end: offset))
                + flattenTurboQuantState(valueState.slice(end: offset))
        }
        set {
            guard !newValue.isEmpty else {
                keyState = nil
                valueState = nil
                offset = 0
                return
            }
            guard let keyCodec, let valueCodec else {
                fatalError("TurboQuant codecs must be restored before setting state.")
            }

            let (newKeyState, keyCount) = unflattenTurboQuantState(
                newValue[...], descriptor: keyCodec.descriptor)
            let (newValueState, _) = unflattenTurboQuantState(
                newValue[keyCount...], descriptor: valueCodec.descriptor)
            keyState = newKeyState
            valueState = newValueState
            offset = newKeyState.length
        }
    }

    public override var metaState: [String] {
        get {
            let metadata = TurboQuantCacheMetadata(
                offset: offset,
                bits: bits,
                seed: seed,
                keyCodec: keyCodec?.descriptor,
                valueCodec: valueCodec?.descriptor
            )
            let encoded = try? JSONEncoder().encode(metadata)
            return [String(data: encoded ?? Data(), encoding: .utf8) ?? "{}"]
        }
        set {
            guard let value = newValue.first, let data = value.data(using: .utf8),
                let metadata = try? JSONDecoder().decode(TurboQuantCacheMetadata.self, from: data)
            else {
                fatalError("TurboQuantKVCache metaState must contain valid JSON metadata.")
            }

            self.offset = metadata.offset
            self.bits = validateTurboQuantBits(metadata.bits)
            self.seed = metadata.seed
            self.keyCodec = metadata.keyCodec.map(rebuildTurboQuantCodec)
            self.valueCodec = metadata.valueCodec.map(rebuildTurboQuantCodec)
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public var nbytes: Int {
        (keyState?.slice(end: offset).nbytes ?? 0) + (valueState?.slice(end: offset).nbytes ?? 0)
    }

    public static func fromCache(
        _ cache: KVCache,
        bits: Float,
        seed: Int = 0
    ) -> TurboQuantKVCache {
        let turboCache = TurboQuantKVCache(bits: bits, seed: seed)
        let existingState = cache.state
        if existingState.count >= 2 {
            _ = turboCache.updateTurboQuant(keys: existingState[0], values: existingState[1])
        }
        return turboCache
    }

    private func ensureCodecs(keys: MLXArray, values: MLXArray) {
        if keyCodec == nil {
            keyCodec = buildTurboQuantCodec(tensor: keys, bits: bits, mode: .prod, seed: seed)
        }
        if valueCodec == nil {
            valueCodec = buildTurboQuantCodec(
                tensor: values,
                bits: bits,
                mode: .mse,
                seed: seed + 1
            )
        }
    }

    private func inferredKVHeads(from state: TurboQuantState) -> Int {
        switch state {
        case .mse(let state):
            return state.norms.dim(1)
        case .prod(let state):
            return state.norms.dim(1)
        case .split(let state):
            return inferredKVHeads(from: state.low)
        }
    }

    private func applyAttentionMask(
        _ scores: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        queryStart: Int,
        queryEnd: Int,
        keyStart: Int,
        keyEnd: Int,
        totalQueries: Int,
        totalTokens: Int
    ) -> MLXArray {
        switch mask {
        case .none:
            return scores
        case .causal:
            let pastTokens = totalTokens - totalQueries
            let queryIndices = MLXArray(
                Int32(pastTokens + queryStart) ..< Int32(pastTokens + queryEnd))
            let keyIndices = MLXArray(Int32(keyStart) ..< Int32(keyEnd))
            var causalMask = queryIndices[0..., .newAxis] .>= keyIndices[.newAxis]
            causalMask = expandedDimensions(causalMask, axes: [0, 1, 2])
            return MLX.where(
                causalMask,
                scores,
                MLXArray(-Float.infinity, dtype: scores.dtype)
            )
        case .array(let maskArray):
            var maskChunk = maskArray[.ellipsis, queryStart ..< queryEnd, keyStart ..< keyEnd]
            if maskChunk.ndim == scores.ndim - 1 {
                maskChunk = expandedDimensions(maskChunk, axis: 2)
            }
            if maskChunk.dtype == .bool {
                return MLX.where(
                    maskChunk,
                    scores,
                    MLXArray(-Float.infinity, dtype: scores.dtype)
                )
            }
            return scores + maskChunk
        case .arrays(let maskArrays):
            guard let firstMask = maskArrays.first else { return scores }
            return applyAttentionMask(
                scores,
                mask: .array(firstMask),
                queryStart: queryStart,
                queryEnd: queryEnd,
                keyStart: keyStart,
                keyEnd: keyEnd,
                totalQueries: totalQueries,
                totalTokens: totalTokens
            )
        }
    }
}
