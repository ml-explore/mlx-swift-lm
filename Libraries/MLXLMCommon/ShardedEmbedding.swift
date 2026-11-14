// Copyright © 2026 Apple Inc.

import MLX
import MLXNN

/// A lookup table split by rows into consecutive shards.
///
/// Checkpoints store some very large tables this way, one tensor per shard. Each shard
/// is a normal `Embedding`, so a model can list ``shards`` in
/// ``InputOnlyEmbeddingsProviding/inputOnlyEmbeddings`` to keep them in file-backed storage.
public final class ShardedEmbedding: Module, UnaryLayer {
    @ModuleInfo(key: "shards") public var shards: [Embedding]

    public init(shards: [Embedding]) {
        precondition(!shards.isEmpty, "ShardedEmbedding needs at least one shard")
        _shards.wrappedValue = shards
        super.init()
    }

    public convenience init(rowCounts: [Int], dimensions: Int) {
        self.init(shards: rowCounts.map { Embedding(embeddingCount: $0, dimensions: dimensions) })
    }

    /// Looks up rows by their index in the whole table. Indices must be in `0 ..< rowCount`.
    ///
    /// Each shard is read once per call, so the cost grows with shards times indices.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var starts: [Int32] = []
        var rowCount = 0
        for shard in shards {
            starts.append(Int32(rowCount))
            rowCount += shard.weight.dim(0)
        }
        precondition(rowCount <= Int(Int32.max), "ShardedEmbedding has too many rows")
        let indices = x.flattened().asType(.int32)
        let startArray = MLXArray(starts)
        let shardIndices =
            (indices.expandedDimensions(axis: -1) .>= startArray).asType(.int32).sum(axis: -1) - 1
        let localIndices = indices - startArray.take(shardIndices, axis: 0)

        // Rows outside a shard clamp to its first or last row, then are masked out.
        func select(stream: StreamOrDevice, _ rows: (Int, MLXArray) -> MLXArray) -> MLXArray {
            var result: MLXArray?
            for (index, shard) in shards.enumerated() {
                let local = clip(localIndices, min: 0, max: shard.weight.dim(0) - 1, stream: stream)
                let selected = rows(index, local)
                if let current = result {
                    let mask = (shardIndices .== Int32(index)).reshaped(
                        [-1] + Array(repeating: 1, count: selected.ndim - 1))
                    result = which(mask, selected, current, stream: stream)
                } else {
                    result = selected
                }
            }
            return result!
        }

        let output: MLXArray
        if let quantized = uniformQuantizedShards() {
            // Gather packed rows from every shard, then dequantize once.
            let stream = gathersOnCPU ? StreamOrDevice.cpu : .default
            let first = quantized[0]
            let weight = select(stream: stream) {
                quantized[$0].weight.take($1, axis: 0, stream: stream)
            }
            let scales = select(stream: stream) {
                quantized[$0].scales.take($1, axis: 0, stream: stream)
            }
            let biases =
                first.biases == nil
                ? nil
                : select(stream: stream) {
                    quantized[$0].biases!.take($1, axis: 0, stream: stream)
                }
            output = dequantized(
                weight, scales: scales, biases: biases, groupSize: first.groupSize,
                bits: first.bits, mode: first.mode)
        } else {
            output = select(stream: .default) { shards[$0]($1) }
        }
        return output.reshaped(x.shape + [-1])
    }

    private func uniformQuantizedShards() -> [QuantizedEmbedding]? {
        let quantized = shards.compactMap { $0 as? QuantizedEmbedding }
        guard quantized.count == shards.count, let first = quantized.first else { return nil }
        let uniform = quantized.allSatisfy {
            $0.groupSize == first.groupSize && $0.bits == first.bits && $0.mode == first.mode
                && $0.globalScale == nil && ($0.biases == nil) == (first.biases == nil)
        }
        return uniform ? quantized : nil
    }

    private var gathersOnCPU: Bool {
        #if canImport(Darwin)
        return shards.allSatisfy { $0 is FileBackedQuantizedEmbedding }
        #else
        return false
        #endif
    }
}
