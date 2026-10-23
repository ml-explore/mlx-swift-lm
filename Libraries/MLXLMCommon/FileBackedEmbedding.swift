// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

#if canImport(Darwin)
import Cmlx
import Darwin
import os
#endif

/// Storage for embedding tables during model loading.
public enum EmbeddingStoragePolicy: Sendable, Equatable {
    /// File-back supported frozen input tables on Apple platforms; other tables stay resident.
    /// Keep checkpoint contents unchanged while the model is in use.
    case automatic
    /// Keep all tables in normal MLX storage.
    case resident
}

/// Embedding tables used only for lookup, never shared with a dense output projection.
public protocol InputOnlyEmbeddingsProviding: Module {
    var inputOnlyEmbeddings: [Embedding] { get }
}

#if canImport(Darwin)
private let embeddingStorageLogger = Logger(
    subsystem: "mlx-swift-lm", category: "embedding-storage")
#endif

package func prepareEmbeddingStorage(
    in model: BaseLanguageModel, policy: EmbeddingStoragePolicy,
    checkpoint: CheckpointTensorStorage? = nil
) -> Int {
    #if canImport(Darwin)
    do {
        return try prepareEmbeddingStorage(
            in: model, policy: policy, directory: FileManager.default.temporaryDirectory,
            checkpoint: checkpoint)
    } catch {
        embeddingStorageLogger.warning(
            "Could not prepare embedding storage: \(String(describing: error))")
        return 0
    }
    #else
    return 0
    #endif
}

#if canImport(Darwin)

/// Copy one array into page-aligned temporary storage. The mapping owns its disk blocks.
package func fileBackedArray(_ source: MLXArray, directory: URL) throws -> MLXArray {
    try withExtendedLifetime(source) {
        let page = Int(getpagesize())
        let (padded, overflow) = source.nbytes.addingReportingOverflow(page - 1)
        guard source.nbytes > 0, !overflow else { throw POSIXError(.EINVAL) }
        let pageCount = padded / page
        let length = pageCount * page
        guard length / source.itemSize <= Int32.max,
            length <= GPU.deviceInfo().maxBufferSize
        else { throw POSIXError(.EOVERFLOW) }
        var path = Array(directory.appendingPathComponent("mlx-embedding-XXXXXX").path.utf8CString)
        let descriptor = mkstemp(&path)
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { close(descriptor) }
        guard unlink(path) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        // Do not fill the file cache with rows that inference may never read.
        _ = fcntl(descriptor, F_NOCACHE, 1)
        guard ftruncate(descriptor, off_t(length)) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        let data = source.asData(access: .noCopyIfContiguous).data
        try data.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { throw POSIXError(.EINVAL) }
            var position = 0
            while position < bytes.count {
                let count = pwrite(
                    descriptor, base.advanced(by: position),
                    min(bytes.count - position, 8 * 1024 * 1024),
                    off_t(position)
                )
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
                position += count
            }
        }
        guard let pointer = mmap(nil, length, PROT_READ | PROT_WRITE, MAP_PRIVATE, descriptor, 0),
            pointer != MAP_FAILED
        else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        // Sparse lookups do not need read-ahead for neighboring rows.
        _ = madvise(pointer, length, MADV_RANDOM)
        let storage = MLXArray(
            rawPointer: pointer, [pageCount, page / source.itemSize], dtype: source.dtype
        ) {
            munmap(pointer, length)
        }
        guard
            mlx_array_data_uint8(storage.ctx).map(UnsafeRawPointer.init)
                == UnsafeRawPointer(pointer)
        else { throw POSIXError(.ENOTSUP) }
        let array = asStrided(storage, source.shape, stream: .cpu)
        eval(array)
        return array
    }
}

package final class FileBackedEmbedding: Embedding {
    package override func callAsFunction(_ x: MLXArray) -> MLXArray {
        weight.take(x, axis: 0, stream: .cpu)
    }
}

package final class FileBackedQuantizedEmbedding: QuantizedEmbedding {
    package override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let indices = x.flattened()
        return dequantized(
            weight.take(indices, axis: 0, stream: .cpu),
            scales: scales.take(indices, axis: 0, stream: .cpu),
            biases: biases.map { $0.take(indices, axis: 0, stream: .cpu) },
            groupSize: groupSize, bits: bits, mode: mode
        ).reshaped(x.shape + [-1])
    }
}

func supportsFileBackedStorage(_ embedding: Embedding) -> Bool {
    guard embedding.trainableParameters().flattened().isEmpty else { return false }
    if let quantized = embedding as? QuantizedEmbedding {
        guard
            type(of: embedding) == QuantizedEmbedding.self
                || embedding is FileBackedQuantizedEmbedding
        else { return false }
        return quantized.mode == .affine && quantized.globalScale == nil
    }
    return type(of: embedding) == Embedding.self || embedding is FileBackedEmbedding
}

/// Prepare embedding access after checkpoint parameters have been applied.
@discardableResult
package func prepareEmbeddingStorage(
    in model: Module, policy: EmbeddingStoragePolicy = .automatic, directory: URL,
    checkpoint: CheckpointTensorStorage? = nil
) throws -> Int {
    guard let provider = model as? any InputOnlyEmbeddingsProviding else { return 0 }
    let eligible = Set(provider.inputOnlyEmbeddings.map(ObjectIdentifier.init))
    guard !eligible.isEmpty else { return 0 }
    var replacements: [(String, Module)] = []
    var prepared: [ObjectIdentifier: Embedding] = [:]
    for (name, module) in model.namedModules() {
        guard !name.isEmpty, eligible.contains(ObjectIdentifier(module)),
            let embedding = module as? Embedding, supportsFileBackedStorage(embedding),
            policy == .automatic || module is FileBackedEmbedding
                || module is FileBackedQuantizedEmbedding
        else { continue }
        let identity = ObjectIdentifier(embedding)
        if let replacement = prepared[identity] {
            replacements.append((name, replacement))
            continue
        }
        func backed(_ array: MLXArray) throws -> MLXArray {
            if let mapped = try? checkpoint?.array(matching: array) { return mapped }
            return try fileBackedArray(array, directory: directory)
        }
        let replacement: Embedding
        if let quantized = embedding as? QuantizedEmbedding {
            switch policy {
            case .automatic:
                replacement = try FileBackedQuantizedEmbedding(
                    weight: backed(quantized.weight), scales: backed(quantized.scales),
                    biases: quantized.biases.map(backed),
                    groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode,
                    globalScale: quantized.globalScale)
            case .resident:
                replacement = QuantizedEmbedding(
                    weight: quantized.weight, scales: quantized.scales, biases: quantized.biases,
                    groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode,
                    globalScale: quantized.globalScale)
            }
        } else {
            switch policy {
            case .automatic:
                replacement = try FileBackedEmbedding(weight: backed(embedding.weight))
            case .resident:
                replacement = Embedding(weight: embedding.weight)
            }
        }
        replacement.freeze()
        replacement.train(embedding.training)
        prepared[identity] = replacement
        replacements.append((name, replacement))
    }
    if !replacements.isEmpty {
        defer { model.invalidateCompiledTraces() }
        try model.update(modules: ModuleChildren.unflattened(replacements), verify: [])
    }
    return prepared.count
}

#endif
