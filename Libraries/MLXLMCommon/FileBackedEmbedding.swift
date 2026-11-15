// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

#if canImport(Darwin)
import Cmlx
import Darwin
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
private let embeddingStorageLogger = MLXLogger(label: "embedding-storage")
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

/// The arrays of the file-backed embedding tables in `module`.
func fileBackedEmbeddingArrays(in module: Module) -> [MLXArray] {
    #if canImport(Darwin)
    return module.namedModules().flatMap { _, module -> [MLXArray] in
        guard module is FileBackedEmbedding || module is FileBackedQuantizedEmbedding else {
            return []
        }
        return module.parameters().flattened().map(\.1)
    }
    #else
    return []
    #endif
}

#if canImport(Darwin)

/// Copy one array into page-aligned temporary storage. The mapping owns its disk blocks.
package func fileBackedArray(_ source: MLXArray, directory: URL) throws -> MLXArray {
    try withExtendedLifetime(source) {
        try temporaryFileArray(like: source, directory: directory) { descriptor in
            let data = source.asData(access: .noCopyIfContiguous).data
            try data.withUnsafeBytes { bytes in
                guard let base = bytes.baseAddress else { throw POSIXError(.EINVAL) }
                try transfer(bytes.count) {
                    pwrite(descriptor, base + $0, min(bytes.count - $0, chunkSize), off_t($0))
                }
            }
        }
    }
}

/// Copy the bytes of an array shaped like `template` at `offset` in `descriptor` into
/// page-aligned temporary storage, one chunk at a time.
func fileBackedArray(
    like template: MLXArray, copying descriptor: Int32, offset: Int64, directory: URL
) throws -> MLXArray {
    try temporaryFileArray(like: template, directory: directory) {
        try copyFileBytes(template.nbytes, from: descriptor, at: offset, to: $0)
    }
}

/// Map the bytes of an array shaped like `template` at `offset` in `descriptor`. The offset
/// must be a multiple of the element size. The array owns the private mapping.
func fileBackedArray(
    like template: MLXArray, mapping descriptor: Int32, offset: Int64
) throws -> MLXArray {
    let page = Int(getpagesize())
    let leading = Int(offset % Int64(page))
    let length = try mappingLength(of: template, leading: leading)
    guard
        let pointer = mmap(
            nil, length, PROT_READ | PROT_WRITE, MAP_PRIVATE, descriptor, offset - Int64(leading)),
        pointer != MAP_FAILED
    else { throw POSIXError.lastError }
    // Sparse lookups do not need read-ahead for neighboring rows.
    _ = madvise(pointer, length, MADV_RANDOM)
    let storage = MLXArray(
        rawPointer: pointer, [length / page, page / template.itemSize], dtype: template.dtype
    ) {
        munmap(pointer, length)
    }
    guard
        mlx_array_data_uint8(storage.ctx).map(UnsafeRawPointer.init)
            == UnsafeRawPointer(pointer)
    else { throw POSIXError(.ENOTSUP) }
    let array = asStrided(
        storage, template.shape, offset: leading / template.itemSize, stream: .cpu)
    eval(array)
    return array
}

// Bytes per read or write. Smaller chunks made cold loads measurably slower.
private let chunkSize = 32 * 1024 * 1024

/// The page-rounded length of a mapping that holds `template` after `leading` bytes.
private func mappingLength(of template: MLXArray, leading: Int) throws -> Int {
    guard template.nbytes > 0 else { throw POSIXError(.EINVAL) }
    let page = Int(getpagesize())
    let (extent, overflow) = template.nbytes.addingReportingOverflow(leading + page - 1)
    let length = extent / page * page
    guard !overflow, length / template.itemSize <= Int32.max,
        length <= GPU.deviceInfo().maxBufferSize
    else { throw POSIXError(.EOVERFLOW) }
    return length
}

/// An array shaped like `template` in an unlinked temporary file that `fill` writes.
private func temporaryFileArray(
    like template: MLXArray, directory: URL, fill: (Int32) throws -> Void
) throws -> MLXArray {
    let length = try mappingLength(of: template, leading: 0)
    var path = Array(directory.appendingPathComponent("mlx-embedding-XXXXXX").path.utf8CString)
    let descriptor = mkstemp(&path)
    guard descriptor >= 0 else { throw POSIXError.lastError }
    defer { close(descriptor) }
    guard unlink(path) == 0 else { throw POSIXError.lastError }
    // Do not fill the file cache with rows that inference may never read.
    _ = fcntl(descriptor, F_NOCACHE, 1)
    guard ftruncate(descriptor, off_t(length)) == 0 else { throw POSIXError.lastError }
    try fill(descriptor)
    return try fileBackedArray(like: template, mapping: descriptor, offset: 0)
}

/// Copy `count` bytes at `offset` in `source` to the start of `destination`.
private func copyFileBytes(
    _ count: Int, from source: Int32, at offset: Int64, to destination: Int32
) throws {
    // Page alignment lets the writes bypass the file cache.
    let buffer = UnsafeMutableRawPointer.allocate(
        byteCount: min(chunkSize, count), alignment: Int(getpagesize()))
    defer { buffer.deallocate() }
    var copied = 0
    while copied < count {
        let length = min(chunkSize, count - copied)
        try transfer(length) {
            pread(source, buffer + $0, length - $0, offset + Int64(copied + $0))
        }
        try transfer(length) {
            pwrite(destination, buffer + $0, length - $0, off_t(copied + $0))
        }
        copied += length
    }
}

/// Call `step` with the bytes done so far until `count` bytes are done.
private func transfer(_ count: Int, _ step: (Int) -> Int) throws {
    var done = 0
    while done < count {
        let result = step(done)
        if result < 0 && errno == EINTR { continue }
        guard result > 0 else { throw result < 0 ? POSIXError.lastError : POSIXError(.EIO) }
        done += result
    }
}

extension POSIXError {
    /// The error for the current `errno`.
    fileprivate static var lastError: POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
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
