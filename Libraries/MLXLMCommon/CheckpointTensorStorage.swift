// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

#if canImport(Darwin)
import Cmlx
import Darwin
import os
#endif

/// Keep source buffers alive while checkpoint preparation can rename or replace arrays.
package final class CheckpointTensorStorage {
    #if canImport(Darwin)
    fileprivate final class File: Sendable {
        let handle: FileHandle
        let url: URL
        let status: stat
        let directory: URL
        private let mappedAddresses = OSAllocatedUnfairLock(initialState: Set<UInt>())

        init(url: URL, directory: URL) throws {
            let handle = try FileHandle(forReadingFrom: url)
            var status = stat()
            guard fstat(handle.fileDescriptor, &status) == 0,
                status.st_mode & S_IFMT == S_IFREG
            else { throw POSIXError(.EINVAL) }
            self.handle = handle
            self.url = url
            self.status = status
            self.directory = directory
        }

        deinit { try? handle.close() }

        func matches(_ current: stat) -> Bool {
            status.st_dev == current.st_dev && status.st_ino == current.st_ino
                && status.st_size == current.st_size
                && status.st_mtimespec.tv_sec == current.st_mtimespec.tv_sec
                && status.st_mtimespec.tv_nsec == current.st_mtimespec.tv_nsec
                && status.st_ctimespec.tv_sec == current.st_ctimespec.tv_sec
                && status.st_ctimespec.tv_nsec == current.st_ctimespec.tv_nsec
        }

        func unchanged(checkPath: Bool) -> Bool {
            var current = stat()
            guard fstat(handle.fileDescriptor, &current) == 0, matches(current) else {
                return false
            }
            return !checkPath || (stat(url.path, &current) == 0 && matches(current))
        }

        func containsMappedAddress(_ address: UInt) -> Bool {
            mappedAddresses.withLock { $0.contains(address) }
        }

        /// The tensor in file-backed storage, or nil if the file changed.
        func array(for array: MLXArray, span: SafetensorSpan) throws -> MLXArray? {
            guard unchanged(checkPath: false) else { return nil }
            guard span.byteOffset >= 0, span.byteCount > 0,
                span.byteCount == array.nbytes,
                span.byteOffset <= status.st_size,
                span.byteCount <= status.st_size - span.byteOffset
            else { throw POSIXError(.EINVAL) }
            // Typed reads need aligned views, so tensors at other offsets are copied first.
            let result =
                if span.byteOffset.isMultiple(of: Int64(MemoryLayout<UInt64>.alignment)) {
                    try fileBackedArray(
                        like: array, mapping: handle.fileDescriptor, offset: span.byteOffset)
                } else {
                    try fileBackedArray(
                        like: array, copying: handle.fileDescriptor, offset: span.byteOffset,
                        directory: directory)
                }
            let address = UInt(bitPattern: mlx_array_data_uint8(result.ctx))
            mappedAddresses.withLock { _ = $0.insert(address) }
            return result
        }
    }

    private struct Source {
        let snapshot: MLXArray
        let strides: [Int]
        let location: Mapping
    }

    private let locations: [String: Mapping]
    private let vocabularySizes: Set<Int>
    private var sources: [UInt: Source] = [:]
    #endif

    package struct Mapping: Sendable {
        #if canImport(Darwin)
        fileprivate let file: File
        fileprivate let span: SafetensorSpan
        #endif

        func array(for array: MLXArray) throws -> MLXArray? {
            #if canImport(Darwin)
            return try file.array(for: array, span: span)
            #else
            return nil
            #endif
        }
    }

    let mappings: [String: Mapping]

    /// Tensors that cannot map in place are copied into unlinked files in `directory`.
    init?(
        model: BaseLanguageModel, policy: EmbeddingStoragePolicy, urls: [URL],
        quantization: BaseConfiguration.PerLayerQuantization? = nil,
        directory: URL = FileManager.default.temporaryDirectory
    ) {
        #if canImport(Darwin)
        guard policy == .automatic,
            let provider = model as? any InputOnlyEmbeddingsProviding,
            !provider.inputOnlyEmbeddings.isEmpty,
            GPU.deviceInfo().maxBufferSize > 0
        else { return nil }
        var locations: [String: Mapping] = [:]
        do {
            for url in urls {
                let file = try File(url: url, directory: directory)
                for span in try safetensorSpansInFileOrder(url: url) {
                    locations[span.name] = Mapping(file: file, span: span)
                }
                guard file.unchanged(checkPath: true) else { return nil }
            }
        } catch { return nil }
        self.locations = locations
        self.vocabularySizes = Set(provider.inputOnlyEmbeddings.map { $0.weight.dim(0) })
        let eligible = Set(provider.inputOnlyEmbeddings.map(ObjectIdentifier.init))
        var mappings: [String: Mapping] = [:]
        for (name, module) in model.namedModules() where eligible.contains(ObjectIdentifier(module))
        {
            guard let embedding = module as? Embedding else { continue }
            let parameters: [String]
            if embedding is QuantizedEmbedding {
                guard supportsFileBackedStorage(embedding) else { continue }
                parameters = ["weight", "scales", "biases"]
            } else if let settings = quantization?.quantization(layer: name) {
                guard type(of: embedding) == Embedding.self,
                    settings.mode == .affine, locations["\(name).scales"] != nil
                else { continue }
                parameters = ["weight", "scales", "biases"]
            } else {
                guard supportsFileBackedStorage(embedding) else { continue }
                parameters = ["weight"]
            }
            for parameter in parameters {
                let path = "\(name).\(parameter)"
                mappings[path] = locations[path]
            }
        }
        self.mappings = mappings
        #else
        return nil
        #endif
    }

    func record(_ weights: [String: MLXArray]) {
        #if canImport(Darwin)
        for (name, value) in weights where value.ndim == 2 && vocabularySizes.contains(value.dim(0))
        {
            guard let location = locations[name], location.file.unchanged(checkPath: true),
                location.span.byteCount == value.nbytes
            else { continue }
            var context = mlx_array_new()
            guard mlx_array_set(&context, value.ctx) == 0 else {
                mlx_array_free(context)
                continue
            }
            let snapshot = MLXArray(context)
            let data = snapshot.asData(access: .noCopy)
            data.data.withUnsafeBytes { bytes in
                guard let pointer = bytes.baseAddress else { return }
                sources[UInt(bitPattern: pointer)] = Source(
                    snapshot: snapshot, strides: data.strides, location: location)
            }
        }
        #endif
    }

    #if canImport(Darwin)
    func array(matching array: MLXArray) throws -> MLXArray? {
        let data = array.asData(access: .noCopy)
        let source = data.data.withUnsafeBytes { bytes -> Source? in
            guard let pointer = bytes.baseAddress else { return nil }
            return sources[UInt(bitPattern: pointer)]
        }
        guard let source, source.snapshot.shape == array.shape,
            source.snapshot.dtype == array.dtype, source.strides == data.strides,
            source.location.file.unchanged(checkPath: false)
        else { return nil }
        let address = data.data.withUnsafeBytes { $0.baseAddress.map(UInt.init(bitPattern:)) }
        if let address, source.location.file.containsMappedAddress(address) {
            return source.snapshot
        }
        return try source.location.array(for: array)
    }
    #endif
}
