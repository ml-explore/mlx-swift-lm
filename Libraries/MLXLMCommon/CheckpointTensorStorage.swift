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
        let maximumLength: Int
        private let mappedAddresses = OSAllocatedUnfairLock(initialState: Set<UInt>())

        init(url: URL, maximumLength: Int) throws {
            let handle = try FileHandle(forReadingFrom: url)
            var status = stat()
            guard fstat(handle.fileDescriptor, &status) == 0,
                status.st_mode & S_IFMT == S_IFREG
            else { throw POSIXError(.EINVAL) }
            self.handle = handle
            self.url = url
            self.status = status
            self.maximumLength = maximumLength
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

        func array(for array: MLXArray, span: SafetensorSpan) throws -> MLXArray? {
            // Keep larger scalar views aligned on runtimes with typed item reads.
            guard unchanged(checkPath: false),
                span.byteOffset.isMultiple(of: Int64(MemoryLayout<UInt64>.alignment))
            else { return nil }
            let page = Int(getpagesize())
            guard span.byteOffset >= 0, span.byteCount > 0,
                span.byteCount == array.nbytes,
                span.byteOffset <= status.st_size,
                span.byteCount <= status.st_size - span.byteOffset
            else { throw POSIXError(.EINVAL) }
            let alignedOffset = span.byteOffset / Int64(page) * Int64(page)
            let offset = Int(span.byteOffset - alignedOffset)
            let (extent, overflow) = array.nbytes.addingReportingOverflow(offset + page - 1)
            guard !overflow else { throw POSIXError(.EOVERFLOW) }
            let pageCount = extent / page
            guard pageCount * page / array.itemSize <= Int32.max,
                pageCount * page <= maximumLength
            else {
                throw POSIXError(.EOVERFLOW)
            }
            let length = pageCount * page
            guard
                let pointer = mmap(
                    nil, length, PROT_READ | PROT_WRITE, MAP_PRIVATE,
                    handle.fileDescriptor, off_t(alignedOffset)),
                pointer != MAP_FAILED
            else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
            _ = madvise(pointer, length, MADV_RANDOM)
            let storage = MLXArray(
                rawPointer: pointer, [pageCount, page / array.itemSize], dtype: array.dtype
            ) {
                munmap(pointer, length)
            }
            guard
                mlx_array_data_uint8(storage.ctx).map(UnsafeRawPointer.init)
                    == UnsafeRawPointer(pointer)
            else { return nil }
            let result = asStrided(
                storage, array.shape, offset: offset / array.itemSize, stream: .cpu)
            eval(result)
            let address = UInt(bitPattern: pointer) + UInt(offset)
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

    init?(
        model: BaseLanguageModel, policy: EmbeddingStoragePolicy, urls: [URL],
        quantization: BaseConfiguration.PerLayerQuantization? = nil
    ) {
        #if canImport(Darwin)
        guard policy == .automatic,
            let provider = model as? any InputOnlyEmbeddingsProviding,
            !provider.inputOnlyEmbeddings.isEmpty
        else { return nil }
        let maximumLength = GPU.deviceInfo().maxBufferSize
        guard maximumLength > 0 else { return nil }
        var locations: [String: Mapping] = [:]
        do {
            for url in urls {
                let file = try File(url: url, maximumLength: maximumLength)
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
