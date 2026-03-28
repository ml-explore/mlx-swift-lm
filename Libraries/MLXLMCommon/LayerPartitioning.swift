// LayerPartitioning.swift — GPU/CPU layer partitioning support
//
// Provides a protocol for models that support running a subset of layers
// on GPU and the rest on CPU. On Apple Silicon with unified memory,
// there is zero data copy cost between CPU and GPU layers.
//
// Usage:
//   1. Model conforms to LayerPartitionable
//   2. After loading, call model.setGPULayers(N) to set the split
//   3. The model's forward pass uses partitionedLayerCall() to route each layer

import Foundation
import MLX
import MLXNN

// MARK: - Layer Partitioning Protocol

/// Protocol for models that support per-layer GPU/CPU device placement.
///
/// Models conforming to this protocol can run a configurable number of
/// transformer layers on GPU and the remainder on CPU, enabling inference
/// of models larger than available GPU memory on Apple Silicon.
///
/// Because Apple Silicon uses unified memory architecture (UMA), there is
/// no data transfer cost between CPU and GPU layers — only the compute
/// engine differs.
public protocol LayerPartitionable: AnyObject {

    /// Number of layers to run on GPU. Layers beyond this index run on CPU.
    /// When nil or >= total layer count, all layers run on GPU (default behavior).
    var gpuLayerCount: Int? { get set }

    /// Total number of transformer layers in the model.
    var totalLayerCount: Int { get }
}

extension LayerPartitionable {

    /// Set the number of layers to run on GPU.
    /// Pass nil to run all layers on GPU (default).
    /// Values are clamped to [0, totalLayerCount].
    public func setGPULayers(_ count: Int?) {
        if let count {
            gpuLayerCount = min(max(0, count), totalLayerCount)
        } else {
            gpuLayerCount = nil
        }
    }

    /// Whether a given layer index should run on GPU.
    public func isGPULayer(_ index: Int) -> Bool {
        guard let gpuCount = gpuLayerCount else { return true }
        return index < gpuCount
    }

    /// The number of layers running on CPU.
    public var cpuLayerCount: Int {
        guard let gpuCount = gpuLayerCount else { return 0 }
        return max(0, totalLayerCount - gpuCount)
    }

    /// Summary string for logging.
    public var partitionSummary: String {
        guard let gpuCount = gpuLayerCount else {
            return "\(totalLayerCount)/\(totalLayerCount) GPU (full)"
        }
        let cpuCount = totalLayerCount - gpuCount
        return "\(gpuCount) GPU / \(cpuCount) CPU"
    }
}

// MARK: - Expert Streaming Protocol

/// Protocol for Mixture-of-Experts (MoE) models that support SSD expert streaming.
///
/// When `streamExperts` is enabled, the model evaluates intermediate states and
/// clears the MLX cache layer-by-layer. This pattern allows the OS Page Cache
/// to effortlessly load multi-gigabyte expert weights on the fly (`mmap`) and
/// discard them immediately to prevent Out-Of-Memory errors on memory-constrained
/// Apple Silicon devices (e.g., loading a 100B model on 16GB RAM).
public protocol StreamableMoE: AnyObject {
    /// Whether this model should stream expert weights from SSD layer-by-layer
    var streamExperts: Bool { get set }
}

// MARK: - Partitioned Layer Execution

/// Execute a single transformer layer on the appropriate device (GPU or CPU).
///
/// This is the core routing function. When a layer is designated for CPU,
/// the entire forward pass of that layer runs on the CPU stream.
/// MLX's lazy evaluation ensures the computation graph is built correctly
/// across device boundaries with zero-copy UMA transfers.
///
/// - Parameters:
///   - index: The layer index (0-based)
///   - gpuLayerCount: Number of layers assigned to GPU (nil = all GPU)
///   - stream: If true, synchronously evaluates the layer and clears MLX cache (Flash-MoE style)
///   - body: The layer forward pass closure
/// - Returns: The layer output
public func partitionedLayerCall<T>(
    index: Int,
    gpuLayerCount: Int?,
    stream: Bool = false,
    body: () -> T
) -> T {
    let result: T
    
    if let gpuCount = gpuLayerCount, index >= gpuCount {
        // CPU layer — scope the computation to the CPU device
        result = Device.withDefaultDevice(.cpu, body)
    } else {
        // GPU layer (default path — no overhead)
        result = body()
    }
    
    if stream, let array = result as? MLXArray {
        // 1. Force evaluation of this single layer.
        // The router hits K experts. The OS pages ONLY those K experts from SSD to RAM.
        eval(array)
        
        // 2. Clear MLX's internal Metal buffer pool.
        // This ensures temporary buffers don't push the Attention weights out of the OS page cache.
        GPU.clearCache()
    }
    
    return result
}
