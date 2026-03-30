import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        // We must force sorting/flattening when SSD streaming is active to properly batch
        // expert kernel dispatches dynamically over contiguous arrays.
        let isSSDStreaming = ProcessInfo.processInfo.environment["EXPERIMENTAL_SSD_STREAM"] != nil
        if isSSDStreaming { MLX.eval(indices) }
        let doSort = (indices.size >= 64) || isSSDStreaming

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        x = downProj(
            activation(xGate) * xUp,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public var tensorName: String?

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        if let envPath = ProcessInfo.processInfo.environment["EXPERIMENTAL_SSD_STREAM"] {
            MLX.eval(indices)
            if indices.size == 0 {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }
            
            let cpuIndices = indices.asArray(UInt32.self)
            var expertResults = [MLXArray]()
            var startIdx = 0
            
            // Resolve safetensors path + tensor name once per forward pass
            let ssdInfo: (path: String, tensorName: String)? = {
                guard let tName = self.tensorName,
                      let filename = ExpertStreamerManager.shared?.getFile(for: tName) else { return nil }
                let path = URL(fileURLWithPath: envPath).appendingPathComponent(filename).path
                return (path, tName)
            }()
            
            while startIdx < cpuIndices.count {
                let currentExpert = Int(cpuIndices[startIdx])
                var endIdx = startIdx + 1
                while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == currentExpert {
                    endIdx += 1
                }
                
                let rangeX = x[startIdx ..< endIdx]
                let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
                
                let readStart = DispatchTime.now().uptimeNanoseconds
                let expertWeight: MLXArray
                if let info = ssdInfo {
                    // Fast path: LoadSSDExpert allocates fresh allocator::malloc memory and
                    // pread()s directly from NVMe into it — full 5 GB/s NVMe throughput.
                    // The cache-keying bug (E → tensor_name) is now fixed so offsets are correct.
                    expertWeight = MLXFast.streamedGatherMM(
                        x: rangeX,
                        wShape: self.weight,
                        activeExpert: UInt32(currentExpert),
                        safetensorsPath: info.path,
                        tensorName: info.tensorName
                    )
                } else {
                    // Slow fallback: use mmap-backed slice + prefault when no SSD mapping available
                    let w = self.weight[currentExpert ..< currentExpert + 1]
                    MLX.eval(w)
                    MLXFast.prefault(w)
                    expertWeight = w
                }
                let readEnd = DispatchTime.now().uptimeNanoseconds
                SSDStreamMetrics.shared.record(bytes: expertWeight.nbytes, timeNs: readEnd - readStart)
                
                let expertScales = self.scales[currentExpert ..< currentExpert + 1]
                var expertBiases: MLXArray? = nil
                if let b = self.biases {
                    expertBiases = b[currentExpert ..< currentExpert + 1]
                }
                
                var expertOutput = MLX.gatherQuantizedMM(
                    rangeX,
                    expertWeight,
                    scales: expertScales,
                    biases: expertBiases,
                    rhsIndices: expertIndices,
                    transpose: true,
                    groupSize: self.groupSize,
                    bits: self.bits,
                    mode: mode,
                    sortedIndices: true
                )
                
                if let bias = self.bias {
                    let biasSlice = bias[currentExpert ..< currentExpert + 1]
                    expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
                }
                
                MLX.eval(expertOutput)
                Stream.gpu.synchronize()
                
                expertResults.append(expertOutput)
                startIdx = endIdx
            }
            
            if expertResults.isEmpty {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }
            
            return MLX.concatenated(expertResults, axis: 0)
        }

        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

}

public class ExpertStreamerManager {
    public static var shared: ExpertStreamerManager?

    public let weightMap: [String: String]

    public init(modelDirectory: URL) {
        var map = [String: String]()
        let indexUrl = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexUrl),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let weightMapJson = json["weight_map"] as? [String: String] {
            map = weightMapJson
        }
        self.weightMap = map
    }

    public func getFile(for tensorName: String) -> String? {
        return weightMap[tensorName]
    }
}

public final class SSDStreamMetrics: @unchecked Sendable {
    public static let shared = SSDStreamMetrics()
    private var totalBytes: Int = 0
    private var totalTimeNs: UInt64 = 0
    private var readCount: Int = 0
    private var lastLogTimeNs: UInt64 = DispatchTime.now().uptimeNanoseconds
    private let lock = NSLock()
    
    public func record(bytes: Int, timeNs: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        totalBytes += bytes
        totalTimeNs += timeNs
        readCount += 1
        
        let now = DispatchTime.now().uptimeNanoseconds
        if now - lastLogTimeNs >= 1_000_000_000 {
            let count = readCount
            let bytes = totalBytes
            let ns = totalTimeNs
            
            self.readCount = 0
            self.totalBytes = 0
            self.totalTimeNs = 0
            self.lastLogTimeNs = now
            
            if count > 0 {
                let mb = Double(bytes) / (1024.0 * 1024.0)
                let avgMs = (Double(ns) / 1_000_000.0) / Double(count)
                print(String(format: "[⚡️ SSD Stream] %.1f MB/s over %d chunks | Avg latency per chunk: %.6f ms", mb, count, avgMs))
                fflush(stdout)
            }
        }
    }
}

