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


// Shared struct for expert range tracking across projections
public struct ExpertRange {
    public let id: Int
    public let start: Int
    public let end: Int
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") public var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") public var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray

    // ── Async pipeline state (SSD streaming optimization) ──
    // Persistent buffers: allocated once per layer, reused across tokens.
    // Avoids per-token buffer allocation + eval overhead.
    private var _persistentGate: [MLXArray]?
    private var _persistentUp: [MLXArray]?
    private var _persistentDown: [MLXArray]?
    // Previous token's expert routing per layer for speculative prefetch.
    private var _previousExpertIds: [Int]?

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
        let isSSDStreaming = ExpertStreamingConfig.shared.isEnabled
        // NOTE: indices eval deferred to inside the cross-projection path below,
        // where it's merged with buffer allocation into fewer eval calls.
        let doSort = (indices.size >= 64) || isSSDStreaming

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        // ── Cross-projection batched SSD streaming path ──────────────────
        // When all 3 projections are quantized and SSD-streaming is active,
        // orchestrate buffer allocation, pread, and compute across all 3
        // projections to minimize MLX.eval() calls:
        //   - Single-token (fast path): 1 eval merges idx + buffer alloc
        //   - Prompt (large batch): 2 evals (idx, then buffers)
        //   - NO final eval — next layer's eval(idx) forces this layer
        // This reduces from 4 evals/layer (original) to 1 eval/layer.
        if isSSDStreaming,
           let qGate = gateProj as? QuantizedSwitchLinear,
           let qUp = upProj as? QuantizedSwitchLinear,
           let qDown = downProj as? QuantizedSwitchLinear,
           let gateSSD = qGate.resolveSSDInfo(),
           let upSSD = qUp.resolveSSDInfo(),
           let downSSD = qDown.resolveSSDInfo() {

            // ── EVAL REDUCTION STRATEGY ──────────────────────────────────────
            // For single-token generation (idx.size ≤ 32), we merge the sorted-
            // indices eval and buffer-allocation eval into ONE call, cutting from
            // 3 evals/layer to 1.  The final MLX.eval(x) is removed entirely:
            // the NEXT layer's SwitchGLU eval(idx) transitively forces this
            // layer's full output (including KV cache) through the lazy
            // dependency chain.  For the last layer, the generation loop's eval
            // of logits handles it.
            // ─────────────────────────────────────────────────────────────────

            if idx.size <= 32 {
                // ── FAST PATH: single-token generation with async I/O-GPU pipeline ──
                //
                // STRATEGY: Overlap NVMe I/O with GPU compute using asyncEval.
                //
                // Cold path (first token): Allocate persistent buffers, merged eval,
                //   full pread — same as ssd-opt-v1 baseline.
                //
                // Warm path (subsequent tokens): asyncEval(idx) starts GPU work
                //   (prev layer expert compute + current attention/router) while
                //   CPU speculatively preads predicted experts (from previous token's
                //   routing) into persistent buffers. After GPU sync, only ~30% of
                //   experts need on-demand pread (misses). Saves ~60ms/token by
                //   hiding I/O behind GPU compute.
                //
                // Memory cost: ~5GB for persistent buffers across 48 layers
                //   (vs ~13GB for the failed in-memory cache approach).

                let maxBuffers = idx.size  // typically 8 (top_k)

                if _persistentGate == nil {
                    // ── COLD PATH: first token, allocate persistent buffers ──
                    _persistentGate = qGate.allocateExpertBuffers(maxBuffers)
                    _persistentUp = qUp.allocateExpertBuffers(maxBuffers)
                    _persistentDown = qDown.allocateExpertBuffers(maxBuffers)

                    // Merged eval: idx + buffer allocations (same as ssd-opt-v1)
                    var toEval: [MLXArray] = [idx]
                    toEval.append(contentsOf: _persistentGate!)
                    toEval.append(contentsOf: _persistentUp!)
                    toEval.append(contentsOf: _persistentDown!)
                    MLX.eval(toEval)

                    // Handle empty indices
                    if idx.size == 0 {
                        var outShape = x.shape
                        outShape[outShape.count - 1] = qDown.outputDims
                        let result = MLXArray.zeros(outShape).asType(.float16)
                        if doSort {
                            return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                        }
                        return MLX.squeezed(result, axis: -2)
                    }

                    // Parse routing
                    let cpuIndices = idx.asArray(UInt32.self)
                    var ranges = [ExpertRange]()
                    var startIdx = 0
                    while startIdx < cpuIndices.count {
                        let eid = Int(cpuIndices[startIdx])
                        var endIdx = startIdx + 1
                        while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                        ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                        startIdx = endIdx
                    }

                    // Full concurrent pread (baseline path)
                    let totalReads = ranges.count * 3
                    DispatchQueue.concurrentPerform(iterations: totalReads) { i in
                        let expertIdx = i / 3
                        let projIdx = i % 3
                        let r = ranges[expertIdx]
                        switch projIdx {
                        case 0:
                            MLXFast.preadInto(self._persistentGate![expertIdx], safetensorsPath: gateSSD.path,
                                              tensorName: gateSSD.tensorName, expertIndex: UInt32(r.id))
                        case 1:
                            MLXFast.preadInto(self._persistentUp![expertIdx], safetensorsPath: upSSD.path,
                                              tensorName: upSSD.tensorName, expertIndex: UInt32(r.id))
                        default:
                            MLXFast.preadInto(self._persistentDown![expertIdx], safetensorsPath: downSSD.path,
                                              tensorName: downSSD.tensorName, expertIndex: UInt32(r.id))
                        }
                    }

                    // Store routing for next token's predictions
                    _previousExpertIds = ranges.map { $0.id }

                    // Lazy compute
                    let usedGate = Array(_persistentGate![0..<ranges.count])
                    let usedUp = Array(_persistentUp![0..<ranges.count])
                    let usedDown = Array(_persistentDown![0..<ranges.count])
                    let xGate = qGate.computeExperts(x, buffers: usedGate, ranges: ranges)
                    let xUp = qUp.computeExperts(x, buffers: usedUp, ranges: ranges)
                    let intermediate = activation(xGate) * xUp
                    x = qDown.computeExperts(intermediate, buffers: usedDown, ranges: ranges)

                } else {
                    // ── WARM PATH: asyncEval + speculative pread pipeline ──

                    // Start GPU work asynchronously: forces prev layer's expert
                    // compute + current layer's attention + router.
                    // GPU time: ~2.7ms. CPU is free immediately.
                    asyncEval(idx)

                    // Speculative pread during GPU async window.
                    // Load previous token's experts into persistent buffers.
                    // ~70% will match this token's routing (expert stickiness).
                    // The 1.7ms of pread overlaps with 2.7ms of GPU work.
                    if let prevIds = _previousExpertIds {
                        let specCount = min(prevIds.count, maxBuffers)
                        let specReads = specCount * 3
                        DispatchQueue.concurrentPerform(iterations: specReads) { i in
                            let slot = i / 3
                            let proj = i % 3
                            let expertId = prevIds[slot]
                            switch proj {
                            case 0:
                                MLXFast.preadInto(self._persistentGate![slot], safetensorsPath: gateSSD.path,
                                                  tensorName: gateSSD.tensorName, expertIndex: UInt32(expertId))
                            case 1:
                                MLXFast.preadInto(self._persistentUp![slot], safetensorsPath: upSSD.path,
                                                  tensorName: upSSD.tensorName, expertIndex: UInt32(expertId))
                            default:
                                MLXFast.preadInto(self._persistentDown![slot], safetensorsPath: downSSD.path,
                                                  tensorName: downSSD.tensorName, expertIndex: UInt32(expertId))
                            }
                        }
                    }

                    // Sync on idx (blocks until GPU finishes attention + router)
                    if idx.size == 0 {
                        var outShape = x.shape
                        outShape[outShape.count - 1] = qDown.outputDims
                        let result = MLXArray.zeros(outShape).asType(.float16)
                        if doSort {
                            return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                        }
                        return MLX.squeezed(result, axis: -2)
                    }

                    // Parse actual routing
                    let cpuIndices = idx.asArray(UInt32.self)
                    var ranges = [ExpertRange]()
                    var startIdx = 0
                    while startIdx < cpuIndices.count {
                        let eid = Int(cpuIndices[startIdx])
                        var endIdx = startIdx + 1
                        while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                        ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                        startIdx = endIdx
                    }
                    let actualIds = ranges.map { $0.id }

                    // Map actual experts to persistent buffer slots.
                    // Hits: buffer slot already has correct data from speculative pread.
                    // Misses: assign to a free slot, pread on demand.
                    var usedGate = [MLXArray]()
                    var usedUp = [MLXArray]()
                    var usedDown = [MLXArray]()

                    if let prevIds = _previousExpertIds {
                        var prevSlotMap = [Int: Int]()  // expertId -> buffer slot
                        for (slot, eid) in prevIds.enumerated() {
                            prevSlotMap[eid] = slot
                        }

                        var usedSlots = Set<Int>()
                        var missInfo = [(rangeIdx: Int, expertId: Int, bufferSlot: Int)]()

                        for (ri, r) in ranges.enumerated() {
                            if let slot = prevSlotMap[r.id], !usedSlots.contains(slot) {
                                // HIT: persistent buffer[slot] has correct expert data
                                usedGate.append(_persistentGate![slot])
                                usedUp.append(_persistentUp![slot])
                                usedDown.append(_persistentDown![slot])
                                usedSlots.insert(slot)
                            } else {
                                // MISS: find a free slot
                                let freeSlot = (0..<maxBuffers).first { !usedSlots.contains($0) }!
                                usedGate.append(_persistentGate![freeSlot])
                                usedUp.append(_persistentUp![freeSlot])
                                usedDown.append(_persistentDown![freeSlot])
                                usedSlots.insert(freeSlot)
                                missInfo.append((ri, r.id, freeSlot))
                            }
                        }

                        // Pread only misses (~30% of experts, ~6 reads at QD=6)
                        if !missInfo.isEmpty {
                            let totalMissReads = missInfo.count * 3
                            DispatchQueue.concurrentPerform(iterations: totalMissReads) { i in
                                let mIdx = i / 3
                                let proj = i % 3
                                let info = missInfo[mIdx]
                                switch proj {
                                case 0:
                                    MLXFast.preadInto(self._persistentGate![info.bufferSlot],
                                                      safetensorsPath: gateSSD.path,
                                                      tensorName: gateSSD.tensorName,
                                                      expertIndex: UInt32(info.expertId))
                                case 1:
                                    MLXFast.preadInto(self._persistentUp![info.bufferSlot],
                                                      safetensorsPath: upSSD.path,
                                                      tensorName: upSSD.tensorName,
                                                      expertIndex: UInt32(info.expertId))
                                default:
                                    MLXFast.preadInto(self._persistentDown![info.bufferSlot],
                                                      safetensorsPath: downSSD.path,
                                                      tensorName: downSSD.tensorName,
                                                      expertIndex: UInt32(info.expertId))
                                }
                            }
                        }
                    } else {
                        // No predictions available — full pread fallback
                        for i in 0..<ranges.count {
                            usedGate.append(_persistentGate![i])
                            usedUp.append(_persistentUp![i])
                            usedDown.append(_persistentDown![i])
                        }
                        let totalReads = ranges.count * 3
                        DispatchQueue.concurrentPerform(iterations: totalReads) { i in
                            let expertIdx = i / 3
                            let projIdx = i % 3
                            let r = ranges[expertIdx]
                            switch projIdx {
                            case 0:
                                MLXFast.preadInto(self._persistentGate![expertIdx], safetensorsPath: gateSSD.path,
                                                  tensorName: gateSSD.tensorName, expertIndex: UInt32(r.id))
                            case 1:
                                MLXFast.preadInto(self._persistentUp![expertIdx], safetensorsPath: upSSD.path,
                                                  tensorName: upSSD.tensorName, expertIndex: UInt32(r.id))
                            default:
                                MLXFast.preadInto(self._persistentDown![expertIdx], safetensorsPath: downSSD.path,
                                                  tensorName: downSSD.tensorName, expertIndex: UInt32(r.id))
                            }
                        }
                    }

                    // Update routing for next token's predictions
                    _previousExpertIds = actualIds

                    // Lazy compute (no eval — next layer forces it)
                    let xGate = qGate.computeExperts(x, buffers: usedGate, ranges: ranges)
                    let xUp = qUp.computeExperts(x, buffers: usedUp, ranges: ranges)
                    let intermediate = activation(xGate) * xUp
                    x = qDown.computeExperts(intermediate, buffers: usedDown, ranges: ranges)
                }

            } else {
                // ── PROMPT PATH: larger batches ──
                // Eval indices first (needed for range count), then allocate exact buffers.
                MLX.eval(idx)

                // Handle empty indices
                if idx.size == 0 {
                    var outShape = x.shape
                    outShape[outShape.count - 1] = qDown.outputDims
                    let result = MLXArray.zeros(outShape).asType(.float16)
                    if doSort {
                        return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                    }
                    return MLX.squeezed(result, axis: -2)
                }

                // Parse expert ranges
                let cpuIndices = idx.asArray(UInt32.self)
                var ranges = [ExpertRange]()
                var startIdx = 0
                while startIdx < cpuIndices.count {
                    let eid = Int(cpuIndices[startIdx])
                    var endIdx = startIdx + 1
                    while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                    ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                    startIdx = endIdx
                }

                // Allocate exact buffer count and eval
                let gateBuffers = qGate.allocateExpertBuffers(ranges.count)
                let upBuffers = qUp.allocateExpertBuffers(ranges.count)
                let downBuffers = qDown.allocateExpertBuffers(ranges.count)
                MLX.eval(gateBuffers + upBuffers + downBuffers)

                // Concurrent pread (same as fast path)
                let totalReads = ranges.count * 3
                DispatchQueue.concurrentPerform(iterations: totalReads) { i in
                    let expertIdx = i / 3
                    let projIdx = i % 3
                    let r = ranges[expertIdx]
                    switch projIdx {
                    case 0:
                        MLXFast.preadInto(gateBuffers[expertIdx], safetensorsPath: gateSSD.path,
                                          tensorName: gateSSD.tensorName, expertIndex: UInt32(r.id))
                    case 1:
                        MLXFast.preadInto(upBuffers[expertIdx], safetensorsPath: upSSD.path,
                                          tensorName: upSSD.tensorName, expertIndex: UInt32(r.id))
                    default:
                        MLXFast.preadInto(downBuffers[expertIdx], safetensorsPath: downSSD.path,
                                          tensorName: downSSD.tensorName, expertIndex: UInt32(r.id))
                    }
                }

                // Lazy compute (no eval — next layer forces it)
                let xGate = qGate.computeExperts(x, buffers: gateBuffers, ranges: ranges)
                let xUp = qUp.computeExperts(x, buffers: upBuffers, ranges: ranges)
                let intermediate = activation(xGate) * xUp
                x = qDown.computeExperts(intermediate, buffers: downBuffers, ranges: ranges)
            }

            if doSort {
                x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
            }
            return MLX.squeezed(x, axis: -2)
        }

        // ── Fallback: original sequential path (non-SSD or non-quantized) ──
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
    @ModuleInfo(key: "weight") public var weight: MLXArray
    @ModuleInfo(key: "bias") public var bias: MLXArray?

    public let inputDims: Int
    public let outputDims: Int
    public let numExperts: Int

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
        if ExpertStreamingConfig.shared.isEnabled {
            MLX.eval(indices)
            if indices.size == 0 {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }

            let cpuIndices = indices.asArray(UInt32.self)
            var expertResults = [MLXArray]()
            var startIdx = 0

            // macOS directNVMe: resolve the safetensors shard + tensor offset once.
            // iOS mmapPageCache: ssdInfo = nil → falls through to mmap prefault below.
            let ssdInfo: (path: String, tensorName: String)? = {
                #if os(macOS)
                guard ExpertStreamingConfig.shared.useDirectNVMe,
                      let tName = self.tensorName,
                      let filename = ExpertStreamerManager.shared?.getFile(for: tName),
                      let dir = ExpertStreamingConfig.shared.modelDirectory else { return nil }
                let path = dir.appendingPathComponent(filename).path
                return (path, tName)
                #else
                return nil  // iOS always uses mmap fallback
                #endif
            }()

            // ---- Parse expert ranges ----
            var ranges = [ExpertRange]()
            while startIdx < cpuIndices.count {
                let eid = Int(cpuIndices[startIdx])
                var endIdx = startIdx + 1
                while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                startIdx = endIdx
            }

            if let info = ssdInfo {
                // ---- Batch-allocate weight buffers (1 eval for all) ----
                var buffers = [MLXArray]()
                for _ in ranges {
                    buffers.append(MLXArray.zeros([1, self.weight.dim(1), self.weight.dim(2)]).asType(self.weight.dtype))
                }
                MLX.eval(buffers)

                // ---- Sequential pread into each fresh buffer ----
                for (i, r) in ranges.enumerated() {
                    MLXFast.preadInto(
                        buffers[i],
                        safetensorsPath: info.path,
                        tensorName: info.tensorName,
                        expertIndex: UInt32(r.id)
                    )
                }

                // ---- GPU compute for all experts ----
                for (i, r) in ranges.enumerated() {
                    let rangeX = x[r.start ..< r.end]
                    let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
                    let expertScales = self.scales[r.id ..< r.id + 1]
                    var expertBiases: MLXArray? = nil
                    if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }

                    var expertOutput = MLX.gatherQuantizedMM(
                        rangeX, buffers[i],
                        scales: expertScales, biases: expertBiases,
                        rhsIndices: expertIndices, transpose: true,
                        groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
                    )
                    if let bias = self.bias {
                        let biasSlice = bias[r.id ..< r.id + 1]
                        expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
                    }
                    let leadingShape = Array(rangeX.shape.dropLast())
                    let canonicalShape = leadingShape + [self.outputDims]
                    if expertOutput.shape != canonicalShape {
                        expertOutput = expertOutput.reshaped(canonicalShape)
                    }
                    expertResults.append(expertOutput)
                }
            } else {
                // iOS mmap fallback — original sequential path with per-expert eval
                for r in ranges {
                    let rangeX = x[r.start ..< r.end]
                    let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
                    let w = self.weight[r.id ..< r.id + 1]
                    MLX.eval(w)
                    MLXFast.prefault(w)
                    let expertScales = self.scales[r.id ..< r.id + 1]
                    var expertBiases: MLXArray? = nil
                    if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }
                    var expertOutput = MLX.gatherQuantizedMM(
                        rangeX, w,
                        scales: expertScales, biases: expertBiases,
                        rhsIndices: expertIndices, transpose: true,
                        groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
                    )
                    if let bias = self.bias {
                        let biasSlice = bias[r.id ..< r.id + 1]
                        expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
                    }
                    let leadingShape = Array(rangeX.shape.dropLast())
                    let canonicalShape = leadingShape + [self.outputDims]
                    if expertOutput.shape != canonicalShape {
                        expertOutput = expertOutput.reshaped(canonicalShape)
                    }
                    MLX.eval(expertOutput)
                    expertResults.append(expertOutput)
                }
            }

            // Batch eval all expert outputs at once (directNVMe path)
            if let _ = ssdInfo, !expertResults.isEmpty {
                MLX.eval(expertResults)
            }

            if expertResults.isEmpty {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }

            // PAPPS Heuristic: Prefetch exactly these experts so they are in cache for the N+1 token.
            if let info = ssdInfo {
                let uniqueIndices = Set(cpuIndices)
                for _ in uniqueIndices {
                    // MLXFast.pappsPrefetch(
                    //     safetensorsPath: info.path,
                    //     tensorName: info.tensorName,
                    //     expertIndex: idx
                    // )
                }
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


    // MARK: - Cross-projection batching helpers (SSD streaming)

    /// Resolve the safetensors path and tensor name for SSD streaming.
    public func resolveSSDInfo() -> (path: String, tensorName: String)? {
        #if os(macOS)
        guard ExpertStreamingConfig.shared.useDirectNVMe,
              let tName = self.tensorName,
              let filename = ExpertStreamerManager.shared?.getFile(for: tName),
              let dir = ExpertStreamingConfig.shared.modelDirectory else { return nil }
        let path = dir.appendingPathComponent(filename).path
        return (path, tName)
        #else
        return nil
        #endif
    }

    /// Allocate zero-filled weight buffers for `count` experts (lazy, not yet eval'd).
    public func allocateExpertBuffers(_ count: Int) -> [MLXArray] {
        var buffers = [MLXArray]()
        for _ in 0..<count {
            buffers.append(MLXArray.zeros([1, self.weight.dim(1), self.weight.dim(2)]).asType(self.weight.dtype))
        }
        return buffers
    }

    /// Load expert weights from SSD into pre-allocated (eval'd) buffers.
    public func loadExpertWeights(_ buffers: [MLXArray], ranges: [ExpertRange], ssdInfo: (path: String, tensorName: String)) {
        for (i, r) in ranges.enumerated() {
            MLXFast.preadInto(
                buffers[i],
                safetensorsPath: ssdInfo.path,
                tensorName: ssdInfo.tensorName,
                expertIndex: UInt32(r.id)
            )
        }
    }

    /// Compute expert outputs using pre-loaded weight buffers. Returns LAZY result (no eval).
    public func computeExperts(_ x: MLXArray, buffers: [MLXArray], ranges: [ExpertRange]) -> MLXArray {
        var expertResults = [MLXArray]()
        for (i, r) in ranges.enumerated() {
            let rangeX = x[r.start ..< r.end]
            let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
            let expertScales = self.scales[r.id ..< r.id + 1]
            var expertBiases: MLXArray? = nil
            if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }

            var expertOutput = MLX.gatherQuantizedMM(
                rangeX, buffers[i],
                scales: expertScales, biases: expertBiases,
                rhsIndices: expertIndices, transpose: true,
                groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
            )
            if let bias = self.bias {
                let biasSlice = bias[r.id ..< r.id + 1]
                expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
            }
            let leadingShape = Array(rangeX.shape.dropLast())
            let canonicalShape = leadingShape + [self.outputDims]
            if expertOutput.shape != canonicalShape {
                expertOutput = expertOutput.reshaped(canonicalShape)
            }
            expertResults.append(expertOutput)
        }

        if expertResults.isEmpty {
            var outShape = x.shape
            outShape[outShape.count - 1] = self.outputDims
            return MLXArray.zeros(outShape).asType(.float16)
        }
        return MLX.concatenated(expertResults, axis: 0)
    }
}

public class ExpertStreamerManager {
    nonisolated(unsafe) public static var shared: ExpertStreamerManager?

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
                // let mb = Double(bytes) / (1024.0 * 1024.0)
                // let avgMs = (Double(ns) / 1_000_000.0) / Double(count)
                // print(String(format: "[⚡️ SSD Stream] %.1f MB/s over %d chunks | Avg latency per chunk: %.6f ms", mb, count, avgMs))
                // fflush(stdout)
            }
        }
    }
}

