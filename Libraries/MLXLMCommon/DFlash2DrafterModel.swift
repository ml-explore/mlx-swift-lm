// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// DFlash2 (https://inco.ai/blog/dflash2/, reference https://github.com/z-lab/dflash)
// drafts a block of tokens in one pass from the target's hidden states; the
// target verifies the block in one forward pass. The drafter and target models
// live in MLXLLM.

// MARK: - Geometry

/// The target shape a drafter was distilled for, or the shape a target has.
public struct DFlash2Geometry: Equatable, Sendable, CustomStringConvertible {
    /// Decoder layers; `nil` when a drafter checkpoint does not record it.
    public var layerCount: Int?
    public var hiddenSize: Int
    public var vocabularySize: Int

    public init(layerCount: Int?, hiddenSize: Int, vocabularySize: Int) {
        self.layerCount = layerCount
        self.hiddenSize = hiddenSize
        self.vocabularySize = vocabularySize
    }

    public var description: String {
        "\(layerCount.map(String.init) ?? "?") layers, hidden \(hiddenSize), vocabulary \(vocabularySize)"
    }

    /// Whether a drafter distilled for `self`, reading `layerIds`, can draft for `target`.
    func accepts(_ target: DFlash2Geometry, layerIds: [Int]) -> Bool {
        guard let targetLayers = target.layerCount else { return false }
        return hiddenSize == target.hiddenSize && vocabularySize == target.vocabularySize
            && (layerCount ?? targetLayers) == targetLayers
            && layerIds.allSatisfy { (0 ..< targetLayers).contains($0) }
    }
}

// MARK: - Drafter

/// One block to propose. Every accept-dependent input may be lazy, so the next
/// proposal can be built while the previous verify pass runs.
public struct DFlash2ProposalRequest {
    /// `[1, width]` token ids: the anchor, then mask tokens.
    public var block: MLXArray
    /// `[1, S, targetLayerIds.count * hidden]` target outputs verified since the last proposal.
    public var targetHidden: MLXArray
    /// Absolute position of `targetHidden` row 0.
    public var contextPosition: Int
    /// `[]` int32 count of leading `targetHidden` rows that are committed context.
    public var validRows: MLXArray
    /// 0 selects greedily; above 0 samples and fills ``DFlash2Proposal/probabilities``.
    public var temperature: Float
    /// Token ids the target never emits; drafts skip them.
    public var suppressedTokens: SuppressTokensProcessor?

    public init(
        block: MLXArray, targetHidden: MLXArray, contextPosition: Int, validRows: MLXArray,
        temperature: Float, suppressedTokens: SuppressTokensProcessor? = nil
    ) {
        self.block = block
        self.targetHidden = targetHidden
        self.contextPosition = contextPosition
        self.validRows = validRows
        self.temperature = temperature
        self.suppressedTokens = suppressedTokens
    }
}

/// The drafted tokens and the per-position candidates they were chosen from.
public struct DFlash2Proposal {
    /// `[1, width - 1]` drafted token ids.
    public var tokens: MLXArray
    /// `[1, width - 1, K]` candidate ids per drafted position.
    public var candidates: MLXArray
    /// Selection probabilities over `candidates`, present when sampled.
    public var probabilities: MLXArray?

    public init(tokens: MLXArray, candidates: MLXArray, probabilities: MLXArray? = nil) {
        self.tokens = tokens
        self.candidates = candidates
        self.probabilities = probabilities
    }
}

/// Per-stream drafter state, owned by the iterator so one drafter serves many streams.
public struct DFlash2DrafterState {
    /// One context cache per drafter layer.
    public var contextCaches: [DFlash2ContextCache]

    public init(contextCaches: [DFlash2ContextCache]) {
        self.contextCaches = contextCaches
    }
}

/// A DFlash2 block drafter.
///
/// It attends over projections of the target's hidden states and predicts every
/// position of an `[anchor, MASK, ...]` block at once. It borrows the target's
/// embedding and head per call and holds no per-stream state, like
/// ``MTPDrafterModel``.
public protocol DFlash2DrafterModel: BaseLanguageModel {
    /// Tokens per verify pass the checkpoint was trained for, anchor included.
    var blockSize: Int { get }
    /// Token id that fills the block's non-anchor positions.
    var maskTokenId: Int { get }
    /// Target layers the drafter reads, in feature-concatenation order.
    var targetLayerIds: [Int] { get }
    /// The target shape the drafter was distilled for.
    var targetGeometry: DFlash2Geometry { get }
    /// Target hidden rows the drafter's context keeps.
    var contextWindow: Int { get }

    func makeState() -> DFlash2DrafterState

    /// Propose one block. `request.targetHidden` rows enter `state` as
    /// placeholders until the iterator resolves them.
    func propose(
        _ request: DFlash2ProposalRequest, target: any DFlash2TargetModel,
        state: inout DFlash2DrafterState
    ) -> DFlash2Proposal
}

// MARK: - Target

/// One verify pass over `[anchor, draft_1, ..., draft_{S-1}]`.
public struct DFlash2VerifyRequest {
    /// `[1, S]` token ids.
    public var tokens: MLXArray
    /// `[1]` int32 absolute position of `tokens[0]`; may be lazy.
    public var position: MLXArray
    /// Largest value `position` can take. Bounds the attention span.
    public var positionUpperBound: Int
    /// Layers whose outputs the drafter reads, in ``DFlash2VerifyResult/hidden`` order.
    public var captureLayers: [Int]

    public init(
        tokens: MLXArray, position: MLXArray, positionUpperBound: Int, captureLayers: [Int]
    ) {
        self.tokens = tokens
        self.position = position
        self.positionUpperBound = positionUpperBound
        self.captureLayers = captureLayers
    }
}

public struct DFlash2VerifyResult {
    /// `[1, S, vocab]`; row `i` predicts input position `i + 1`.
    public var logits: MLXArray
    /// `[1, S, hidden]` per requested capture layer.
    public var hidden: [MLXArray]
    /// One capture per gated-delta layer, in cache order.
    public var recurrentCaptures: [GatedDeltaCapture]

    public init(logits: MLXArray, hidden: [MLXArray], recurrentCaptures: [GatedDeltaCapture]) {
        self.logits = logits
        self.hidden = hidden
        self.recurrentCaptures = recurrentCaptures
    }
}

/// A hybrid attention / gated-delta target that DFlash2 can speculate for.
///
/// A verify pass never commits: it writes attention rows at the request's
/// position without moving any offset and returns recurrent state as captures.
/// The iterator commits once the accept count is known.
public protocol DFlash2TargetModel: LanguageModel {
    var dflash2Geometry: DFlash2Geometry { get }
    /// Embedding table the drafter borrows.
    var dflash2Embedding: Embedding { get }
    /// LM head the drafter borrows; `nil` when tied to the embedding.
    var dflash2Head: Linear? { get }

    /// Whether the verify pass can drive `cache`.
    func dflash2SupportsCache(_ cache: [KVCache]) -> Bool

    /// A prefill forward that also returns the outputs of `captureLayers`.
    func dflash2Prefill(
        _ tokens: MLXArray, cache: [KVCache], captureLayers: [Int]
    ) -> (logits: MLXArray, hidden: [MLXArray])

    func dflash2Verify(_ request: DFlash2VerifyRequest, cache: [KVCache]) -> DFlash2VerifyResult
}

// MARK: - Recurrent capture

/// What one gated-delta layer consumed in a verify pass: enough to recompute its
/// state as if only a prefix of the pass had run.
public struct GatedDeltaCapture {
    /// `concat([convState, qkv])`, `[1, K - 1 + S, convDim]`.
    public var convInput: MLXArray
    /// Normalized, scaled `q`/`k` (`[1, S, Hk, Dk]`) and `v` (`[1, S, Hv, Dv]`).
    public var q: MLXArray
    public var k: MLXArray
    public var v: MLXArray
    /// Raw gate projections, `[1, S, Hv]`.
    public var a: MLXArray
    public var b: MLXArray
    public var aLog: MLXArray
    public var dtBias: MLXArray
    /// Recurrent state before the pass.
    public var initialState: MLXArray

    public init(
        convInput: MLXArray, q: MLXArray, k: MLXArray, v: MLXArray,
        a: MLXArray, b: MLXArray, aLog: MLXArray, dtBias: MLXArray,
        initialState: MLXArray
    ) {
        self.convInput = convInput
        self.q = q
        self.k = k
        self.v = v
        self.a = a
        self.b = b
        self.aLog = aLog
        self.dtBias = dtBias
        self.initialState = initialState
    }

    /// The per-pass arrays, so a compiled verify body can return them as outputs.
    package var arrays: [MLXArray] { [convInput, q, k, v, a, b] }

    package init(arrays: [MLXArray], aLog: MLXArray, dtBias: MLXArray, initialState: MLXArray) {
        precondition(arrays.count == 6, "convInput, q, k, v, a, b")
        self.init(
            convInput: arrays[0], q: arrays[1], k: arrays[2], v: arrays[3], a: arrays[4],
            b: arrays[5], aLog: aLog, dtBias: dtBias, initialState: initialState)
    }

    /// The layer's recurrent and conv state after the first `validCount` positions.
    ///
    /// Positions past `validCount` are masked out of the scan, so the result is
    /// exact for any count and `validCount` may be a lazy `[]` int32 array.
    public func replay(validCount: MLXArray) -> (recurrent: MLXArray, conv: MLXArray) {
        let outputs = compiledGatedDeltaReplay([
            q, k, v, a, b, aLog, dtBias, initialState, convInput, validCount,
        ])
        return (outputs[0], outputs[1])
    }
}

/// One trace serves every layer: their shapes match.
private let compiledGatedDeltaReplay: @Sendable ([MLXArray]) -> [MLXArray] = compile { inputs in
    let (q, k, v) = (inputs[0], inputs[1], inputs[2])
    let (a, b, aLog, dtBias) = (inputs[3], inputs[4], inputs[5], inputs[6])
    let (state, convInput, validCount) = (inputs[7], inputs[8], inputs[9])
    let s = q.dim(1)
    let mask = (MLXArray(Int32(0) ..< Int32(s)) .< validCount.asType(.int32))
        .expandedDimensions(axis: 0)
    let (_, newState) = gatedDeltaUpdate(
        q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: state, mask: mask)
    // The conv state after `validCount` positions: rows [validCount, validCount + K - 1).
    let kernelRows = convInput.dim(1) - s
    let rows = (validCount.asType(.int32) + MLXArray(Int32(0) ..< Int32(kernelRows)))
        .reshaped([1, kernelRows, 1])
    let conv = contiguous(takeAlong(convInput, rows, axis: 1))
    return [newState, conv]
}

// MARK: - Keyed traces

/// Compiled traces built on demand, one per key. `Module.invalidateCompiledTraces()`
/// finds it by reflection, so replacing modules drops every trace.
package final class KeyedCompiledTraces<Key: Hashable, Owner: Module>: CompiledTraceInvalidating {
    private let lock = NSLock()
    private var traces: [Key: CompiledTrace<Owner>] = [:]

    package init() {}

    package var count: Int { lock.withLock { traces.count } }

    package func trace(_ key: Key, make: () -> CompiledTrace<Owner>) -> CompiledTrace<Owner> {
        lock.withLock {
            if let existing = traces[key] { return existing }
            let trace = make()
            traces[key] = trace
            return trace
        }
    }

    package func invalidate() {
        lock.withLock { traces.removeAll() }
    }
}

// MARK: - Attention cache rows

extension KVCacheSimple {
    /// Write `S` rows at `position` (`[1]` int32, possibly lazy) without moving
    /// `offset`, and return the first `visibleLength` rows. Rows past the
    /// committed offset are scratch for a later write to overwrite.
    package func writeRows(
        keys newKeys: MLXArray, values newValues: MLXArray,
        position: MLXArray, visibleLength: Int
    ) -> (MLXArray, MLXArray) {
        if keys == nil || visibleLength > keys!.dim(2) {
            let capacity = (visibleLength + step - 1) / step * step
            let kShape = [newKeys.dim(0), newKeys.dim(1), capacity, newKeys.dim(3)]
            let vShape = [newValues.dim(0), newValues.dim(1), capacity, newValues.dim(3)]
            let grownKeys = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let grownValues = MLXArray.zeros(vShape, dtype: newValues.dtype)
            if let keys, let values {
                // Keep the whole buffer: scratch rows may belong to a pass in flight.
                self.keys = concatenated(
                    [keys, grownKeys[.ellipsis, keys.dim(2)..., 0...]], axis: 2)
                self.values = concatenated(
                    [values, grownValues[.ellipsis, values.dim(2)..., 0...]], axis: 2)
            } else {
                keys = grownKeys
                values = grownValues
            }
        }
        let rows = (position.asType(.int32) + MLXArray(Int32(0) ..< Int32(newKeys.dim(2))))
            .reshaped([1, 1, -1, 1])
        keys = putAlong(keys!, rows, values: newKeys, axis: 2)
        values = putAlong(values!, rows, values: newValues, axis: 2)
        return (
            keys![.ellipsis, ..<visibleLength, 0...],
            values![.ellipsis, ..<visibleLength, 0...]
        )
    }

    /// The cache now holds `count` positions written by ``writeRows(keys:values:position:visibleLength:)``.
    package func commitRows(count: Int) {
        offset = count
    }
}

// MARK: - Drafter context cache

/// Sliding-window cache of one drafter layer's context keys and values.
///
/// A pipelined round appends the whole verify block before its accept count is
/// known, so rows enter as placeholders with explicit positions and
/// `resolve(newest:valid:)` commits the accepted prefix. Rows are compacted to
/// the newest valid `window` rows once every few hundred appends; until then
/// the attention mask hides placeholders and rows past the window.
public final class DFlash2ContextCache {
    public let window: Int

    private var keyStore: MLXArray?
    private var valueStore: MLXArray?
    private var storedCount = 0
    private let compactionSlack = 256

    /// Absolute position of each stored row.
    package private(set) var rowPositions: [Int32] = []
    /// Whether each stored row is committed context or a placeholder.
    package private(set) var rowValid: [Bool] = []

    public init(window: Int) {
        precondition(window > 0, "DFlash2ContextCache needs a positive window")
        self.window = window
    }

    /// Stored rows, placeholders included.
    public var count: Int { storedCount }

    package var keys: MLXArray? { keyStore?[.ellipsis, ..<storedCount, 0...] }
    package var values: MLXArray? { valueStore?[.ellipsis, ..<storedCount, 0...] }

    /// Append `[1, heads, n, dim]` rows as placeholders at `positions`, and return
    /// the stored keys and values.
    @discardableResult
    package func append(
        keys newKeys: MLXArray, values newValues: MLXArray, positions: [Int32]
    ) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(positions.count == n, "one position per appended row")
        if keyStore == nil {
            let capacity = Swift.max(n, window + compactionSlack)
            keyStore = MLXArray.zeros(
                [newKeys.dim(0), newKeys.dim(1), capacity, newKeys.dim(3)], dtype: newKeys.dtype)
            valueStore = MLXArray.zeros(
                [newValues.dim(0), newValues.dim(1), capacity, newValues.dim(3)],
                dtype: newValues.dtype)
        }
        if storedCount + n > keyStore!.dim(2) || storedCount > window + compactionSlack {
            compact(reserving: n)
        }
        keyStore![.ellipsis, storedCount ..< (storedCount + n), 0...] = newKeys
        valueStore![.ellipsis, storedCount ..< (storedCount + n), 0...] = newValues
        storedCount += n
        rowPositions.append(contentsOf: positions)
        rowValid.append(contentsOf: Array(repeating: false, count: n))
        return (keys!, values!)
    }

    /// Commit the first `valid` of the newest `newest` rows. The rest stay
    /// placeholders and drop at the next compaction.
    package func resolve(newest: Int, valid: Int) {
        precondition(newest <= storedCount, "resolving more rows than stored")
        let base = storedCount - newest
        for i in 0 ..< newest {
            rowValid[base + i] = i < valid
        }
    }

    /// Keep the newest `window` valid rows in a fresh padded buffer.
    private func compact(reserving n: Int) {
        let kept = (0 ..< storedCount).filter { rowValid[$0] }.suffix(window)
        let gather = MLXArray(kept.map { Int32($0) })
        let keptKeys = MLX.take(keyStore!, gather, axis: 2)
        let keptValues = MLX.take(valueStore!, gather, axis: 2)
        let capacity = Swift.max(window + compactionSlack, kept.count + n)
        let padK = [keptKeys.dim(0), keptKeys.dim(1), capacity - kept.count, keptKeys.dim(3)]
        let padV = [
            keptValues.dim(0), keptValues.dim(1), capacity - kept.count, keptValues.dim(3),
        ]
        keyStore = concatenated([keptKeys, MLXArray.zeros(padK, dtype: keptKeys.dtype)], axis: 2)
        valueStore = concatenated(
            [keptValues, MLXArray.zeros(padV, dtype: keptValues.dtype)], axis: 2)
        rowPositions = kept.map { rowPositions[$0] }
        rowValid = Array(repeating: true, count: kept.count)
        storedCount = kept.count
    }
}
