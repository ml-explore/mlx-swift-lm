// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// DFlash2 (https://inco.ai/blog/dflash2/, reference implementation
// https://github.com/z-lab/dflash) drafts a whole block of tokens in one
// parallel pass from the target model's hidden states, then the target
// verifies the block in one forward pass. These protocols are what the
// ``DFlash2SpeculativeTokenIterator`` needs from the two models; the concrete
// drafter (`DFlash2DraftModel`) and target (`Qwen35TextModel`) live in MLXLLM.

// MARK: - Drafter

/// One block proposal: the selector's token path and the per-position
/// candidates it chose from.
public struct DFlash2Proposal {
    /// Drafted token ids, `[1, blockSize - 1]`.
    public var tokens: MLXArray
    /// Top-K candidate ids per drafted position, `[1, blockSize - 1, K]`.
    public var candidates: MLXArray
    /// Selection probability over `candidates` per position, present only
    /// when the proposal was sampled (`temperature > 0`).
    public var probabilities: MLXArray?

    public init(tokens: MLXArray, candidates: MLXArray, probabilities: MLXArray? = nil) {
        self.tokens = tokens
        self.candidates = candidates
        self.probabilities = probabilities
    }
}

/// Per-stream drafter state, owned by the iterator and passed to the drafter
/// on every proposal. Drafter instances hold no per-stream state, so one
/// drafter serves many iterators.
public struct DFlash2DrafterState {
    /// One sliding-window context cache per drafter layer.
    public var contextCaches: [DFlash2ContextCache]

    public init(contextCaches: [DFlash2ContextCache]) {
        self.contextCaches = contextCaches
    }
}

/// A DFlash2 block-parallel drafter.
///
/// The drafter attends over projections of the target's hidden states (its
/// context) and predicts every position of a `[anchor, MASK, ...]` block at
/// once. It borrows the target's embedding table and LM head per call and is
/// stateless with respect to the target, like ``MTPDrafterModel``.
public protocol DFlash2DrafterModel: BaseLanguageModel {
    /// Tokens per verify pass the checkpoint was trained for (anchor included).
    var blockSize: Int { get }
    /// Token id filling the block's non-anchor positions.
    var maskTokenId: Int { get }
    /// Target layers whose outputs the drafter reads, in feature-concat order.
    var targetLayerIds: [Int] { get }
    /// Depth of the target the drafter was distilled for.
    var targetLayerCount: Int { get }
    /// Target hidden rows the drafter's context window retains.
    var contextWindow: Int { get }

    /// Fresh per-stream state.
    func makeState() -> DFlash2DrafterState

    /// Propose one block.
    ///
    /// Every accept-dependent input may be a lazy array, so the proposal can
    /// be built while the previous verify pass is still running on the GPU.
    ///
    /// - Parameters:
    ///   - block: `[1, blockSize]` token ids; position 0 is the anchor.
    ///   - targetHidden: `[1, S, targetLayerIds.count * hidden]` target
    ///     outputs for the `S` positions verified since the last proposal.
    ///   - contextPosition: absolute position of `targetHidden` row 0.
    ///   - validRows: `[]` int32 count of leading `targetHidden` rows that
    ///     are committed context; the rest are masked out.
    ///   - temperature: 0 selects greedily; above 0 samples the path and
    ///     fills ``DFlash2Proposal/probabilities``.
    ///   - target: the model whose embedding and head the drafter borrows.
    ///   - state: the stream's context caches; `targetHidden` rows are
    ///     appended as placeholders until `resolve(newest:valid:)`.
    func propose(
        block: MLXArray,
        targetHidden: MLXArray,
        contextPosition: Int,
        validRows: MLXArray,
        temperature: Float,
        target: any DFlash2TargetModel,
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
    /// Largest value `position` can resolve to. Bounds the attention span.
    public var positionUpperBound: Int
    /// Layers whose outputs the drafter needs, in ``DFlash2VerifyResult/hidden`` order.
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
    /// One capture per gated-delta layer, in cache order, so the iterator can
    /// rewind recurrent state to any accepted prefix.
    public var recurrentCaptures: [GatedDeltaCapture]

    public init(logits: MLXArray, hidden: [MLXArray], recurrentCaptures: [GatedDeltaCapture]) {
        self.logits = logits
        self.hidden = hidden
        self.recurrentCaptures = recurrentCaptures
    }
}

/// A hybrid attention/recurrent target that a DFlash2 drafter can speculate for.
///
/// A verify pass computes; it never commits. Attention rows are written into
/// the cache buffers at the request's (possibly lazy) position without moving
/// any offset, and recurrent state is returned as captures. The iterator
/// commits once the accept count is known.
public protocol DFlash2TargetModel: LanguageModel {
    /// Decoder layer count, checked against the drafter's target geometry.
    var dflash2LayerCount: Int { get }
    /// Embedding table the drafter borrows for the block.
    var dflash2Embedding: Embedding { get }
    /// LM head the drafter borrows; nil when tied to the embedding.
    var dflash2Head: Linear? { get }

    /// Whether the verify pass can drive this cache: plain `KVCacheSimple`
    /// attention entries and `MambaCache` recurrent entries.
    func dflash2SupportsCache(_ cache: [KVCache]) -> Bool

    /// Ordinary prefill over `tokens` that also returns the outputs of
    /// `captureLayers` (`[1, S, hidden]` each, in request order).
    func dflash2Prefill(
        _ tokens: MLXArray, cache: [KVCache], captureLayers: [Int]
    ) -> (logits: MLXArray, hidden: [MLXArray])

    /// The verify pass. See ``DFlash2VerifyRequest`` and ``DFlash2VerifyResult``.
    func dflash2Verify(_ request: DFlash2VerifyRequest, cache: [KVCache]) -> DFlash2VerifyResult
}

// MARK: - Recurrent capture

/// What one gated-delta layer consumed during a verify pass, enough to
/// recompute its state as if only a prefix of the pass had run.
public struct GatedDeltaCapture {
    /// `concat([convState, qkv])`: the conv input, `[1, K - 1 + S, convDim]`.
    public var convInput: MLXArray
    /// Post-norm, post-scale `q`/`k` (`[1, S, Hk, Dk]`) and `v` (`[1, S, Hv, Dv]`).
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

    /// The capture's per-pass arrays, in ``init(arrays:aLog:dtBias:initialState:)``
    /// order, so a compiled verify body can return them as outputs.
    package var arrays: [MLXArray] { [convInput, q, k, v, a, b] }

    package init(arrays: [MLXArray], aLog: MLXArray, dtBias: MLXArray, initialState: MLXArray) {
        precondition(arrays.count == 6, "convInput, q, k, v, a, b")
        self.init(
            convInput: arrays[0], q: arrays[1], k: arrays[2], v: arrays[3], a: arrays[4],
            b: arrays[5], aLog: aLog, dtBias: dtBias, initialState: initialState)
    }

    /// The layer's state after the first `validCount` positions of the pass.
    ///
    /// Replays every position with the steps past `validCount` masked out.
    /// A masked step leaves the scan state untouched, so the result equals
    /// the accepted-prefix replay for every count, and `validCount` may be a
    /// lazy `[]` int32 array. Returns the recurrent and conv states.
    public func replay(validCount: MLXArray) -> (recurrent: MLXArray, conv: MLXArray) {
        let outputs = compiledGatedDeltaReplay([
            q, k, v, a, b, aLog, dtBias, initialState, convInput, validCount,
        ])
        return (outputs[0], outputs[1])
    }
}

/// One trace serves every layer: shapes match across a model's gated-delta
/// layers, and the elementwise work around the scan fuses into a few launches.
private let compiledGatedDeltaReplay: @Sendable ([MLXArray]) -> [MLXArray] = compile { inputs in
    let (q, k, v) = (inputs[0], inputs[1], inputs[2])
    let (a, b, aLog, dtBias) = (inputs[3], inputs[4], inputs[5], inputs[6])
    let (state, convInput, validCount) = (inputs[7], inputs[8], inputs[9])
    let s = q.dim(1)
    let mask = (MLXArray(Int32(0) ..< Int32(s)) .< validCount.asType(.int32))
        .expandedDimensions(axis: 0)
    let (_, newState) = gatedDeltaUpdate(
        q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: state, mask: mask)
    // Conv state after `validCount` positions: rows [validCount, validCount + K - 1).
    let kernelRows = convInput.dim(1) - s
    let rows = (validCount.asType(.int32) + MLXArray(Int32(0) ..< Int32(kernelRows)))
        .reshaped([1, kernelRows, 1])
    let conv = contiguous(takeAlong(convInput, rows, axis: 1))
    return [newState, conv]
}

// MARK: - Attention cache rows

extension KVCacheSimple {
    /// Write `S` rows at `position` (a `[1]` int32 array, possibly lazy)
    /// without moving `offset`, and return the first `visibleLength` rows for
    /// the pass's attention. Rows past the committed offset are scratch: a
    /// later write at a smaller position simply overwrites them.
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

    /// Commit rows written by ``writeRows(keys:values:position:visibleLength:)``:
    /// the cache now holds `count` positions.
    package func commitRows(count: Int) {
        offset = count
    }
}

// MARK: - Drafter context cache

/// Sliding-window cache of one drafter layer's context keys and values.
///
/// Rows enter as placeholders with explicit positions, because a pipelined
/// round appends the whole verify block before the accept count is known;
/// `resolve(newest:valid:)` commits the accepted prefix once it is. The
/// window is enforced lazily: rows are stored in a padded buffer and
/// compacted to the newest valid `window` rows once per few hundred appends,
/// while the attention mask windows by distance and hides placeholders, so
/// what attention sees is exactly a trimmed cache.
public final class DFlash2ContextCache {
    /// Rows the window retains.
    public let window: Int

    private var keyStore: MLXArray?
    private var valueStore: MLXArray?
    private var storedCount = 0
    private let compactionSlack = 256

    /// Absolute position each stored row was written with.
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

    /// Append rows (`[1, heads, n, dim]`) as placeholders at `positions` and
    /// return the stored keys and values.
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

    /// Commit the first `valid` of the newest `newest` rows; the rest stay
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
