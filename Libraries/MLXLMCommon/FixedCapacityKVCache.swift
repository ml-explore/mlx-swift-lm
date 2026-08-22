// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// A fixed-capacity key/value cache for models that conform to
/// ``FixedCapacityKVCacheProviding``.
///
/// ## Why this cache exists
///
/// ``KVCacheSimple`` tracks its write position with a Swift `Int` and uses it as
/// the bounds of array slices inside `update(keys:values:)`. When a model step is
/// traced for `MLX.compile(inputs:outputs:shapeless:_:)`,
/// those integers are embedded in the graph as *constants*: the compiled step keeps
/// writing into the slot and attending over the window captured at trace time, so
/// decode falls into repeating token loops (issue #406).
///
/// For compatible models, ``FixedCapacityKVCache`` removes every position-dependent
/// Swift integer from the traced step:
///
/// - buffers are allocated once at full capacity, so shapes never change across
///   decode steps and the compiled graph is reused;
/// - the write position is an `MLXArray` carried through ``innerState()`` and
///   therefore threaded through compile as an input *and* an output;
/// - the KV write, RoPE offset and attention mask are all graph operations
///   derived from that position array.
///
/// ## Usage
///
/// ```swift
/// let session = try CompiledDecodeSession(
///     model: cacheProvider,
///     prompt: promptTokens,
///     capacity: promptTokens.dim(1) + maxNewTokens)
/// var token = nextToken(from: session.prefillLogits)
/// while session.remainingCapacity > 0 {
///     let logits = try session.step(token.reshaped(1, 1))
///     token = nextToken(from: logits)
/// }
/// ```
///
/// ## Contracts
///
/// - Prefer ``CompiledDecodeSession`` over constructing the cache directly. It
///   performs eager prefill before tracing and enforces capacity on the host.
/// - Low-level callers must prefill before the first compiled call and must not
///   invoke the model once the cache reaches capacity.
/// - One write position is tracked, so every row of a batch shares it:
///   batch 1, or same-length rows.
/// - The attention read spans the full capacity with unwritten slots masked out,
///   which trades a little attention compute for static shapes.
///
/// Use this cache only for models conforming to ``FixedCapacityKVCacheProviding``.
/// Those models have been audited to keep position-dependent work in MLX graph
/// values and to use a standard full-attention cache for every layer. The cache
/// is also correct when used eagerly, but ``KVCacheSimple`` is faster there
/// because it attends only over written slots.
public final class FixedCapacityKVCache: KVCache, Updatable {

    /// The capacity, in tokens, of the key/value buffers.
    public let maxTokens: Int

    private var keys: MLXArray?
    private var values: MLXArray?

    /// The write position as a `[1]` int32 array.
    ///
    /// This is the piece of state ``KVCacheSimple`` keeps as a Swift `Int`; as an
    /// array it is part of ``innerState()`` and advances through compiled calls
    /// instead of being baked into the graph as a constant.
    private var position: MLXArray?

    public init(maxTokens: Int) {
        precondition(maxTokens > 0, "FixedCapacityKVCache.maxTokens must be positive")
        precondition(
            maxTokens <= Int(Int32.max),
            "FixedCapacityKVCache.maxTokens must fit in an Int32")
        self.maxTokens = maxTokens
    }

    // MARK: - KVCache

    public var offset: Int {
        guard let position else { return 0 }
        return Int(position.item(Int32.self))
    }

    /// Array-valued offset so rotary position embeddings participate in the
    /// compiled graph instead of being read as a Swift `Int` constant.
    ///
    /// The `+ 0` snapshots the position before ``update(keys:values:)`` advances
    /// it, mirroring ``BatchPositionedKVCache``.
    public var ropeOffset: RoPEOffset {
        .batch((position ?? MLXArray([Int32(0)])) + 0)
    }

    public var maxSize: Int? { maxTokens }

    public var isTrimmable: Bool { true }

    /// Full-capacity buffers return unwritten slots as well as written ones, so
    /// every call -- single-token decode included -- must be masked; the shared
    /// mask helpers delegate to ``makeMask(n:windowSize:returnArray:)`` when
    /// this is `true`.
    public var requiresAttentionMask: Bool { true }

    @discardableResult
    public func update(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let n = newKeys.dim(2)
        precondition(n > 0, "FixedCapacityKVCache.update requires a non-empty sequence")
        precondition(
            n <= maxTokens,
            "FixedCapacityKVCache capacity \(maxTokens) cannot hold a chunk of \(n) tokens")

        if keys == nil {
            allocate(with: newKeys, values: newValues)
        }
        let writePosition = position!

        // Compile-safe scatter: the write indices are graph values derived from
        // the threaded position. Unlike a dynamic Swift slice, putAlong remains
        // dynamic when the compiled graph is reused, and it updates only the new
        // rows rather than expanding them across the full capacity.
        let writeIndices =
            (writePosition + MLXArray(Int32(0) ..< Int32(n)))
            .reshaped(1, 1, n, 1)
        keys = putAlong(keys!, writeIndices, values: newKeys, axis: 2)
        values = putAlong(values!, writeIndices, values: newValues, axis: 2)

        position = writePosition + n

        return (keys!, values!)
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // Unwritten slots must never be attended, so the answer is always a
        // material mask built from the position array -- never a symbolic mode,
        // regardless of `returnArray`.
        let currentPosition = position ?? MLXArray([Int32(0)])
        let keySlots = MLXArray(Int32(0) ..< Int32(maxTokens)).reshaped(1, 1, 1, maxTokens)
        let queryPositions =
            currentPosition
            + MLXArray(Int32(0) ..< Int32(n)).reshaped(
                1, 1, n, 1)

        var mask = keySlots .<= queryPositions
        if let windowSize {
            mask = mask & (keySlots .> (queryPositions - windowSize))
        }
        return .array(mask)
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        guard n > 0, position != nil else { return 0 }
        let current = offset
        let trimmed = min(current, n)
        position = MLXArray([Int32(current - trimmed)])
        return trimmed
    }

    public func copy() -> any KVCache {
        let new = FixedCapacityKVCache(maxTokens: maxTokens)
        new.keys = keys?[.ellipsis]
        new.values = values?[.ellipsis]
        new.position = position.map { $0[.ellipsis] }
        return new
    }

    public var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            let used = offset
            if used == maxTokens {
                return [keys, values]
            }
            return [
                keys[.ellipsis, ..<used, 0...],
                values[.ellipsis, ..<used, 0...],
            ]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("FixedCapacityKVCache state must have exactly 2 arrays (keys, values)")
            }
            var newKeys = newValue[0]
            var newValues = newValue[1]
            let used = newKeys.dim(2)
            precondition(
                newKeys.dim(0) == newValues.dim(0) && newKeys.dim(1) == newValues.dim(1)
                    && used == newValues.dim(2),
                "FixedCapacityKVCache key/value state must have matching batch, head, and sequence dimensions"
            )
            precondition(
                used <= maxTokens,
                "FixedCapacityKVCache capacity \(maxTokens) cannot hold state of \(used) tokens")

            // Restore the fixed-capacity shape so the cache stays usable inside
            // compiled traces after a round-trip through serialization.
            if used < maxTokens {
                var keyPadShape = newKeys.shape
                keyPadShape[2] = maxTokens - used
                var valuePadShape = newValues.shape
                valuePadShape[2] = maxTokens - used
                newKeys = concatenated(
                    [newKeys, MLXArray.zeros(keyPadShape, dtype: newKeys.dtype)], axis: 2)
                newValues = concatenated(
                    [newValues, MLXArray.zeros(valuePadShape, dtype: newValues.dtype)], axis: 2)
            }
            keys = newKeys
            values = newValues
            position = MLXArray([Int32(used)])
        }
    }

    public var metaState: [String] {
        get { ["\(maxTokens)"] }
        set {
            guard newValue.count == 1, Int(newValue[0]) == maxTokens else {
                fatalError(
                    "FixedCapacityKVCache meta_state must be [\"\\(maxTokens)\"] for this cache")
            }
        }
    }

    /// The position array rides in ``innerState()`` with the buffers, which is
    /// what lets `MLX.compile(inputs:outputs:shapeless:_:)`
    /// thread it through the graph. Also satisfies `Evaluatable`.
    public func innerState() -> [MLXArray] {
        guard let keys, let values, let position else { return [] }
        return [keys, values, position]
    }

    // MARK: - Private

    private func allocate(with newKeys: MLXArray, values newValues: MLXArray) {
        let batch = newKeys.dim(0)
        let kvHeads = newKeys.dim(1)
        let kHeadDim = newKeys.dim(3)
        let vHeadDim = newValues.dim(3)

        keys = MLXArray.zeros(
            [batch, kvHeads, maxTokens, kHeadDim], dtype: newKeys.dtype)
        values = MLXArray.zeros(
            [batch, kvHeads, maxTokens, vHeadDim], dtype: newValues.dtype)
        position = MLXArray([Int32(0)])
    }
}

/// A model that can construct a fixed-capacity cache for its forward pass.
///
/// This is a behavioral capability rather than a marker for a particular
/// execution mechanism. A conforming model promises that every
/// position-dependent operation can consume graph-valued cache positions and
/// that this factory returns the correct cache topology. Sliding-window,
/// recurrent, hybrid, and model-specific layouts must provide their own
/// implementation or omit the conformance.
public protocol FixedCapacityKVCacheProviding: LanguageModel {
    /// Create a fixed-capacity cache suitable for graph-reused decoding.
    func newFixedCapacityCache(maxTokens: Int) throws -> [FixedCapacityKVCache]
}

extension FixedCapacityKVCacheProviding where Self: KVCacheDimensionProvider {
    /// Create one fixed-capacity compiled cache per attention layer.
    ///
    /// `maxTokens` is the total sequence capacity, including the prompt and all
    /// generated tokens, rather than a rotating resident-window size.
    public func newFixedCapacityCache(maxTokens: Int) throws -> [FixedCapacityKVCache] {
        guard maxTokens > 0, maxTokens <= Int(Int32.max) else {
            throw KVCacheConfigurationError.invalidCapacity(maxTokens)
        }
        return kvHeads.map { _ in FixedCapacityKVCache(maxTokens: maxTokens) }
    }
}
