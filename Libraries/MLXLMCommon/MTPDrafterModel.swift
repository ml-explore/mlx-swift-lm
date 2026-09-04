// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A target model cannot provide the architecture-specific state required by
/// an MTP drafter.
public enum MTPDrafterCompatibilityError: Error, Sendable, Equatable, LocalizedError {
    case incompatibleTarget(drafter: String, expected: String, actual: String)

    public var errorDescription: String? {
        switch self {
        case .incompatibleTarget(let drafter, let expected, let actual):
            "\(drafter) requires \(expected); received \(actual)."
        }
    }
}

/// Protocol for Multi-Token Prediction (MTP) speculative drafter models.
///
/// Mirrors `EmbeddingModel`'s relationship to `BaseLanguageModel`: this
/// protocol refines `BaseLanguageModel` with drafter-specific surface, so
/// implementations inherit weight loading and `sanitize` hooks while defining
/// their own forward signature.
///
/// MTP drafters do **not** conform to `LanguageModel` — their I/O contract is
/// different: a drafter consumes the target's last hidden state and per
/// layer-type shared K/V, produces a block of K-1 candidate tokens in a
/// single call, and does not store transient round-state on the model object.
/// The `MTPSpeculativeTokenIterator` extracts the shared K/V from the
/// target's `LMOutput.state` and threads it to the drafter as a method
/// argument.
///
/// Conforming types are expected to be stateless with respect to the target
/// model: every per-round input — including the target itself — flows through
/// `draftBlock(...)` as a method parameter. This makes drafter instances safe
/// to share across iterators without per-iterator mutable state on the model
/// instance. Drafters that need their own per-stream state additionally
/// conform to ``StatefulMTPDrafterModel``.
public protocol MTPDrafterModel: BaseLanguageModel {
    /// Largest total verification block the drafter can produce efficiently.
    /// `nil` means the caller may choose any block size.
    var maximumBlockSize: Int? { get }

    /// Whether this drafter consumes target-emitted shared K/V. Qwen owns a
    /// private full-attention cache and therefore only needs target hidden
    /// states; Gemma-style assistants consume the shared target K/V directly.
    var requiresSharedTargetKV: Bool { get }

    /// Whether the drafter must be prefilled over the shifted prompt before
    /// its first proposal. Qwen's private decoder cache requires this.
    var requiresPromptPrefill: Bool { get }

    /// Whether this drafter is currently supported only for greedy decoding.
    /// This keeps stochastic generation on the ordinary target path until a
    /// probability-ratio acceptance sampler is available.
    var requiresGreedySampling: Bool { get }

    /// Validate that `target` provides the architecture-specific state and
    /// model structure required by this drafter.
    ///
    /// Callers must invoke this before prefill or drafting. Implementations
    /// should keep this check side-effect free and throw
    /// ``MTPDrafterCompatibilityError`` for unsupported pairings.
    func validateCompatibility(with target: any LanguageModel) throws

    /// K-step drafting from a constant position.
    ///
    /// Returns the proposed tokens as a `[B, blockSize - 1]` MLXArray. The
    /// drafter holds no transient round-state on the model instance — every
    /// per-round input is threaded as a method argument.
    ///
    /// - Parameters:
    ///   - target: The main language model this drafter is speculating for.
    ///     Used to look up target-derived constants (input embeddings, embed
    ///     scale, etc.) inline per round; conformers must not retain or
    ///     mutate references derived from `target`.
    ///   - lastToken: Bonus token from the target's last verify pass, shape `[B]`.
    ///   - lastHidden: Target's last hidden state, shape `[B, 1, backbone_hidden_size]`.
    ///   - sharedKV: Dict keyed by `layer_type` (`"full_attention"` /
    ///     `"sliding_attention"`) mapping to `(keys, values)` `MLXArray`s for
    ///     the last layer of that layer-type in the target.
    ///   - positionDeltas: Optional target-emitted position delta state for
    ///     multimodal RoPE models whose continuation positions are not
    ///     derivable from cache length alone.
    ///   - queryOffset: Constant absolute position for the round (the
    ///     position the bonus token sits at in the target's KV cache).
    ///     Passed as a Swift `Int` rather than an `MLXArray` to avoid the
    ///     `.item()` round-trip that would otherwise stall the GPU per
    ///     speculation round.
    ///   - blockSize: Total tokens in the round (the drafter returns
    ///     `blockSize - 1`; the bonus token is implicit).
    ///   - sampler: `LogitSampler` to apply to each step's logits.
    /// - Returns: `[B, blockSize - 1]` token array.
    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray
}

extension MTPDrafterModel {
    public var maximumBlockSize: Int? { nil }
    public var requiresSharedTargetKV: Bool { true }
    public var requiresPromptPrefill: Bool { false }
    public var requiresGreedySampling: Bool { false }
}

/// Target-side capability for rewinding an in-place speculative verify pass.
///
/// Rollback belongs to the target/cache implementation, not to the drafter.
/// The depth is explicit so a future multi-token drafter cannot accidentally
/// select an in-place path whose recurrent checkpoint is too shallow.
public protocol SpeculativeCacheRewindModel {
    var maximumNativeTargetCacheRewind: Int { get }
}

/// Per-stream state for MTP drafters that need their own transient storage.
///
/// The state is intentionally separate from ``LMOutput/State``: `LMOutput`
/// belongs to the target model's step output, while this value is owned by the
/// MTP iterator and passed only to the drafter.
public struct MTPDrafterState {
    public var cache: [KVCache]

    /// Absolute next position in a drafter-owned autoregressive cache.
    public var nextPosition: Int

    /// A proposal already computed while committing the previous verified
    /// round. Qwen uses this to avoid advancing its cache twice.
    public var seedToken: MLXArray?
    public var seedHidden: MLXArray?

    /// Number of tentative cache entries appended by the current proposal.
    public var proposalAppended: Int

    public init(
        cache: [KVCache],
        nextPosition: Int = 0,
        seedToken: MLXArray? = nil,
        seedHidden: MLXArray? = nil,
        proposalAppended: Int = 0
    ) {
        self.cache = cache
        self.nextPosition = nextPosition
        self.seedToken = seedToken
        self.seedHidden = seedHidden
        self.proposalAppended = proposalAppended
    }
}

/// Optional capability for MTP drafters that maintain per-stream state.
///
/// This keeps the base ``MTPDrafterModel`` surface minimal for stateless
/// drafters such as Gemma 4 assistants while giving Qwen-style MTP predictors
/// an iterator-owned state that can carry and trim their private KV cache. The
/// state must not be stored on the model instance because drafter models can be
/// shared across concurrent generation streams.
public protocol StatefulMTPDrafterModel: MTPDrafterModel {
    /// Create per-stream state owned and trimmed by the iterator.
    func makeState(parameters: GenerateParameters?) -> MTPDrafterState

    /// Prefill drafter-owned state from the target's prompt hidden states.
    func prepareDrafterState(
        target: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    )

    /// Stateful variant of ``MTPDrafterModel/draftBlock(target:lastToken:lastHidden:sharedKV:positionDeltas:queryOffset:blockSize:sampler:)``.
    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray

    /// Reconcile tentative proposal writes with the sequence accepted by the
    /// target, then seed the next proposal if the architecture supports it.
    func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    )
}

/// Optional capability for stateful MTP drafters whose private state can be
/// normalized at a generation boundary and resumed from a later prompt suffix.
///
/// `ChatSession` deliberately does not know how a drafter's cache is shifted,
/// which seed values are speculative, or how target hidden states must be
/// paired with suffix tokens. The MTP iterator owns that lifecycle and calls
/// this interface at the model seam. Stateful drafters that do not conform are
/// still usable for cold generations; a warm target cache continues without
/// speculation rather than combining it with a fresh drafter state.
public protocol ResumableMTPDrafterModel: StatefulMTPDrafterModel {
    /// Convert `state` to a canonical boundary after the target cache has
    /// discarded any verified-but-unemitted lookahead.
    ///
    /// Return `false` when the private state cannot be proven to represent the
    /// supplied target position. The caller then keeps the target cache and
    /// drops only the MTP continuation.
    func finalizeDrafterState(
        target: any LanguageModel,
        targetBoundaryHidden: MLXArray,
        targetProcessedTokenCount: Int,
        discardedTargetTokens: Int,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState
    ) -> Bool

    /// Extend a canonical boundary over a prompt suffix and prime the next
    /// speculative round.
    ///
    /// `targetBoundaryHidden` is the hidden state of the final token already
    /// represented by the warm target cache. `suffixTargetHidden` contains one
    /// row per newly evaluated suffix token, and `firstBonus` is the token
    /// sampled after that suffix.
    func resumeDrafterState(
        target: any LanguageModel,
        suffixTokens: MLXArray,
        suffixTargetHidden: MLXArray,
        targetBoundaryHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> Bool
}

/// Normalize a shifted drafter cache at a resumable target boundary.
///
/// Shifted MTP drafters pair each input token with the target hidden state from
/// the preceding position. At a generation boundary their private cache must
/// therefore end one position before the target cache so the saved boundary
/// hidden state can bridge the next prompt suffix.
package func finalizeShiftedMTPDrafterState(
    targetProcessedTokenCount: Int,
    discardedTargetTokens: Int,
    state: inout MTPDrafterState
) -> Bool {
    let trim = discardedTargetTokens + 1
    guard state.proposalAppended == 0,
        targetProcessedTokenCount > 0,
        state.nextPosition == targetProcessedTokenCount + discardedTargetTokens,
        state.nextPosition >= trim,
        trimPromptCache(state.cache, numTokens: trim) == trim
    else { return false }

    state.nextPosition -= trim
    state.seedToken = nil
    state.seedHidden = nil
    return state.nextPosition == targetProcessedTokenCount - 1
        && state.cache.allSatisfy { $0.offset == state.nextPosition }
}

/// Iterator-owned MTP continuation carried atomically with a ChatSession's
/// target cache, target model state, and transcript.
///
/// `targetProcessedTokenCount` is only an alignment witness. The authoritative
/// position remains `KVCacheStorage.processedTokenCount`.
package struct MTPDrafterContinuation {
    package var state: MTPDrafterState
    package let targetBoundaryHidden: MLXArray
    package let targetProcessedTokenCount: Int

    package init(
        state: consuming MTPDrafterState,
        targetBoundaryHidden: MLXArray,
        targetProcessedTokenCount: Int
    ) {
        self.state = consume state
        self.targetBoundaryHidden = targetBoundaryHidden
        self.targetProcessedTokenCount = targetProcessedTokenCount
    }
}

extension StatefulMTPDrafterModel {
    public func prepareDrafterState(
        target _: any LanguageModel,
        promptTokens _: MLXArray,
        targetHidden _: MLXArray,
        firstBonus _: MLXArray,
        positionDeltas _: MLXArray?,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {}

    public func commitDrafterState(
        target _: any LanguageModel,
        targetHidden _: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken _: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        let rejected = draftTokens.dim(-1) - acceptedCount
        if rejected > 0 {
            trimPromptCache(state.cache, numTokens: rejected)
        }
    }
}

/// Lightweight context for an MTP drafter — simpler than `ModelContext`
/// because drafters have no tokenizer, no user input processor, no chat
/// template.
///
/// Not `Sendable`; cross-domain access goes through ``MTPDrafterContainer``.
public struct MTPDrafterContext {
    public var configuration: ModelConfiguration
    public var model: any MTPDrafterModel

    public init(configuration: ModelConfiguration, model: any MTPDrafterModel) {
        self.configuration = configuration
        self.model = model
    }
}

/// Sendable container for an ``MTPDrafterContext``.
///
/// Mirrors the ``ModelContainer`` pattern: a `final class : Sendable` that
/// wraps the non-Sendable context in a `SerialAccessContainer` and exposes
/// async `perform(_:)` closures for serialized access.
public final class MTPDrafterContainer: Sendable {
    private let context: SerialAccessContainer<MTPDrafterContext>

    public var configuration: ModelConfiguration {
        get async {
            await context.read { $0.configuration }
        }
    }

    public init(context: consuming MTPDrafterContext) {
        self.context = .init(context)
    }

    /// Perform an action on the ``MTPDrafterContext``. Callers _must_ eval
    /// any `MLXArray` before returning as `MLXArray` is not `Sendable`.
    public func perform<R: Sendable>(
        _ action: @Sendable (MTPDrafterContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0)
        }
    }
}

// MARK: - Cross-model state keys
//
// Public ``LMOutput/Key`` declarations for MTP speculative decoding. The
// target model (e.g. ``Gemma4`` in MLXVLM) writes these into its
// ``LMOutput/state`` when the iterator opts in via ``mtpEmitFlagKey``; the
// ``MTPSpeculativeTokenIterator`` reads them and threads them to the drafter
// as method arguments. Public scope is required because writer and reader
// live in different modules.

/// Target writes the hidden representation required by its MTP head here.
/// This is architecture-specific, but must match the representation used to
/// train the paired MTP head. Qwen and Gemma assistants both publish their
/// post-final-norm representation. The iterator threads it to the drafter
/// unchanged as `lastHidden`.
public let mtpLastHiddenStatesKey =
    LMOutput.Key<MLXArray>("mtp.lastHiddenStates")

/// Target writes one `(keys, values)` tuple per `layer_type`
/// (`"full_attention"`, `"sliding_attention"`) here, drawn from the last
/// layer of each type. ``MTPSpeculativeTokenIterator`` reads it and threads
/// it to the drafter as `sharedKV`.
public let mtpSharedKVStatesKey =
    LMOutput.Key<[String: (MLXArray, MLXArray)]>("mtp.sharedKVStates")

/// Target writes the absolute cache offset for each emitted shared-K/V
/// layer-type here. The K/V tensors themselves may have a capped sequence
/// axis when backed by a rotating cache, so their shape is not always the
/// absolute RoPE/query position.
public let mtpSharedKVOffsetsKey =
    LMOutput.Key<[String: Int]>("mtp.sharedKVOffsets")

/// Target writes optional position delta state here for MTP drafters that
/// need to reproduce target-specific RoPE continuation positions.
public let mtpPositionDeltasKey =
    LMOutput.Key<MLXArray>("mtp.positionDeltas")

/// The MTP iterator sets this key on the ``LMOutput/State`` it passes into
/// the main model on each call to opt the target into emitting
/// ``mtpLastHiddenStatesKey`` and ``mtpSharedKVStatesKey``. An absent key
/// reads as `false` (no emit), so non-MTP callers are unaffected.
public let mtpEmitFlagKey = LMOutput.Key<Bool>("mtp.emitDrafterState")

/// Requests a recurrent-cache checkpoint after this many verification input
/// tokens. Hybrid Qwen models use `1` for MTP-1 so a rejected draft restores
/// state after the always-committed bonus token without replaying the model.
public let mtpCacheCheckpointIndexKey =
    LMOutput.Key<Int>("mtp.cacheCheckpointIndex")

/// Which cache entry each ``mtpSharedKVStatesKey`` tuple was read from, keyed the same way.
///
/// A speculative round keeps only part of what its verify pass wrote, so the emitted snapshot has
/// to be reconciled against the cache afterwards. That reconciliation is exact only if the
/// consumer knows the entry behind each tuple: a sliding layer's snapshot is bounded by its ring
/// and a global layer's is not, and at the crossing the two are indistinguishable by length.
///
/// Public alongside its siblings, because it is part of the same contract rather than an
/// implementation detail of one writer: a target that sets ``mtpSharedKVStatesKey`` is expected to
/// set this too, and a snapshot that arrives without it is refused rather than reconciled against
/// a guessed bound. A target outside this package cannot satisfy that contract without it.
public let mtpSharedKVSourceIndicesKey =
    LMOutput.Key<[String: Int]>("mtp.sharedKVSourceIndices")

// MARK: - Iterator stats surface

/// Introspection surface for token iterators that perform MTP speculative
/// decoding.
///
/// The iterator value lives inside `generateLoopTask` and never escapes the
/// stream, so its per-stream draft proposal and acceptance counters are not
/// reachable through the high-level `generate(...)` API. Conforming the
/// iterator to this protocol lets `generateLoopTask` downcast and thread the
/// counters through the emitted `.info` event's
/// ``GenerateCompletionInfo/proposedDraftTokens``,
/// ``GenerateCompletionInfo/acceptedDraftTokens``, and
/// ``GenerateCompletionInfo/passthroughReason`` fields. Non-MTP iterators do
/// not conform; the downcast returns nil and the fields default to nil for
/// non-MTP streams.
public protocol MTPStatsCollecting {
    /// Total tokens proposed across all speculation rounds in the stream.
    var proposedDraftTokens: Int { get }

    /// Total tokens accepted by the target across all speculation rounds.
    var acceptedDraftTokens: Int { get }

    /// Nil when speculation completed with a resumable boundary. Non-nil if
    /// sticky passthrough engaged or a stateful continuation had to be
    /// discarded, with the first failure reason captured by the iterator.
    var passthroughReason: String? { get }
}
