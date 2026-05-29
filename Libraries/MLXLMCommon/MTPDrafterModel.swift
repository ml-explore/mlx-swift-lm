// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

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
/// single call, and holds no transient round-state between calls. The
/// `MTPSpeculativeTokenIterator` (Phase B) extracts the shared K/V from the
/// target's `LMOutput.state` and threads it to the drafter as a method
/// argument.
///
/// One-time binding: `bind(target:)` caches read-only references to the
/// target's input embeddings, embed scale, and per-layer type metadata.
/// Stored references are read-only during eval (consistent with PR #283's
/// "no mutation during eval" invariant).
public protocol MTPDrafterModel: BaseLanguageModel {
    /// One-time setup. Caches references to the target's input embeddings,
    /// embed scale, and per-layer type metadata. Must be called before any
    /// `draftBlock(...)` invocation. Stored state is read-only during eval.
    ///
    /// - Parameter target: The main language model that this drafter speculates for.
    func bind(target: any LanguageModel)

    /// K-step drafting from a constant position.
    ///
    /// Returns the proposed tokens as a `[B, blockSize - 1]` MLXArray. The
    /// drafter holds no transient round-state between calls — every per-round
    /// input is threaded as a method argument.
    ///
    /// - Parameters:
    ///   - lastToken: Bonus token from the target's last verify pass, shape `[B]`.
    ///   - lastHidden: Target's last hidden state, shape `[B, 1, backbone_hidden_size]`.
    ///   - sharedKV: Dict keyed by `layer_type` (`"full_attention"` /
    ///     `"sliding_attention"`) mapping to `(keys, values)` `MLXArray`s for
    ///     the last layer of that layer-type in the target.
    ///   - positionIds: Constant position for the round, shape `[B, 1]`.
    ///   - blockSize: Total tokens in the round (the drafter returns
    ///     `blockSize - 1`; the bonus token is implicit).
    ///   - sampler: `LogitSampler` to apply to each step's logits.
    /// - Returns: `[B, blockSize - 1]` token array.
    func draftBlock(
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionIds: MLXArray,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray
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

/// Target writes its post-final-norm hidden state here (pre-lm_head,
/// pre-softcap). ``MTPSpeculativeTokenIterator`` reads it and threads it to
/// the drafter as `lastHidden`.
public let mtpLastHiddenStatesKey =
    LMOutput.Key<MLXArray>("mtp.lastHiddenStates")

/// Target writes one `(keys, values)` tuple per `layer_type`
/// (`"full_attention"`, `"sliding_attention"`) here, drawn from the last
/// layer of each type. ``MTPSpeculativeTokenIterator`` reads it and threads
/// it to the drafter as `sharedKV`.
public let mtpSharedKVStatesKey =
    LMOutput.Key<[String: (MLXArray, MLXArray)]>("mtp.sharedKVStates")

/// The MTP iterator sets this key on the ``LMOutput/State`` it passes into
/// the main model on each call to opt the target into emitting
/// ``mtpLastHiddenStatesKey`` and ``mtpSharedKVStatesKey``. An absent key
/// reads as `false` (no emit), so non-MTP callers are unaffected.
public let mtpEmitFlagKey = LMOutput.Key<Bool>("mtp.emitDrafterState")

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

    /// nil if the iterator stayed in speculative mode for the full stream;
    /// non-nil if sticky-passthrough engaged, with the reason string captured
    /// at the moment of engagement.
    var passthroughReason: String? { get }
}
