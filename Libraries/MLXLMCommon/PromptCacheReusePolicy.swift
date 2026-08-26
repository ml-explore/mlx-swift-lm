// Copyright © 2025 Apple Inc.

import Foundation

/// How a newly rendered prompt should be reconciled with the tokens a live
/// KV cache already represents.
///
/// A decision describes *what* to do; applying it (trimming, rebuilding,
/// prefilling) belongs to whoever owns the caches.
enum PromptCacheReuseDecision: Equatable {

    /// Feed the whole prompt. The cache holds nothing worth reusing.
    case prefillAll

    /// The cache already represents a usable prefix. Feed only
    /// `promptTokens[suffixStart...]`; afterwards the cache represents
    /// `representedTokens`.
    ///
    /// `representedTokens` is not necessarily the rendered prompt. A response
    /// protocol may keep generated tokens that a cold template render cannot
    /// reproduce, in which case the ledger diverges from the render by design.
    case appendSuffix(suffixStart: Int, representedTokens: [Int])

    /// Feed a suffix into the main cache while discarding unusable speculative
    /// state (an ordinary draft cache or an MTP continuation). The current
    /// generation must use the main model only.
    case appendSuffixToMain(suffixStart: Int, representedTokens: [Int])

    /// Rewind every cache by `trimCount` (down to `commonPrefixLength`), then
    /// feed `promptTokens[commonPrefixLength...]`.
    case trimToCommonPrefix(commonPrefixLength: Int, trimCount: Int)

    /// Rewind only the target cache, discard speculative state, and continue
    /// target-only from the common prefix.
    case trimToCommonPrefixMainOnly(commonPrefixLength: Int, trimCount: Int)

    /// The cache cannot be reconciled with this prompt. Discard it, drop any
    /// carried model state, and feed the whole prompt.
    case rebuild

    /// `true` when a non-empty cached prefix is carried into this turn.
    var reusesCachedPrefix: Bool {
        switch self {
        case .appendSuffix, .appendSuffixToMain, .trimToCommonPrefix,
            .trimToCommonPrefixMainOnly:
            return true
        case .prefillAll, .rebuild:
            return false
        }
    }
}

/// The prompt-side facts of one turn that affect cache reuse.
struct PromptCacheTurn: Sendable {
    /// The full rendered prompt for this turn.
    var promptTokens: [Int]

    /// This turn's messages introduce new media, so the cached text prefix is
    /// no longer a valid prefix of the model's actual input.
    var carriesNewMedia: Bool = false

    /// The prepared input contains image/video/audio tensors, which the rewind
    /// path cannot account for.
    var carriesPreparedMedia: Bool = false

    /// The prepared input carries an explicit attention mask; a partial prefill
    /// would apply it against the wrong positions.
    var carriesAttentionMask: Bool = false

    /// Per-call model state (e.g. M-RoPE deltas) is carried across turns. Such
    /// state is anchored to a prefill and cannot be rewound.
    var carriesModelState: Bool = false

    /// This turn appends tool results to a transcript whose last assistant
    /// message issued tool calls, i.e. the generation loop is resuming rather
    /// than starting a new exchange.
    ///
    /// Response protocols that keep private per-turn state in the cache use
    /// this to recognize a restart they can splice onto.
    var isToolResultContinuation: Bool = false

    /// Generated tokens returned to the previous caller but not yet represented
    /// by the main cache. Speculative iterators can leave their final verifier
    /// sample in this state.
    var previousGenerationUncommittedTokens: [Int] = []

    /// Number of structured assistant tool-call messages represented by the
    /// rendered prompt. A protocol can compare this with raw control-token
    /// occurrences to reject ambiguous boundaries introduced by message text.
    var structuredToolCallCount: Int?

    /// The session is configured to use a draft model when possible. Protocol
    /// rules use this to avoid resuming from a private main-cache path with a
    /// missing or divergent draft cache.
    var usesSpeculativeDecoding: Bool = false
}

/// What the caches currently hold.
struct PromptCacheState: Sendable {
    /// Exact tokens the main cache represents, per the session's ledger. Empty
    /// means the ledger was invalidated and nothing may be spliced onto.
    var cachedTokens: [Int]

    /// Authoritative logical position of the main cache — the model-wide
    /// timeline maintained by ``KVCacheStorage``, not a per-entry offset.
    var processedTokenCount: Int

    /// The main cache's timeline agrees with the ledger length.
    var mainCacheIsAligned: Bool = false

    /// Live auxiliary state is available for speculative continuation. This
    /// means a draft KV cache for ordinary speculation, an aligned continuation
    /// for stateful MTP, or the stateless MTP capability itself.
    var hasSpeculativeState: Bool = false

    /// The draft cache's timeline agrees with the ledger length. This remains
    /// `true` when the auxiliary state agrees with the target timeline.
    var speculativeStateIsAligned: Bool = true

    /// Whether auxiliary state can follow a target rewind. Stateful MTP
    /// continuations are append-only in the first implementation.
    var speculativeStateIsRewindable: Bool = true

    /// The current full prompt can rebuild missing ordinary draft-model state
    /// if speculation is ultimately admitted. This preserves suffix-only
    /// target generation when a later memory-policy check falls back, while
    /// still allowing both caches to be rebuilt before active speculation.
    var canRebuildSpeculativeStateFromPrompt: Bool = false

    /// Whether a missing or unusable speculative continuation should preserve
    /// the valid target cache and continue target-only. Enabled for MTP; the
    /// ordinary draft-model path retains its existing rebuild behavior.
    var allowsMainOnlyFallback: Bool = false

    /// Every cache supports rewinding.
    var isTrimmable: Bool = false
}

/// One reusability rule.
///
/// Returning `nil` means "this rule does not apply"; the policy then consults
/// the next rule. This is the extension point for response protocols whose
/// on-device token stream is not reproducible by a chat-template render.
protocol PromptCacheReuseRule: Sendable {
    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision?
}

/// Decides how to reuse a KV cache across turns by consulting an ordered list
/// of rules.
///
/// The policy is free of MLX and session state, so the whole decision table can
/// be unit tested directly and the caller's only job is to *apply* the result.
struct PromptCacheReusePolicy: Sendable {

    /// Rules that apply to every model, in priority order.
    static let standardRules: [any PromptCacheReuseRule] = [
        ExtendCachedPrefixRule(),
        RewindToCommonPrefixRule(),
    ]

    private let rules: [any PromptCacheReuseRule]

    /// - Parameter protocolRules: rules contributed by the model's response
    ///   protocol. They are consulted before the standard rules, because a
    ///   protocol that keeps unrenderable state in the cache must claim the
    ///   turn before generic prefix comparison is attempted on token streams
    ///   that are not comparable.
    init(protocolRules: [any PromptCacheReuseRule] = []) {
        self.rules = protocolRules + Self.standardRules
    }

    func decide(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision {
        for rule in rules {
            if let decision = rule.reuse(turn: turn, cache: cache) {
                return decision
            }
        }
        return .rebuild
    }
}

// MARK: - Standard rules

/// Appends the trailing tokens when the prompt strictly extends the cache.
struct ExtendCachedPrefixRule: PromptCacheReuseRule {
    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        guard !cache.cachedTokens.isEmpty,
            cache.mainCacheIsAligned,
            !turn.carriesNewMedia,
            !turn.carriesAttentionMask,
            turn.promptTokens.count > cache.cachedTokens.count,
            turn.promptTokens.starts(with: cache.cachedTokens)
        else {
            return nil
        }

        if turn.usesSpeculativeDecoding {
            if cache.hasSpeculativeState && cache.speculativeStateIsAligned {
                return .appendSuffix(
                    suffixStart: cache.cachedTokens.count,
                    representedTokens: turn.promptTokens)
            }
            if cache.canRebuildSpeculativeStateFromPrompt {
                return .appendSuffix(
                    suffixStart: cache.cachedTokens.count,
                    representedTokens: turn.promptTokens)
            }
            if cache.allowsMainOnlyFallback {
                return .appendSuffixToMain(
                    suffixStart: cache.cachedTokens.count,
                    representedTokens: turn.promptTokens)
            }
            return nil
        }

        return .appendSuffix(
            suffixStart: cache.cachedTokens.count, representedTokens: turn.promptTokens)
    }
}

/// Rewinds to the longest common prefix, or rebuilds when that is unsafe.
///
/// This rule always decides, so it is the natural terminal rule.
struct RewindToCommonPrefixRule: PromptCacheReuseRule {
    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        // A cache at position zero has nothing to rewind and nothing to
        // invalidate, so the prompt is simply prefilled into it.
        guard cache.processedTokenCount != 0 else {
            return .prefillAll
        }

        let commonPrefixLength = zip(turn.promptTokens, cache.cachedTokens)
            .prefix { $0 == $1 }
            .count
        let trimCount = cache.cachedTokens.count - commonPrefixLength

        let canRewind =
            commonPrefixLength > 0
            && commonPrefixLength < turn.promptTokens.count
            && trimCount > 0
            && cache.mainCacheIsAligned
            && cache.isTrimmable
            && !turn.carriesNewMedia
            && !turn.carriesPreparedMedia
            && !turn.carriesAttentionMask
            && !turn.carriesModelState

        guard canRewind else {
            // The template changed an already-cached portion of the transcript,
            // or this input carries state/media that cannot be rewound safely.
            // Rebuild rather than combining a mismatched prompt with stale
            // model state.
            return .rebuild
        }

        if turn.usesSpeculativeDecoding {
            if cache.hasSpeculativeState,
                cache.speculativeStateIsAligned,
                cache.speculativeStateIsRewindable
            {
                return .trimToCommonPrefix(
                    commonPrefixLength: commonPrefixLength, trimCount: trimCount)
            }
            if cache.canRebuildSpeculativeStateFromPrompt {
                return .trimToCommonPrefix(
                    commonPrefixLength: commonPrefixLength, trimCount: trimCount)
            }
            if cache.allowsMainOnlyFallback {
                return .trimToCommonPrefixMainOnly(
                    commonPrefixLength: commonPrefixLength, trimCount: trimCount)
            }
            return .rebuild
        }

        return .trimToCommonPrefix(commonPrefixLength: commonPrefixLength, trimCount: trimCount)
    }
}
