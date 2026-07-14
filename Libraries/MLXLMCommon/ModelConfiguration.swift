// Copyright © 2024 Apple Inc.

import Foundation

/// Configuration for a given model:  at least an org/name identifier or a directory with the model files.
///
/// Optionally callers can provide some default values and overrides for:
///
/// - a default prompt
/// - EOS tokens / strings
/// - tool calling formats
///
/// Some of these are specific to LLMs and VLMs -- embedding models will ignore those properties.
///
/// See e.g. `MLXLM.ModelRegistry` for an example of use.
public struct ModelConfiguration: Sendable {

    public enum DirectoryError: LocalizedError, Equatable {
        case unresolvedModelDirectory(String)
        case unresolvedTokenizerDirectory(String)

        public var errorDescription: String? {
            switch self {
            case .unresolvedModelDirectory(let id):
                return "Model configuration '\(id)' has not been resolved to a local directory."
            case .unresolvedTokenizerDirectory(let id):
                return "Tokenizer source '\(id)' has not been resolved to a local directory."
            }
        }
    }

    /// The backing storage for the model's location.
    public enum Identifier: Sendable {
        /// A Hugging Face Hub repository identifier (e.g., "BAAI/bge-small-en-v1.5").
        case id(String, revision: String = "main")
        /// A file system URL pointing to a local model directory.
        case directory(URL)
    }

    /// The model's identifier (ID or Directory).
    public var id: Identifier

    /// A display-friendly name for the model.
    ///
    /// For Hub models, this returns the repo ID. For local directories,
    /// it returns a path-based name (e.g., "ParentDir/ModelDir").
    public var name: String {
        switch id {
        case .id(let id, _):
            id
        case .directory(let url):
            url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
        }
    }

    /// The resolved local directory containing model files.
    ///
    /// - Throws: ``DirectoryError/unresolvedModelDirectory(_:)`` if this configuration still
    ///   identifies a remote model by ID rather than a local directory.
    package var modelDirectory: URL {
        get throws {
            switch id {
            case .directory(let directory):
                return directory
            case .id(let id, _):
                throw DirectoryError.unresolvedModelDirectory(id)
            }
        }
    }

    /// The resolved local directory containing tokenizer files.
    ///
    /// If ``tokenizerSource`` is `nil`, this falls back to ``modelDirectory``.
    ///
    /// - Throws: ``DirectoryError/unresolvedTokenizerDirectory(_:)`` if the tokenizer still
    ///   points to a remote source by ID, or ``DirectoryError/unresolvedModelDirectory(_:)``
    ///   if no separate tokenizer source is set and the model itself is unresolved.
    package var tokenizerDirectory: URL {
        get throws {
            switch tokenizerSource {
            case .directory(let directory):
                return directory
            case .id(let id, _):
                throw DirectoryError.unresolvedTokenizerDirectory(id)
            case nil:
                return try modelDirectory
            }
        }
    }

    /// Where to load the tokenizer from when it differs from the model directory.
    ///
    /// - `.id`: download from a remote provider (requires a ``Downloader``)
    /// - `.directory`: load from a local path
    /// - `nil`: use the same directory as the model
    public let tokenizerSource: TokenizerSource?

    /// A reasonable default prompt for the model
    public var defaultPrompt: String

    /// Additional tokens to use for end of string (specified as strings, converted to IDs at runtime)
    public var extraEOSTokens: Set<String>

    /// Text sequences that stop decoded generation when encountered.
    ///
    /// If this is `nil`, decoded stop strings fall back to ``extraEOSTokens``.
    /// Set this explicitly, including to an empty set, to override that fallback.
    public var stopStrings: Set<String>?

    /// Text sequences to use for decoded stop-string matching.
    public var effectiveStopStrings: Set<String> {
        stopStrings ?? extraEOSTokens
    }

    /// EOS token IDs loaded from config.json/generation_config.json
    public var eosTokenIds: Set<Int> = []

    /// Tool call format for this model (nil = default JSON format)
    public var toolCallFormat: ToolCallFormat?

    /// Model-specific policy for keeping a cached chat transcript canonical
    /// across ``ChatSession`` continuation turns. `nil` (the default)
    /// disables continuation repair. Like ``extraEOSTokens`` and
    /// ``toolCallFormat``, this captures chat-template knowledge on the
    /// model side so callers never handle template-specific token framing.
    public var continuationPolicy: ContinuationPolicy?

    public init(
        id: String, revision: String = "main",
        tokenizerSource: TokenizerSource? = nil,
        defaultPrompt: String = "",
        extraEOSTokens: Set<String> = [],
        stopStrings: Set<String>? = nil,
        eosTokenIds: Set<Int> = [],
        toolCallFormat: ToolCallFormat? = nil,
        continuationPolicy: ContinuationPolicy? = nil
    ) {
        self.id = .id(id, revision: revision)
        self.tokenizerSource = tokenizerSource
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.stopStrings = stopStrings
        self.eosTokenIds = eosTokenIds
        self.toolCallFormat = toolCallFormat
        self.continuationPolicy = continuationPolicy
    }

    public init(
        directory: URL,
        tokenizerSource: TokenizerSource? = nil,
        defaultPrompt: String = "",
        extraEOSTokens: Set<String> = [],
        stopStrings: Set<String>? = nil,
        eosTokenIds: Set<Int> = [],
        toolCallFormat: ToolCallFormat? = nil,
        continuationPolicy: ContinuationPolicy? = nil
    ) {
        self.id = .directory(directory)
        self.tokenizerSource = tokenizerSource
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.stopStrings = stopStrings
        self.eosTokenIds = eosTokenIds
        self.toolCallFormat = toolCallFormat
        self.continuationPolicy = continuationPolicy
    }

    /// Model-specific knowledge for repairing a continuation prompt that is
    /// appended to a non-empty KV cache.
    ///
    /// ``ChatSession`` determines the *semantics* (is this a continuation?
    /// was the previous model turn truncated or completed?) and this policy
    /// supplies the template-specific mechanics, keeping the session API
    /// model-agnostic.
    public struct ContinuationPolicy: Sendable, Equatable {

        /// Text that closes a *truncated* model turn (e.g. `"<turn|>\n"` for
        /// Gemma 4, `"<end_of_turn>\n"` for Gemma 3). Applied only when the
        /// previous generation did not finish with a stop token — a
        /// completed turn already has its terminator in the cache.
        public let truncatedTurnClosure: String?

        /// Whether the model's chat template begins with `{{ bos_token }}`
        /// and therefore re-emits `<bos>` on every render; the duplicate is
        /// stripped from continuation prompts. The strip only triggers when
        /// the first prompt token actually equals the tokenizer's BOS.
        public let stripsRepeatedBOS: Bool

        public init(truncatedTurnClosure: String? = nil, stripsRepeatedBOS: Bool = true) {
            self.truncatedTurnClosure = truncatedTurnClosure
            self.stripsRepeatedBOS = stripsRepeatedBOS
        }
    }

    /// Maps this configuration's behavioral properties into a
    /// ``ResolvedModelConfiguration`` with the given directories.
    ///
    /// This is a pure data mapping with no I/O. The directories should
    /// already be resolved (downloaded or local) before calling this method.
    public func resolved(
        modelDirectory: URL, tokenizerDirectory: URL
    ) -> ResolvedModelConfiguration {
        ResolvedModelConfiguration(
            modelDirectory: modelDirectory,
            tokenizerDirectory: tokenizerDirectory,
            name: name,
            defaultPrompt: defaultPrompt,
            extraEOSTokens: extraEOSTokens,
            stopStrings: stopStrings,
            eosTokenIds: eosTokenIds,
            toolCallFormat: toolCallFormat)
    }

}

extension ModelConfiguration: Equatable {

}

extension ModelConfiguration.Identifier: Equatable {

    public static func == (lhs: ModelConfiguration.Identifier, rhs: ModelConfiguration.Identifier)
        -> Bool
    {
        switch (lhs, rhs) {
        case (.id(let lhsID, let lhsRevision), .id(let rhsID, let rhsRevision)):
            lhsID == rhsID && lhsRevision == rhsRevision
        case (.directory(let lhsURL), .directory(let rhsURL)):
            lhsURL == rhsURL
        default:
            false
        }
    }
}
