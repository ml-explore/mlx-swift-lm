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

    /// Reasoning (chain-of-thought) protocol for this model (nil = non-reasoning model)
    public var reasoningConfig: ReasoningConfig? = nil

    /// Whether a complete text boundary in this model's stream is a safe place
    /// to end a step for ``ChatSession/steer(_:policy:response:)``.
    ///
    /// A declared reasoning model streams its chain of thought as ordinary text,
    /// so an early boundary could split a reasoning block. Those models take
    /// their natural step boundary instead.
    package var allowsEarlySteeringBoundary: Bool {
        reasoningConfig == nil
    }

    /// How to choose which safetensors files in the model directory hold the model's weights.
    ///
    /// The default, ``WeightFileSelection/automatic``, handles a well-packaged checkpoint and
    /// the common packaging mistakes. Set ``WeightFileSelection/allFilesPresent`` for a
    /// checkpoint whose index is known to omit weights the model needs -- see the caveats on
    /// that case before reaching for it.
    public var weightFileSelection: WeightFileSelection = .automatic

    /// Overrides the ``MessageGenerator`` the model would otherwise supply.
    ///
    /// A model class is shared by every checkpoint of its model type, so a fine-tune that
    /// needs a different chat-template shape cannot express that on the model itself without
    /// affecting its siblings -- e.g. TranslateGemma, which loads through the same `gemma3`
    /// text path as plain Gemma 3. Set this on the registry entry (or by the caller) instead.
    /// `nil` keeps the model's own default.
    public var messageGenerator: (any MessageGenerator)? = nil

    public init(
        id: String, revision: String = "main",
        tokenizerSource: TokenizerSource? = nil,
        defaultPrompt: String = "",
        extraEOSTokens: Set<String> = [],
        stopStrings: Set<String>? = nil,
        eosTokenIds: Set<Int> = [],
        toolCallFormat: ToolCallFormat? = nil,
        reasoningConfig: ReasoningConfig? = nil,
        messageGenerator: (any MessageGenerator)? = nil
    ) {
        self.id = .id(id, revision: revision)
        self.tokenizerSource = tokenizerSource
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.stopStrings = stopStrings
        self.eosTokenIds = eosTokenIds
        self.toolCallFormat = toolCallFormat
        self.reasoningConfig = reasoningConfig
        self.messageGenerator = messageGenerator
    }

    public init(
        directory: URL,
        tokenizerSource: TokenizerSource? = nil,
        defaultPrompt: String = "",
        extraEOSTokens: Set<String> = [],
        stopStrings: Set<String>? = nil,
        eosTokenIds: Set<Int> = [],
        toolCallFormat: ToolCallFormat? = nil,
        reasoningConfig: ReasoningConfig? = nil,
        messageGenerator: (any MessageGenerator)? = nil
    ) {
        self.id = .directory(directory)
        self.tokenizerSource = tokenizerSource
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.stopStrings = stopStrings
        self.eosTokenIds = eosTokenIds
        self.toolCallFormat = toolCallFormat
        self.reasoningConfig = reasoningConfig
        self.messageGenerator = messageGenerator
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
            toolCallFormat: toolCallFormat,
            reasoningConfig: reasoningConfig,
            messageGenerator: messageGenerator,
            weightFileSelection: weightFileSelection)
    }

}

extension ModelConfiguration: Equatable {

    // Keep in sync with the stored properties above: synthesis is impossible because
    // `messageGenerator` is not `Equatable`, so a new property will not appear here on its own.
    public static func == (lhs: ModelConfiguration, rhs: ModelConfiguration) -> Bool {
        lhs.id == rhs.id
            && lhs.tokenizerSource == rhs.tokenizerSource
            && lhs.defaultPrompt == rhs.defaultPrompt
            && lhs.extraEOSTokens == rhs.extraEOSTokens
            && lhs.stopStrings == rhs.stopStrings
            && lhs.eosTokenIds == rhs.eosTokenIds
            && lhs.toolCallFormat == rhs.toolCallFormat
            && lhs.reasoningConfig == rhs.reasoningConfig
            && lhs.weightFileSelection == rhs.weightFileSelection
            && sameMessageGenerator(lhs.messageGenerator, rhs.messageGenerator)
    }

    /// ``MessageGenerator`` is not `Equatable` -- generators are stateless, so identity of
    /// the concrete type is the meaningful comparison.
    private static func sameMessageGenerator(
        _ lhs: (any MessageGenerator)?, _ rhs: (any MessageGenerator)?
    ) -> Bool {
        switch (lhs, rhs) {
        case (nil, nil):
            true
        case (.some(let lhs), .some(let rhs)):
            ObjectIdentifier(type(of: lhs)) == ObjectIdentifier(type(of: rhs))
        default:
            false
        }
    }
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
