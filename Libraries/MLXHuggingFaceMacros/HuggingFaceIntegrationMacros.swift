import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

@main
struct Macros: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        DownloaderMacro.self,
        TokenizerAdaptorMacro.self,
        TokenizerLoaderMacro.self,
        LoadContainerMacro.self,
        LoadContextMacro.self,
        LanguageModelMacro.self,
    ]
}

public struct DownloaderMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        let argument = node.arguments.first?.expression.description ?? "HubClient()"

        return
            """
            // make sure you:
            //
            // import HuggingFace
            //
            { (hubApi: HuggingFace.HubClient) -> MLXLMCommon.Downloader in
                struct HubBridge: MLXLMCommon.Downloader {
                    private let upstream: HuggingFace.HubClient

                    init(_ upstream: HuggingFace.HubClient) {
                        self.upstream = upstream
                    }

                    public func download(
                        id: String,
                        revision: String?,
                        matching patterns: [String],
                        useLatest: Bool,
                        progressHandler: @Sendable @escaping (Foundation.Progress) -> Void
                    ) async throws -> URL {                        
                        guard let repoID = HuggingFace.Repo.ID(rawValue: id) else {
                            throw HuggingFaceDownloaderError.invalidRepositoryID(id)
                        }
                        let revision = revision ?? "main"

                        return try await upstream.downloadSnapshot(
                            of: repoID,
                            revision: revision,
                            matching: patterns,
                            progressHandler: { @MainActor progress in
                                progressHandler(progress)
                            }
                        )
                    }                    
                }

                return HubBridge(hubApi)
            }(\(raw: argument))
            """
    }
}

public struct TokenizerAdaptorMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let argument = node.arguments.first?.expression else {
            throw MacroExpansionError.message("#adaptHuggingFaceTokenizer requires an argument")
        }

        return
            """
            // make sure you:
            //
            // import Tokenizers
            //
            { (huggingFaceTokenizer: Tokenizers.Tokenizer) -> MLXLMCommon.Tokenizer in
                struct TokenizerBridge: MLXLMCommon.BoundedStreamingDecodeTokenizer {
                    private let upstream: any Tokenizers.Tokenizer

                    init(_ upstream: any Tokenizers.Tokenizer) {
                        self.upstream = upstream
                    }

                    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
                        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
                    }

                    // swift-transformers uses `decode(tokens:)` instead of `decode(tokenIds:)`.
                    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
                        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
                    }

                    func convertTokenToId(_ token: String) -> Int? {
                        upstream.convertTokenToId(token)
                    }

                    func convertIdToToken(_ id: Int) -> String? {
                        upstream.convertIdToToken(id)
                    }

                    var bosToken: String? { upstream.bosToken }
                    var eosToken: String? { upstream.eosToken }
                    var unknownToken: String? { upstream.unknownToken }

                    // swift-transformers' supported decoder pipeline has bounded
                    // suffix context (UTF-8 byte fallback plus fixed cleanup
                    // patterns). Keep a conservative overlap when streaming so
                    // already-emitted prefixes do not remain in the decode buffer.
                    var streamingDecodeContextSize: Int? { 16 }

                    func applyChatTemplate(
                        messages: [[String: any Sendable]],
                        tools: [[String: any Sendable]]?,
                        additionalContext: [String: any Sendable]?
                    ) throws -> [Int] {
                        do {
                            return try upstream.applyChatTemplate(
                                messages: messages, tools: tools, additionalContext: additionalContext)
                        } catch Tokenizers.TokenizerError.missingChatTemplate {
                            throw MLXLMCommon.TokenizerError.missingChatTemplate
                        }
                    }
                }

                return TokenizerBridge(huggingFaceTokenizer)
            }(\(argument))
            """
    }
}

public struct TokenizerLoaderMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        return
            """
            { () -> MLXLMCommon.TokenizerLoader in
                struct TransformersLoader: MLXLMCommon.TokenizerLoader {
                    public init() {}

                    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
                        // make sure you:
                        //
                        // import Tokenizers
                        //
                        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
                        return #adaptHuggingFaceTokenizer(upstream)
                    }
                }

                return TransformersLoader()
            }()
            """
    }
}

public struct LoadContainerMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let configuration = node.arguments.first?.expression else {
            throw MacroExpansionError.message(
                "#huggingFaceLoadModelContainer requires a configuration")
        }

        let progress =
            if let expr = node.arguments.first(where: { $0.label?.text == "progressHandler" })?
                .expression
            {
                expr.description
            } else {
                "{ _ in }"
            }

        return
            """
            loadModelContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: \(configuration),
                progressHandler: \(raw: progress))
            """
    }
}

public struct LoadContextMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let configuration = node.arguments.first?.expression else {
            throw MacroExpansionError.message("#huggingFaceLoadModel requires a configuration")
        }

        let progress =
            if let expr = node.arguments.first(where: { $0.label?.text == "progressHandler" })?
                .expression
            {
                expr.description
            } else {
                "{ _ in }"
            }

        return
            """
            loadModel(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: \(configuration),
                progressHandler: \(raw: progress))
            """
    }
}

public struct LanguageModelMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        func argument(_ label: String) -> String? {
            node.arguments.first(where: { $0.label?.text == label })?
                .expression.description
        }

        guard let configuration = argument("configuration") else {
            throw MacroExpansionError.message(
                "#huggingFaceLanguageModel requires a configuration")
        }

        // Forward the caller's configuration verbatim, then forward
        // capabilities / configurationResolver only when the caller supplied
        // them so the MLXLanguageModel init's own defaults apply otherwise.
        var arguments = ["configuration: \(configuration)"]
        if let capabilities = argument("capabilities") {
            arguments.append("capabilities: \(capabilities)")
        }
        if let resolver = argument("configurationResolver") {
            arguments.append("configurationResolver: \(resolver)")
        }
        arguments.append(
            """
            weightsLocation: { id in
                    let cache = HuggingFace.HubCache.default
                    guard let repo = HuggingFace.Repo.ID(rawValue: id) else {
                        return cache.cacheDirectory
                    }
                    if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
                        let snapshot = try? cache.snapshotPath(repo: repo, kind: .model, commitHash: commit) {
                        return snapshot
                    }
                    return cache.repoDirectory(repo: repo, kind: .model)
                }
            """)
        arguments.append(
            """
            load: { configuration, progressHandler in
                    try await loadModelContainer(
                        from: #hubDownloader(),
                        using: #huggingFaceTokenizerLoader(),
                        configuration: configuration,
                        progressHandler: progressHandler)
                }
            """)

        let argumentList = arguments.joined(separator: ",\n    ")
        return
            """
            MLXLanguageModel(
                \(raw: argumentList))
            """
    }
}

enum MacroExpansionError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case .message(let text): return text
        }
    }
}
