// Copyright © 2026 Apple Inc.

import Foundation

// MARK: - Inference from a chat template

extension ToolCallFormat {

    /// Markers that identify a dialect in a chat template.
    ///
    /// A signature matches when every marker in any one group is present, which
    /// covers both dialects identified by a single tag and those that can only
    /// be told apart by a pair of them.
    private struct Signature {
        let format: ToolCallFormat
        let markerGroups: [[String]]

        init(_ format: ToolCallFormat, anyOf markerGroups: [[String]]) {
            self.format = format
            self.markerGroups = markerGroups
        }

        init(_ format: ToolCallFormat, allOf markers: String...) {
            self.init(format, anyOf: [markers])
        }

        func matches(_ template: String) -> Bool {
            markerGroups.contains { $0.allSatisfy(template.contains) }
        }
    }

    /// Ordered most specific first: a template that renders `<|tool_call>` also
    /// contains `<tool_call>`, so the broader signatures have to come last.
    ///
    /// Mirrors `_infer_tool_parser` in mlx-lm's `tokenizer_utils.py`. Protocols
    /// with token-level framing (Harmony, Onyx) are deliberately absent: they
    /// are selected by the models that own them, and guessing them from text
    /// would reroute the whole response protocol rather than one payload shape.
    private static let signatures: [Signature] = [
        Signature(.minimaxM2, allOf: "<minimax:tool_call>"),
        Signature(.gemma4, allOf: "<|tool_call>", "<tool_call|>"),
        Signature(.gemma, allOf: "<start_function_call>"),
        Signature(.glm4, allOf: "<arg_key>"),
        Signature(.lfm2, allOf: "<|tool_list_start|>"),
        Signature(
            .xmlFunction, anyOf: [["<tool_call>\n<function="], [#"<tool_call>\n<function="#]]),
        Signature(.kimiK2, allOf: "<|tool_calls_section_begin|>"),
        Signature(.mistral, allOf: "[TOOL_CALLS]"),
        Signature(.json, allOf: "<tool_call>", "tool_call.name"),
    ]

    /// The dialect a chat template renders, or `nil` when it renders none that
    /// is recognized.
    ///
    /// This is the last resort for a checkpoint whose architecture does not
    /// declare a format: the template is what actually teaches the model how to
    /// write a call, so it identifies the dialect even for an unknown model.
    public static func inferred(fromChatTemplate template: String) -> ToolCallFormat? {
        signatures.first { $0.matches(template) }?.format
    }

    /// The format named by a `tool_parser_type` entry, using the parser names
    /// mlx-lm publishes so a checkpoint carrying one is honored as written.
    public init?(toolParserType name: String) {
        switch name.lowercased() {
        case "minimax_m2": self = .minimaxM2
        case "gemma4": self = .gemma4
        case "function_gemma", "gemma": self = .gemma
        case "glm4", "glm47": self = .glm4
        case "pythonic", "lfm2": self = .lfm2
        case "qwen3_coder", "qwen3_xml", "xml_function": self = .xmlFunction
        case "kimi_k2": self = .kimiK2
        case "mistral": self = .mistral
        case "json_tools", "json": self = .json
        default: return nil
        }
    }
}

// MARK: - Inference from a checkpoint on disk

extension ToolCallFormat {

    /// Resolves a model declaration against the dialect selected by its checkpoint.
    ///
    /// An explicit `tool_parser_type` wins, matching mlx-lm, because it is the
    /// checkpoint author correcting what the template alone would imply. Otherwise,
    /// the selected tool template refines a model declaration that cannot parse the
    /// dialect it prompts. This matters for architecture heuristics such as Llama 3:
    /// Hermes checkpoints use that architecture while prompting framed JSON instead
    /// of Llama's inline JSON.
    ///
    /// A model-declared superset remains selected. For example, Qwen 3.5 prompts XML
    /// but its parser deliberately accepts both XML and framed Hermes JSON.
    package static func resolved(
        forTokenizerDirectory directory: URL,
        modelFormat: ToolCallFormat? = nil
    ) -> ToolCallFormat? {
        let configuration = TokenizerToolCallConfiguration(directory: directory)

        if let declared = configuration.toolParserType.flatMap(ToolCallFormat.init(toolParserType:))
        {
            return declared
        }

        guard
            let templateFormat = configuration.chatTemplate.flatMap(
                ToolCallFormat.inferred(fromChatTemplate:))
        else { return modelFormat }
        guard let modelFormat else { return templateFormat }

        return modelFormat.accepts(templateFormat: templateFormat) ? modelFormat : templateFormat
    }

    /// Whether a model-selected parser is a superset of the dialect prompted by
    /// the selected chat template.
    private func accepts(templateFormat: ToolCallFormat) -> Bool {
        switch self {
        case .qwen35:
            return templateFormat == .xmlFunction || templateFormat == .json
        case .gptOSS, .atem:
            // These select complete token-stream protocols, not interchangeable
            // payload parsers. Incidental textual markers must not reroute them.
            return true
        default:
            return self == templateFormat
        }
    }
}

/// The two tokenizer fields that describe tool-call syntax.
///
/// Read straight from disk rather than through the tokenizer, which exposes
/// only whether a chat template exists and not its text.
struct TokenizerToolCallConfiguration {
    let chatTemplate: String?
    let toolParserType: String?

    init(directory: URL) {
        let file = Self.decodeConfiguration(
            at: directory.appending(component: "tokenizer_config.json"))

        // Newer checkpoints ship the template as a sibling file instead.
        let sidecar = try? String(
            contentsOf: directory.appending(component: "chat_template.jinja"), encoding: .utf8)

        self.chatTemplate = file?.chatTemplate?.text ?? sidecar
        self.toolParserType = file?.toolParserType
    }

    private static func decodeConfiguration(at url: URL) -> ConfigurationFile? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ConfigurationFile.self, from: data)
    }

    private struct ConfigurationFile: Decodable {
        let chatTemplate: ChatTemplate?
        let toolParserType: String?

        enum CodingKeys: String, CodingKey {
            case chatTemplate = "chat_template"
            case toolParserType = "tool_parser_type"
        }
    }

    /// A template is either the template itself or a list of named templates.
    /// Format inference models a tool-enabled request, so it follows
    /// swift-transformers by preferring `tool_use` and falling back to `default`.
    enum ChatTemplate: Decodable {
        case single(String)
        case named([Named])

        struct Named: Decodable {
            let name: String
            let template: String
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let template = try? container.decode(String.self) {
                self = .single(template)
            } else {
                self = .named(try container.decode([Named].self))
            }
        }

        var text: String? {
            switch self {
            case .single(let template):
                return template
            case .named(let templates):
                return templates.first { $0.name == "tool_use" }?.template
                    ?? templates.first { $0.name == "default" }?.template
            }
        }
    }
}
