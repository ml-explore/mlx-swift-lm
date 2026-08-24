// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Tool-call format inference")
struct ToolCallFormatInferenceTests {

    // MARK: - Chat template signatures

    @Test(
        "Each dialect is recognized by its template markers",
        arguments: [
            ("<minimax:tool_call>{...}</minimax:tool_call>", ToolCallFormat.minimaxM2),
            (#"<|tool_call>call:{{ name }}{...}<tool_call|>"#, .gemma4),
            ("<start_function_call>call:{{ name }}<end_function_call>", .gemma),
            ("<arg_key>{{ key }}</arg_key><arg_value>{{ value }}</arg_value>", .glm4),
            ("<|tool_list_start|>[{{ tools }}]<|tool_list_end|>", .lfm2),
            ("<tool_call>\n<function={{ name }}>", .xmlFunction),
            ("<|tool_calls_section_begin|>functions.{{ name }}", .kimiK2),
            ("[TOOL_CALLS]{{ name }}[ARGS]", .mistral),
            (#"<tool_call>{"name": "{{ tool_call.name }}"}</tool_call>"#, .json),
        ] as [(String, ToolCallFormat)]
    )
    func recognizesDialect(template: String, expected: ToolCallFormat) {
        #expect(ToolCallFormat.inferred(fromChatTemplate: template) == expected)
    }

    @Test("A template escaping its newline still selects the XML dialect")
    func recognizesEscapedNewlineInXMLTemplate() {
        #expect(
            ToolCallFormat.inferred(fromChatTemplate: #"<tool_call>\n<function={{ name }}>"#)
                == .xmlFunction)
    }

    @Test("The XML dialect outranks the generic JSON signature")
    func specificSignatureWinsOverGeneric() {
        // A Qwen3-Coder style template mentions both; the JSON signature is the
        // broadest and must not shadow the dialect the model actually emits.
        let template = """
            {% if tool_call.name %}<tool_call>
            <function={{ tool_call.name }}></tool_call>{% endif %}
            """
        #expect(ToolCallFormat.inferred(fromChatTemplate: template) == .xmlFunction)
    }

    @Test("A template with no tool syntax infers nothing")
    func templateWithoutToolSyntaxInfersNothing() {
        #expect(
            ToolCallFormat.inferred(fromChatTemplate: "{{ messages[0]['content'] }}") == nil)
        #expect(ToolCallFormat.inferred(fromChatTemplate: "") == nil)
    }

    // MARK: - Declared parser names

    @Test(
        "A declared parser name maps to its dialect",
        arguments: [
            ("qwen3_coder", ToolCallFormat.xmlFunction),
            ("qwen3_xml", .xmlFunction),
            ("function_gemma", .gemma),
            ("gemma4", .gemma4),
            ("glm47", .glm4),
            ("pythonic", .lfm2),
            ("kimi_k2", .kimiK2),
            ("mistral", .mistral),
            ("minimax_m2", .minimaxM2),
            ("json_tools", .json),
        ] as [(String, ToolCallFormat)]
    )
    func mapsDeclaredParserName(name: String, expected: ToolCallFormat) {
        #expect(ToolCallFormat(toolParserType: name) == expected)
    }

    @Test("An unknown parser name maps to nothing")
    func unknownParserNameMapsToNothing() {
        #expect(ToolCallFormat(toolParserType: "not_a_parser") == nil)
    }

    // MARK: - Reading a checkpoint

    @Test("The chat template in the tokenizer config is used")
    func readsTemplateFromTokenizerConfig() throws {
        let directory = try TokenizerFixture.make([
            "tokenizer_config.json": #"{"chat_template": "[TOOL_CALLS]{{ name }}[ARGS]"}"#
        ])
        defer { TokenizerFixture.remove(directory) }

        #expect(ToolCallFormat.resolved(forTokenizerDirectory: directory) == .mistral)
    }

    @Test("A named template list prefers tool_use for format inference")
    func namedTemplateListPrefersToolUse() throws {
        let directory = try TokenizerFixture.make([
            "tokenizer_config.json": #"""
            {"chat_template": [
                {"name": "tool_use", "template": "<arg_key>x</arg_key>"},
                {"name": "default", "template": "<|tool_list_start|>"}
            ]}
            """#
        ])
        defer { TokenizerFixture.remove(directory) }

        #expect(ToolCallFormat.resolved(forTokenizerDirectory: directory) == .glm4)
    }

    @Test("Hermes tool_use refines the heuristic Llama 3 declaration to framed JSON")
    func hermesTemplateListRefinesLlama3Format() throws {
        let directory = try TokenizerFixture.make([
            "tokenizer_config.json": #"""
            {"chat_template": [
                {"name": "default", "template": "{{ bos_token }}{{ messages }}"},
                {"name": "tool_use", "template": "For each function call return a json object within <tool_call></tool_call>: <tool_call>{\"name\": {{ tool_call.name }}, \"arguments\": {}}</tool_call>"}
            ]}
            """#
        ])
        defer { TokenizerFixture.remove(directory) }

        let format = ToolCallFormat.resolved(
            forTokenizerDirectory: directory, modelFormat: .llama3)
        #expect(format == .json)

        let processor = ToolCallProcessor(format: try #require(format))
        #expect(processor.processChunk("<tool_call>\n") == nil)
        #expect(
            processor.processChunk(
                #"{"name":"get_weather","arguments":{"location":"Paris"}}"#) == nil)
        #expect(processor.processChunk("\n</tool_call>") == nil)
        #expect(processor.toolCalls.count == 1)
        let call = try #require(processor.toolCalls.first)
        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments == ["location": .string("Paris")])
        #expect(call.id != nil)
    }

    @Test("A compatible model superset is preserved over template inference")
    func compatibleModelFormatIsPreserved() throws {
        let xmlDirectory = try TokenizerFixture.make([
            "tokenizer_config.json": #"{"chat_template":"<tool_call>\n<function={{ name }}>"}"#
        ])
        defer { TokenizerFixture.remove(xmlDirectory) }

        let jsonDirectory = try TokenizerFixture.make([
            "tokenizer_config.json":
                #"{"chat_template":"<tool_call>{{ tool_call.name }}</tool_call>"}"#
        ])
        defer { TokenizerFixture.remove(jsonDirectory) }

        #expect(
            ToolCallFormat.resolved(
                forTokenizerDirectory: xmlDirectory, modelFormat: .qwen35) == .qwen35)
        #expect(
            ToolCallFormat.resolved(
                forTokenizerDirectory: jsonDirectory, modelFormat: .qwen35) == .qwen35)
    }

    @Test("A model declaration is used when the selected template has no known dialect")
    func modelFormatIsFallbackForUnknownTemplate() throws {
        let directory = try TokenizerFixture.make([
            "tokenizer_config.json": #"{"chat_template":"{{ messages }}"}"#
        ])
        defer { TokenizerFixture.remove(directory) }

        #expect(
            ToolCallFormat.resolved(
                forTokenizerDirectory: directory, modelFormat: .llama3) == .llama3)
    }

    @Test("A complete response protocol is not replaced by payload inference")
    func protocolFormatIsAuthoritative() throws {
        let directory = try TokenizerFixture.make([
            "tokenizer_config.json":
                #"{"chat_template":"<tool_call>{{ tool_call.name }}</tool_call>"}"#
        ])
        defer { TokenizerFixture.remove(directory) }

        #expect(
            ToolCallFormat.resolved(
                forTokenizerDirectory: directory, modelFormat: .gptOSS) == .gptOSS)
        #expect(
            ToolCallFormat.resolved(
                forTokenizerDirectory: directory, modelFormat: .atem) == .atem)
    }

    @Test("A named template list falls back to default without tool_use")
    func namedTemplateListFallsBackToDefault() throws {
        let directory = try TokenizerFixture.make([
            "tokenizer_config.json": #"""
            {"chat_template": [
                {"name": "other", "template": "<arg_key>x</arg_key>"},
                {"name": "default", "template": "<|tool_list_start|>"}
            ]}
            """#
        ])
        defer { TokenizerFixture.remove(directory) }

        #expect(ToolCallFormat.resolved(forTokenizerDirectory: directory) == .lfm2)
    }

    @Test("An unselected named template is not used for inference")
    func namedTemplateListWithoutAutomaticChoiceResolvesToNothing() throws {
        let directory = try TokenizerFixture.make([
            "tokenizer_config.json": #"""
            {"chat_template": [
                {"name": "other", "template": "<arg_key>x</arg_key>"}
            ]}
            """#
        ])
        defer { TokenizerFixture.remove(directory) }

        #expect(ToolCallFormat.resolved(forTokenizerDirectory: directory) == nil)
    }

    @Test("A sidecar chat template file is used when the config has none")
    func readsSidecarTemplateFile() throws {
        let directory = try TokenizerFixture.make([
            "tokenizer_config.json": #"{"eos_token": "</s>"}"#,
            "chat_template.jinja": "<|tool_calls_section_begin|>",
        ])
        defer { TokenizerFixture.remove(directory) }

        #expect(ToolCallFormat.resolved(forTokenizerDirectory: directory) == .kimiK2)
    }

    @Test("A declared parser type overrides what the template implies")
    func declaredParserTypeWinsOverTemplate() throws {
        let directory = try TokenizerFixture.make([
            "tokenizer_config.json": #"""
            {"tool_parser_type": "qwen3_coder", "chat_template": "[TOOL_CALLS]"}
            """#
        ])
        defer { TokenizerFixture.remove(directory) }

        #expect(
            ToolCallFormat.resolved(
                forTokenizerDirectory: directory, modelFormat: .llama3) == .xmlFunction)
    }

    @Test("A checkpoint with nothing to go on resolves to nothing")
    func missingOrUnreadableFilesResolveToNothing() throws {
        let empty = try TokenizerFixture.make([:])
        defer { TokenizerFixture.remove(empty) }
        #expect(ToolCallFormat.resolved(forTokenizerDirectory: empty) == nil)

        let malformed = try TokenizerFixture.make(["tokenizer_config.json": "{not json"])
        defer { TokenizerFixture.remove(malformed) }
        #expect(ToolCallFormat.resolved(forTokenizerDirectory: malformed) == nil)

        #expect(
            ToolCallFormat.resolved(
                forTokenizerDirectory: URL(fileURLWithPath: "/nonexistent-checkpoint")) == nil)
    }
}

/// A throwaway directory holding tokenizer files.
private enum TokenizerFixture {
    static func make(_ files: [String: String]) throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(component: "tool-format-inference-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (name, contents) in files {
            try contents.write(
                to: directory.appending(component: name), atomically: true, encoding: .utf8)
        }
        return directory
    }

    static func remove(_ directory: URL) {
        try? FileManager.default.removeItem(at: directory)
    }
}
