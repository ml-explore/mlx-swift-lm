// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

struct NullToolCallArgumentIntegrationTests {
    // transformers 5.23.0.dev0 with the same template, for
    // `"arguments": {"city": "Paris", "unit": None}`.
    private static let reference = """
        <|startoftext|><|im_start|>system
        <tools>
        {"name": "get_weather", "description": "Get the weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "description": "Temperature unit"}}, "required": ["city"]}}
        </tools><|im_end|>
        <|im_start|>user
        Weather in Paris?<|im_end|>
        <|im_start|>assistant
        <think></think><tool_call>
        <function=get_weather>
        <parameter=city>
        Paris
        </parameter>
        <parameter=unit>
        null
        </parameter>
        </function>
        </tool_call><|im_end|>
        <|im_start|>tool
        <tool_response>
        18C
        </tool_response><|im_end|>
        <|im_start|>user
        And tomorrow?<|im_end|>
        <|im_start|>assistant
        <think>
        """

    @Test func lfm3RendersNullArgumentLikeTransformers() async throws {
        let downloader: any Downloader = #hubDownloader()
        let loader: any TokenizerLoader = #huggingFaceTokenizerLoader()
        let directory = try await downloader.download(
            id: "LiquidAI/LFM3-1.7B-MLX-4bit",
            revision: "91f34088d6946482cbcae11b85d4657093618c34",
            matching: ["*.json", "*.jinja"], useLatest: false, progressHandler: { _ in })
        let tokenizer = try await loader.load(from: directory)

        let tool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string", "description": "City name"],
                        "unit": ["type": "string", "description": "Temperature unit"],
                    ] as [String: any Sendable],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let call = ToolCall(
            function: .init(
                name: "get_weather", arguments: ["city": .string("Paris"), "unit": .null]))
        let messages = DefaultMessageGenerator().generate(messages: [
            .user("Weather in Paris?"), .assistant("", toolCalls: [call]), .tool("18C"),
            .user("And tomorrow?"),
        ])
        let tokens = try tokenizer.applyChatTemplate(messages: messages, tools: [tool])
        let prompt = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)

        // swift-jinja sorts the keys of the tool JSON, so compare from the end of the tools block.
        let tail = { (text: String) in text.components(separatedBy: "</tools>").last }
        #expect(prompt.hasPrefix("<|startoftext|><|im_start|>system\n<tools>\n"))
        #expect(tail(prompt) == tail(Self.reference))
    }
}
