import Foundation
import Testing
import Tokenizers

// Opt in with swift-transformers 1.4.0 or another tokenizer with #397 and #398.
@Suite(
    .serialized, .enabled(if: ProcessInfo.processInfo.environment["MLX_RUN_LFM3_TOKENIZER"] == "1"))
struct LFM3TokenizerIntegrationTests {
    private func tokenizer() async throws -> any Tokenizers.Tokenizer {
        if let path = ProcessInfo.processInfo.environment["MLX_LFM3_TOKENIZER_DIRECTORY"] {
            return try await AutoTokenizer.from(modelFolder: URL(filePath: path))
        }
        return try await AutoTokenizer.from(
            pretrained: "LiquidAI/LFM3-1.7B", revision: "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7")
    }

    // Goldens from tokenizers 0.23.3 and the pinned LiquidAI tokenizer.
    @Test func tokenIDsMatchHuggingFace() async throws {
        let tokenizer = try await tokenizer()
        let cases: [(String, [Int])] = [
            (".reduce", [96744]),
            ("line.\r\nThis", [1421, 64122, 1942]),
            ("Hello, world!", [35808, 20, 1530, 9]),
            ("Reading\nCongratulations", [49973, 207, 44069, 2867, 4461]),
            ("0\u{fe0f}\u{20e3}", [24, 47353, 167, 234, 105]),
            ("¡Hola! 日本語 中文 العربية", [32245, 48, 4105, 9, 37054, 12673, 50525, 4966, 54206]),
            (
                "let sum = values.reduce(0, +)",
                [1161, 9632, 550, 3895, 96744, 16, 24, 20, 1510, 17]
            ),
        ]
        for (text, expected) in cases {
            #expect(tokenizer.encode(text: text, addSpecialTokens: false) == expected)
        }
        #expect(tokenizer.convertTokenToId("<|im_end|>") == 124900)
        #expect(tokenizer.convertTokenToId("<tool_call>") == 125019)
        #expect(tokenizer.convertTokenToId("</tool_call>") == 125020)
    }

    @Test func chatAndToolTemplatesMatchHuggingFace() async throws {
        let tokenizer = try await tokenizer()
        let plain = try tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": "Hello"]
        ])
        #expect(
            plain == [124894, 124899, 5922, 207, 35808, 124900, 207, 124899, 63514, 207, 124901])
        let disabled = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "Hello"]],
            tools: nil, additionalContext: ["enable_thinking": false])
        #expect(
            tokenizer.decode(tokens: disabled, skipSpecialTokens: false).hasSuffix(
                "<think></think>"))
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "Be precise."],
            ["role": "user", "content": "Weather?"],
            [
                "role": "assistant",
                "tool_calls": [
                    [
                        "type": "function",
                        "function": [
                            "name": "get_weather", "arguments": ["location": "Paris"],
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ],
            ],
            ["role": "tool", "content": "Sunny"],
            ["role": "user", "content": "Thanks"],
        ]
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "parameters": [
                        "type": "object", "properties": ["location": ["type": "string"]],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]
        let actual = try tokenizer.applyChatTemplate(messages: messages, tools: tools)
        // Nested schema keys use the same sorted order as Swift Jinja tojson.
        #expect(
            actual == [
                124894, 124899, 23630, 207, 4184, 16845, 63908, 125017, 207, 38184, 2554, 6380, 496,
                1265, 71, 51492, 1377, 496, 29701, 6380, 36579, 38805, 6380, 36579, 22061, 6380,
                36579, 5882, 6380, 496, 3987, 87961, 496, 5882, 6380, 496, 7022, 115440, 125018,
                124900, 207, 124899, 5922, 207, 97056, 39, 124900, 207, 124899, 63514, 207, 124901,
                124902, 125019, 207, 125021, 1265, 71, 51492, 703, 125023, 22061, 703, 41176, 207,
                125024, 207, 125022, 207, 125020, 124900, 207, 124899, 46765, 207, 125025, 207,
                35243, 3905, 207, 125026, 124900, 207, 124899, 5922, 207, 29163, 124900, 207,
                124899, 63514, 207, 124901,
            ])
    }
}
