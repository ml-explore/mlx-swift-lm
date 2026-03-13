import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

@testable import MLXLLM

func assertEqual(
    _ v1: Any, _ v2: Any, path: [String] = [], file: StaticString = #filePath, line: UInt = #line
) {
    switch (v1, v2) {
    case let (v1, v2) as (String, String):
        XCTAssertEqual(v1, v2, file: file, line: line)

    case let (v1, v2) as ([Any], [Any]):
        XCTAssertEqual(
            v1.count, v2.count, "Arrays not equal size at \(path)", file: file, line: line)

        for (index, (v1v, v2v)) in zip(v1, v2).enumerated() {
            assertEqual(v1v, v2v, path: path + [index.description], file: file, line: line)
        }

    case let (v1, v2) as ([String: any Sendable], [String: any Sendable]):
        XCTAssertEqual(
            v1.keys.sorted(), v2.keys.sorted(),
            "\(String(describing: v1.keys.sorted())) and \(String(describing: v2.keys.sorted())) not equal at \(path)",
            file: file, line: line)

        for (k, v1v) in v1 {
            if let v2v = v2[k] {
                assertEqual(v1v, v2v, path: path + [k], file: file, line: line)
            } else {
                XCTFail("Missing value for \(k) at \(path)", file: file, line: line)
            }
        }
    default:
        XCTFail(
            "Unable to compare \(String(describing: v1)) and \(String(describing: v2)) at \(path)",
            file: file, line: line)
    }
}

public class UserInputTests: XCTestCase {

    public func testStandardConversion() {
        let chat: [Chat.Message] = [
            .system("You are a useful agent."),
            .user("Tell me a story."),
        ]

        let messages = DefaultMessageGenerator().generate(messages: chat)

        let expected = [
            [
                "role": "system",
                "content": "You are a useful agent.",
            ],
            [
                "role": "user",
                "content": "Tell me a story.",
            ],
        ]

        XCTAssertEqual(expected, messages as? [[String: String]])
    }

    public func testQwen2ConversionText() {
        let chat: [Chat.Message] = [
            .system("You are a useful agent."),
            .user("Tell me a story."),
        ]

        let messages = Qwen2VLMessageGenerator().generate(messages: chat)

        let expected = [
            [
                "role": "system",
                "content": [
                    [
                        "type": "text",
                        "text": "You are a useful agent.",
                    ]
                ],
            ],
            [
                "role": "user",
                "content": [
                    [
                        "type": "text",
                        "text": "Tell me a story.",
                    ]
                ],
            ],
        ]

        assertEqual(expected, messages)
    }

    // MARK: - Mistral3 Message Generator Tests

    public func testMistral3ConversionText() {
        let chat: [Chat.Message] = [
            .system("You are a useful agent."),
            .user("Tell me a story."),
        ]

        let messages = Mistral3MessageGenerator().generate(messages: chat)

        let expected: [[String: any Sendable]] = [
            [
                "role": "system",
                "content": [
                    [
                        "type": "text",
                        "text": "You are a useful agent.",
                    ]
                ],
            ],
            [
                "role": "user",
                "content": [
                    [
                        "type": "text",
                        "text": "Tell me a story.",
                    ]
                ],
            ],
        ]

        assertEqual(expected, messages)
    }

    public func testMistral3ConversionWithImage() {
        let chat: [Chat.Message] = [
            .user(
                "What is this?",
                images: [
                    .url(
                        URL(
                            string: "https://opensource.apple.com/images/projects/mlx.f5c59d8b.png")!
                    )
                ])
        ]

        let messages = Mistral3MessageGenerator().generate(messages: chat)

        let expected: [[String: any Sendable]] = [
            [
                "role": "user",
                "content": [
                    [
                        "type": "image"
                    ],
                    [
                        "type": "text",
                        "text": "What is this?",
                    ],
                ],
            ]
        ]

        assertEqual(expected, messages)
    }

    public func testMistral3ConversionToolRole() {
        let chat: [Chat.Message] = [
            .tool("The weather is sunny, 14°C.")
        ]

        let messages = Mistral3MessageGenerator().generate(messages: chat)

        let expected: [[String: any Sendable]] = [
            [
                "role": "tool",
                "content": [
                    [
                        "type": "text",
                        "text": "The weather is sunny, 14°C.",
                    ]
                ],
            ]
        ]

        assertEqual(expected, messages)
    }

    // MARK: - Qwen2 Message Generator Tests

    public func testQwen2ConversionImage() {
        let chat: [Chat.Message] = [
            .system("You are a useful agent."),
            .user(
                "What is this?",
                images: [
                    .url(
                        URL(
                            string: "https://opensource.apple.com/images/projects/mlx.f5c59d8b.png")!
                    )
                ]),
        ]

        let messages = Qwen2VLMessageGenerator().generate(messages: chat)

        let expected = [
            [
                "role": "system",
                "content": [
                    [
                        "type": "text",
                        "text": "You are a useful agent.",
                    ]
                ],
            ],
            [
                "role": "user",
                "content": [
                    [
                        "type": "text",
                        "text": "What is this?",
                    ],
                    [
                        "type": "image"
                    ],
                ],
            ],
        ]

        assertEqual(expected, messages)

        let userInput = UserInput(chat: chat)
        XCTAssertEqual(userInput.images.count, 1)
    }

    // MARK: - GPT-OSS Message Generator Tests

    public func testGPTOSSMessageGeneratorConvertsStoredAssistantToolCalls() {
        let chat: [Chat.Message] = [
            .assistant(
                "{\"name\":\"functions.web_search\",\"arguments\":{\"query\":\"latest news Dubai\"}}"
            ),
            .tool("{\"results\":[]}"),
        ]

        let messages = GPTOSSMessageGenerator().generate(messages: chat)

        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[0]["role"] as? String, "assistant")

        guard let toolCalls = messages[0]["tool_calls"] as? [[String: any Sendable]] else {
            XCTFail("Missing assistant.tool_calls")
            return
        }
        XCTAssertEqual(toolCalls.count, 1)

        guard let function = toolCalls[0]["function"] as? [String: any Sendable] else {
            XCTFail("Missing function payload")
            return
        }
        XCTAssertEqual(function["name"] as? String, "web_search")

        guard let arguments = function["arguments"] as? [String: any Sendable] else {
            XCTFail("Missing function.arguments")
            return
        }
        XCTAssertEqual(arguments["query"] as? String, "latest news Dubai")

        XCTAssertEqual(messages[1]["role"] as? String, "tool")
        XCTAssertEqual(messages[1]["content"] as? String, "{\"results\":[]}")
    }

    public func testGPTOSSMessageGeneratorKeepsPlainAssistantContent() {
        let chat: [Chat.Message] = [
            .assistant("I will look that up."),
            .tool("{\"results\":[]}"),
        ]

        let messages = GPTOSSMessageGenerator().generate(messages: chat)

        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[0]["role"] as? String, "assistant")
        XCTAssertEqual(messages[0]["content"] as? String, "I will look that up.")
        XCTAssertNil(messages[0]["tool_calls"])
        XCTAssertEqual(messages[1]["role"] as? String, "tool")
    }

}
