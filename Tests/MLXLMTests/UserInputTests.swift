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

    public func testGemma4ConversionText() {
        let chat: [Chat.Message] = [
            .system("You are a useful agent."),
            .user("Tell me a story."),
        ]

        let messages = Gemma4MessageGenerator().generate(messages: chat)

        let expected: [[String: any Sendable]] = [
            [
                "role": "system",
                "content": "You are a useful agent.",
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

    public func testCustomMessageGeneratorsPreserveToolMetadata() throws {
        let toolCall = ToolCall(
            function: .init(
                name: "get_weather",
                arguments: [
                    "location": .string("Paris")
                ]
            ),
            id: "call_custom_123"
        )
        let assistant = Chat.Message.assistant("Checking weather", toolCalls: [toolCall])
        let toolResult = Chat.Message.tool(#"{"temperature":18}"#, id: "call_custom_123")

        let generators: [(String, (Chat.Message) -> MLXLMCommon.Message)] = [
            ("Qwen2VL", { Qwen2VLMessageGenerator().generate(message: $0) }),
            ("Qwen3VL", { Qwen3VLMessageGenerator().generate(message: $0) }),
            ("Gemma4", { Gemma4MessageGenerator().generate(message: $0) }),
            ("FastVLM", { FastVLMMessageGenerator().generate(message: $0) }),
            ("GlmOcr", { GlmOcrMessageGenerator().generate(message: $0) }),
            ("Mistral3", { Mistral3MessageGenerator().generate(message: $0) }),
        ]

        for (name, generate) in generators {
            let rawAssistant = generate(assistant)
            XCTAssertEqual(rawAssistant["role"] as? String, "assistant", name)
            XCTAssertNotNil(rawAssistant["content"], name)

            let calls = try XCTUnwrap(
                rawAssistant["tool_calls"] as? [[String: any Sendable]], name)
            XCTAssertEqual(calls.count, 1, name)
            XCTAssertEqual(calls[0]["id"] as? String, "call_custom_123", name)
            XCTAssertEqual(calls[0]["type"] as? String, "function", name)

            let function = try XCTUnwrap(calls[0]["function"] as? [String: any Sendable], name)
            XCTAssertEqual(function["name"] as? String, "get_weather", name)
            let arguments = try XCTUnwrap(function["arguments"] as? [String: any Sendable], name)
            XCTAssertEqual(arguments["location"] as? String, "Paris", name)

            let rawToolResult = generate(toolResult)
            XCTAssertEqual(rawToolResult["role"] as? String, "tool", name)
            XCTAssertNotNil(rawToolResult["content"], name)
            XCTAssertEqual(rawToolResult["tool_call_id"] as? String, "call_custom_123", name)
        }
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
                        "type": "image"
                    ],
                    [
                        "type": "text",
                        "text": "What is this?",
                    ],
                ],
            ],
        ]

        assertEqual(expected, messages)

        let userInput = UserInput(chat: chat)
        XCTAssertEqual(userInput.images.count, 1)
    }

    public func testGemma4ConversionImage() {
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

        let messages = Gemma4MessageGenerator().generate(messages: chat)

        let expected: [[String: any Sendable]] = [
            [
                "role": "system",
                "content": "You are a useful agent.",
            ],
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
            ],
        ]

        assertEqual(expected, messages)
    }

    // MARK: - Init self.images / self.videos sync (#182)

    public func testInitFromPromptStringPopulatesImages() throws {
        // Reproducer for #182: a `.chat` prompt is built from the parameters
        // but `prompt.didSet` does not fire during init, so `self.images`
        // used to stay empty.
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let placeholder = CIImage(
            color: CIColor(red: 0.5, green: 0.5, blue: 0.5, colorSpace: cs)!
        ).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))

        let input = UserInput(
            prompt: "What is in this image?",
            images: [.ciImage(placeholder)])
        XCTAssertEqual(
            input.images.count, 1,
            "UserInput(prompt:images:) must surface the images parameter on self.images")
        XCTAssertEqual(input.videos.count, 0)
    }

    public func testInitFromPromptEnumPopulatesImagesForChat() throws {
        // The `init(prompt: Prompt, images:, videos:, ...)` overload also had
        // `case .chat: break` and dropped the images parameter on the floor.
        // After the fix it derives images from the chat messages instead.
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let placeholder = CIImage(
            color: CIColor(red: 0.5, green: 0.5, blue: 0.5, colorSpace: cs)!
        ).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))

        let chat: [Chat.Message] = [
            .user("describe", images: [.ciImage(placeholder)])
        ]
        let input = UserInput(prompt: .chat(chat))
        XCTAssertEqual(
            input.images.count, 1,
            "UserInput(prompt:.chat) must derive self.images from the chat messages")
    }

    public func testInitFromPromptStringPopulatesVideos() throws {
        // Same bug, video edition.
        let videoURL = URL(fileURLWithPath: "/tmp/nonexistent.mp4")
        let input = UserInput(
            prompt: "describe this video",
            videos: [.url(videoURL)])
        XCTAssertEqual(input.videos.count, 1)
        XCTAssertEqual(input.images.count, 0)
    }

    // MARK: - GPT-OSS Message Generator Tests

    public func testGPTOSSMessageGeneratorSerializesStructuredToolCalls() {
        let call = ToolCall(
            function: .init(
                name: "web_search", arguments: ["query": "latest news Dubai"]),
            id: "call_1")
        let chat: [Chat.Message] = [
            .assistant("", toolCalls: [call]),
            .tool("{\"results\":[]}", id: "call_1"),
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
        XCTAssertEqual(messages[1]["tool_call_id"] as? String, "call_1")
    }

    public func testGPTOSSMessageGeneratorDoesNotInferToolCallsFromContent() {
        let jsonish = "{\"name\":\"functions.web_search\",\"arguments\":{\"query\":\"x\"}}"
        let messages = GPTOSSMessageGenerator().generate(messages: [.assistant(jsonish)])
        XCTAssertNil(messages[0]["tool_calls"])
        XCTAssertEqual(messages[0]["content"] as? String, jsonish)
    }

    public func testGPTOSSMessageGeneratorKeepsPlainAssistantContent() {
        let chat: [Chat.Message] = [.assistant("I will look that up.")]

        let messages = GPTOSSMessageGenerator().generate(messages: chat)

        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(messages[0]["role"] as? String, "assistant")
        XCTAssertEqual(messages[0]["content"] as? String, "I will look that up.")
        XCTAssertNil(messages[0]["tool_calls"])
    }

    // MARK: - VideoProcessing Tests

    public func testVideoProcessingDefaultInit() {
        let vp = UserInput.VideoProcessing()
        XCTAssertNil(vp.sampling)
        XCTAssertNil(vp.targetFrames)
        XCTAssertNil(vp.targetFramesPerSecond)
    }

    public func testVideoProcessingTargetFrames() {
        var vp = UserInput.VideoProcessing(targetFrames: 16)
        XCTAssertEqual(vp.sampling, .targetFrames(16))
        XCTAssertEqual(vp.targetFrames, 16)
        XCTAssertNil(vp.targetFramesPerSecond)

        vp.targetFrames = 32
        XCTAssertEqual(vp.sampling, .targetFrames(32))
        XCTAssertEqual(vp.targetFrames, 32)

        vp.targetFrames = nil
        XCTAssertNil(vp.sampling)
        XCTAssertNil(vp.targetFrames)
    }

    public func testVideoProcessingFramesPerSecond() {
        var vp = UserInput.VideoProcessing(targetFramesPerSecond: 2.5)
        XCTAssertEqual(vp.sampling, .framesPerSecond(2.5))
        XCTAssertEqual(vp.targetFramesPerSecond, 2.5)
        XCTAssertNil(vp.targetFrames)

        vp.targetFramesPerSecond = 5.0
        XCTAssertEqual(vp.sampling, .framesPerSecond(5.0))
        XCTAssertEqual(vp.targetFramesPerSecond, 5.0)

        vp.targetFramesPerSecond = nil
        XCTAssertNil(vp.sampling)
        XCTAssertNil(vp.targetFramesPerSecond)
    }

    public func testVideoProcessingSamplingEnum() {
        let vp1 = UserInput.VideoProcessing(sampling: .targetFrames(8))
        XCTAssertEqual(vp1.targetFrames, 8)
        XCTAssertNil(vp1.targetFramesPerSecond)

        let vp2 = UserInput.VideoProcessing(sampling: .framesPerSecond(1.0))
        XCTAssertEqual(vp2.targetFramesPerSecond, 1.0)
        XCTAssertNil(vp2.targetFrames)
    }

    public func testUserInputProcessingWithVideo() {
        let processing = UserInput.Processing(
            video: .init(targetFrames: 24),
            minPixels: 100,
            maxPixels: 500
        )
        XCTAssertEqual(processing.video.targetFrames, 24)
        XCTAssertEqual(processing.minPixels, 100)
        XCTAssertEqual(processing.maxPixels, 500)

        let input = UserInput(chat: [.user("hello")], processing: processing)
        XCTAssertEqual(input.processing.video.targetFrames, 24)
    }

}
