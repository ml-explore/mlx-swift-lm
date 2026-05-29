// Copyright © 2025 Apple Inc.

import Foundation

public enum Chat {
    public struct Message {
        /// The role of the message sender.
        public var role: Role

        /// The content of the message.
        public var content: String

        /// Array of image data associated with the message.
        public var images: [UserInput.Image]

        /// Array of video data associated with the message.
        public var videos: [UserInput.Video]

        /// For `.tool` messages: the id of the tool call this message answers.
        public var toolCallId: String?

        /// For `.assistant` messages: the tool calls this turn emitted.
        public var toolCalls: [ToolCall]?

        public init(
            role: Role, content: String, images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [],
            toolCallId: String? = nil,
            toolCalls: [ToolCall]? = nil
        ) {
            self.role = role
            self.content = content
            self.images = images
            self.videos = videos
            self.toolCallId = toolCallId
            self.toolCalls = toolCalls
        }

        public static func system(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []
        ) -> Self {
            Self(role: .system, content: content, images: images, videos: videos)
        }

        public static func assistant(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = [],
            toolCalls: [ToolCall]? = nil
        ) -> Self {
            Self(
                role: .assistant, content: content, images: images, videos: videos,
                toolCalls: toolCalls)
        }

        public static func user(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []
        ) -> Self {
            Self(role: .user, content: content, images: images, videos: videos)
        }

        public static func tool(_ content: String, id: String? = nil) -> Self {
            Self(role: .tool, content: content, toolCallId: id)
        }

        public enum Role: String, Sendable {
            case user
            case assistant
            case system
            case tool
        }
    }
}

/// Protocol for something that can convert structured
/// ``Chat/Message`` into model specific ``Message``
/// (raw dictionary) format.
///
/// Typically this is owned and used by a ``UserInputProcessor``:
///
/// ```swift
/// public func prepare(input: UserInput) async throws -> LMInput {
///     let messages = Qwen2VLMessageGenerator().generate(from: input)
///     ...
/// ```
public protocol MessageGenerator: Sendable {

    /// Generates messages from the input.
    func generate(from input: UserInput) -> [Message]

    /// Returns array of `[String: any Sendable]` aka ``Message``
    func generate(messages: [Chat.Message]) -> [Message]

    /// Returns `[String: any Sendable]`, aka ``Message``.
    func generate(message: Chat.Message) -> Message
}

extension MessageGenerator {

    public func generate(message: Chat.Message) -> Message {
        var dict: Message = [
            "role": message.role.rawValue,
            "content": message.content,
        ]
        if let id = message.toolCallId {
            dict["tool_call_id"] = id
        }
        if let calls = message.toolCalls {
            dict["tool_calls"] = calls.map { call -> [String: any Sendable] in
                var entry: [String: any Sendable] = [
                    "type": "function",
                    "function": [
                        "name": call.function.name,
                        "arguments": call.function.arguments.mapValues { $0.sendableValue },
                    ] as [String: any Sendable],
                ]
                if let id = call.id { entry["id"] = id }
                return entry
            }
        }
        return dict
    }

    public func generate(messages: [Chat.Message]) -> [Message] {
        var rawMessages: [Message] = []

        for message in messages {
            let raw = generate(message: message)
            rawMessages.append(raw)
        }

        return rawMessages
    }

    public func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            generate(messages: [.user(text)])
        case .messages(let messages):
            messages
        case .chat(let messages):
            generate(messages: messages)
        }
    }
}

/// Default implementation of ``MessageGenerator`` that produces a
/// `role`, `content`, `tool_call_id`, and `tool_calls` using default implementation.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct DefaultMessageGenerator: MessageGenerator {
    public init() {}
}

/// Implementation of ``MessageGenerator`` that produces a
/// `role` and `content` but omits `system` roles.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct NoSystemMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(messages: [Chat.Message]) -> [Message] {
        messages
            .filter { $0.role != .system }
            .map { generate(message: $0) }
    }
}
