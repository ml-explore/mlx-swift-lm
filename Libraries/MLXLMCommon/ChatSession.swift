// Copyright © 2025 Apple Inc.

import CoreGraphics
import Foundation
import MLX

/// Simplified API for multi-turn conversations with LLMs and VLMs.
///
/// For example:
///
/// ```swift
/// let modelContainer = try await loadModelContainer(id: "mlx-community/Qwen3-4B-4bit")
/// let session = ChatSession(modelContainer)
/// print(try await session.respond(to: "What are two things to see in San Francisco?"))
/// print(try await session.respond(to: "How about a great place to eat?"))
/// ```
///
/// - Note: `ChatSession` is not thread-safe. Each session should be used from a single
///   task/thread at a time. The underlying `ModelContainer` handles thread safety for
///   model operations.
public final class ChatSession {

    private let model: ModelContainer
    private let instructions: String?
    private let cache: SerialAccessContainer<[KVCache]>
    private let processing: UserInput.Processing
    private let generateParameters: GenerateParameters
    private let additionalContext: [String: any Sendable]?

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init([])
        self.processing = processing
        self.generateParameters = generateParameters
        self.additionalContext = additionalContext
    }

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session
    ///   - generateParameters: parameters that control generation
    ///   - processing: media processing configuration for images/videos
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        generateParameters: GenerateParameters = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init([])
        self.processing = processing
        self.generateParameters = generateParameters
        self.additionalContext = additionalContext
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video]
    ) async throws -> String {
        // images and videos are not Sendable (MLXArray) but they are consumed
        // and are only being sent to the inner async
        let message = SendableBox<Chat.Message>(
            .user(prompt, images: images, videos: videos)
        )

        let output = try await model.perform {
            [
                instructions, processing, additionalContext, cache, generateParameters
            ] context in
            // wrap this to hop to the inner async
            let context = SendableBox(context)

            // with exclusive access to the cache
            return try await cache.update { cache in
                let context = context.consume()

                var messages: [Chat.Message] = []
                if let instructions {
                    messages.append(.system(instructions))
                }
                messages.append(message.consume())

                let userInput = UserInput(
                    chat: messages, processing: processing, additionalContext: additionalContext)
                let input = try await context.processor.prepare(input: userInput)

                if cache.isEmpty {
                    cache = context.model.newCache(parameters: generateParameters)
                }

                var output = ""
                for await generation in try MLXLMCommon.generate(
                    input: input, cache: cache, parameters: generateParameters, context: context
                ) {
                    if let chunk = generation.chunk {
                        output += chunk
                    }
                }

                Stream().synchronize()

                return output
            }
        }
        return output
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    /// - Returns: the model's response
    public func respond(
        to prompt: String,
        image: UserInput.Image? = nil,
        video: UserInput.Video? = nil
    ) async throws -> String {
        try await respond(
            to: prompt,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? []
        )
    }

    /// Produces a streaming response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        images: consuming [UserInput.Image],
        videos: consuming [UserInput.Video]
    ) -> AsyncThrowingStream<String, Error> {
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()

        // images and videos are not Sendable (MLXArray) but they are consumed
        // and are only being sent to the inner async
        let message = SendableBox<Chat.Message>(
            .user(prompt, images: images, videos: videos)
        )

        let task = Task {
            [
                model,
                instructions, processing, additionalContext, cache, generateParameters
            ] in
            do {
                try await model.perform {
                    [
                        instructions, processing, additionalContext, cache, generateParameters
                    ] context in
                    // wrap this to hop to the inner async
                    let context = SendableBox(context)

                    try await cache.update { cache in
                        let context = context.consume()

                        var messages: [Chat.Message] = []
                        if let instructions {
                            messages.append(.system(instructions))
                        }
                        messages.append(message.consume())

                        let userInput = UserInput(
                            chat: messages, processing: processing,
                            additionalContext: additionalContext)
                        let input = try await context.processor.prepare(input: userInput)

                        if cache.isEmpty {
                            cache = context.model.newCache(parameters: generateParameters)
                        }

                        let iterator = try TokenIterator(
                            input: input, model: context.model, cache: cache,
                            parameters: generateParameters)

                        let (stream, task) = MLXLMCommon.generateTask(
                            input: input, context: context, iterator: iterator
                        )

                        var fullResponse = ""
                        for await item in stream {
                            if let chunk = item.chunk {
                                fullResponse += chunk
                                if case .terminated = continuation.yield(chunk) {
                                    break
                                }
                            }
                        }

                        await task.value

                        continuation.finish()
                    }
                }
            } catch {
                continuation.finish(throwing: error)
            }
        }

        continuation.onTermination = { _ in
            task.cancel()
        }

        return stream
    }

    /// Produces a streaming response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        image: UserInput.Image? = nil,
        video: UserInput.Video? = nil
    ) -> AsyncThrowingStream<String, Error> {
        streamResponse(
            to: prompt,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? []
        )
    }

    /// Clear the session history and cache, preserving system instructions.
    public func clear() async {
        await cache.update { cache in
            cache = []
        }
    }

    public func synchronize() async {
        await cache.read { _ in }
    }
}
