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
    private var messages: [Chat.Message]
    private var cache: [KVCache]
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
        self.messages = instructions.map { [.system($0)] } ?? []
        self.cache = []
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
        self.messages = instructions.map { [.system($0)] } ?? []
        self.cache = []
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
        messages.append(.user(prompt, images: images, videos: videos))

        // TODO dkoski
        // TODO dkoski -- images and videos are not Sendable because they might be MLXArray
        // TODO dkoski -- also the messages passed should just be system prompt + user message.  kvcache handles the rest
        let output = try await model.perform { container in
            let context = container.context
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
        messages.append(.assistant(output))
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
        images: [UserInput.Image],
        videos: [UserInput.Video]
    ) -> AsyncThrowingStream<String, Error> {
        messages.append(.user(prompt, images: images, videos: videos))

        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()

        let task = Task {
            do {
                try await self.performStreaming(continuation: continuation)
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
    public func clear() {
        messages = messages.filter { $0.role == .system }
        cache = []
    }

    // MARK: - Private

    private func performStreaming(
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async throws {
        // TODO dkoski
        // TODO dkoski -- images and videos are not Sendable because they might be MLXArray
        // TODO dkoski -- also the messages passed should just be system prompt + user message.  kvcache handles the rest
        try await model.perform { container in
            print("START")
            let context = container.context
            let userInput = UserInput(
                chat: messages, processing: processing, additionalContext: additionalContext)
            let input = try await context.processor.prepare(input: userInput)

            if cache.isEmpty {
                cache = context.model.newCache(parameters: generateParameters)
            }

            // TODO dkoski get the task so we can wait
            var fullResponse = ""
            for await item in try MLXLMCommon.generate(
                input: input, cache: cache, parameters: generateParameters, context: context
            ) {
                if let chunk = item.chunk {
                    fullResponse += chunk
                    if case .terminated = continuation.yield(chunk) {
                        break
                    }
                }
            }

            Stream().synchronize()

            messages.append(.assistant(fullResponse))
            print("END")
            continuation.finish()
        }
    }

    public func synchronize() async {
        await model.synchronize()
    }
}
