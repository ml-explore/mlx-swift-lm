#  Evaluation

The simplified LLM/VLM API allows you to load a model and evaluate prompts with only a few lines of code.

For example, this loads a model and asks a question and a follow-on question:

```swift
let model = try await loadModel(
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

The second question actually refers to information (the location) from the first
question -- this context is maintained inside the `ChatSession` object.

If you need a one-shot prompt/response simply create a `ChatSession`, evaluate
the prompt and discard.  Multiple `ChatSession` instances could also be used
(at the cost of the memory in the `KVCache`) to handle multiple streams of
context.

## Streaming Output

The previous example produced the entire response in one call.  Often
users want to see the text as it is generated -- you can do this with
a stream:

```swift
let model = try await loadModel(
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)

for try await item in session.streamResponse(to: "Why is the sky blue?") {
    print(item, terminator: "")
}
print()
```

## Structured Chat Continuation

`ChatSession` can also continue from structured `Chat.Message` values. This
is useful for agent loops that consume tool calls from `streamDetails(to:role:images:videos:)`
and then append one or more `.tool` messages without rebuilding the whole
conversation history:

```swift
var toolResults: [Chat.Message] = []

for try await item in session.streamDetails(
    to: "What is the weather in Paris?",
    images: [],
    videos: []
) {
    if case .toolCall(let toolCall) = item {
        let toolResult = try await callTool(toolCall)
        toolResults.append(.tool(toolResult))
    }
}

if !toolResults.isEmpty {
    let answer = try await session.respond(to: toolResults)
    print(answer)
}
```

When a session is initialized with history, the first generation must tokenize
and prefill that history to create a KV cache. Reuse the same `ChatSession` and
continue with structured messages to avoid paying that full prefill cost on
each tool turn.

## VLMs (Vision Language Models)

This same API supports VLMs as well.  Simply present the image or video
to the `ChatSession`:

```swift
let model = try await loadModel(
    using: TokenizersLoader(),
    id: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
)
let session = ChatSession(model)

let answer1 = try await session.respond(
    to: "what kind of creature is in the picture?",
    image: .url(URL(fileURLWithPath: "support/test.jpg"))
)
print(answer1)

// we can ask a followup question referring back to the previous image
let answer2 = try await session.respond(
    to: "What is behind the dog?"
)
print(answer2)
```

## Advanced Usage

The `ChatSession` has a number of parameters you can supply when creating it:

- **instructions**: optional instructions to the chat session, e.g. describing what type of responses to give
    - for example you might instruct the language model to respond in rhyme or
        talking like a famous character from a movie
    - or that the responses should be very brief
- **generateParameters**: parameters that control the generation of output, e.g. token limits and temperature
    - see `GenerateParameters`
- **processing**: optional media processing instructions
