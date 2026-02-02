# Tool Calling

## Overview

mlx-swift-lm supports function calling / tool use with multiple model-specific formats. Models can generate structured tool calls that applications parse and execute, returning results back to the model.

**Files:**
- `Libraries/MLXLMCommon/Tool/Tool.swift`
- `Libraries/MLXLMCommon/Tool/ToolCall.swift`
- `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift`
- `Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift`
- `Libraries/MLXLMCommon/Tool/ToolParameter.swift`

## Quick Reference

| Type | Purpose |
|------|---------|
| `Tool<Input, Output>` | Define a callable tool with typed `handler` |
| `ToolCall` | Parsed tool call from model output; has `execute(with:)` method |
| `ToolCallFormat` | Enum of supported formats |
| `ToolCallProcessor` | Streaming tool call detection |
| `ToolParameter` | Parameter definition for schema |
| `ToolSpec` | JSON schema dictionary type |

> **Note:** The `execute(with:)` method belongs to `ToolCall`, not `Tool`. You pass the `Tool` instance to `toolCall.execute(with: tool)` for type-safe execution.

## Supported Formats

| Format | Models | Example Output |
|--------|--------|----------------|
| `.json` | Llama, Qwen, most models | `<tool_call>{"name":"f","arguments":{...}}</tool_call>` |
| `.lfm2` | LFM2 | `<\|tool_call_start\|>{"name":"f",...}<\|tool_call_end\|>` |
| `.xmlFunction` | Qwen3 Coder | `<function=name><parameter=k>v</parameter></function>` |
| `.glm4` | GLM4 | `func<arg_key>k</arg_key><arg_value>v</arg_value>` |
| `.gemma` | Gemma | `call:name{key:value}` |
| `.kimiK2` | Kimi K2 | `functions.name:0<\|tool_call_argument_begin\|>{...}` |
| `.minimaxM2` | MiniMax M2 | `<invoke name="f"><parameter name="k">v</parameter></invoke>` |

## Defining Tools

### Basic Tool Definition

```swift
struct WeatherInput: Codable {
    let location: String
    let unit: String?
}

struct WeatherOutput: Codable {
    let temperature: Double
    let condition: String
}

let weatherTool = Tool<WeatherInput, WeatherOutput>(
    name: "get_weather",
    description: "Get current weather for a location",
    parameters: [
        .required("location", type: .string, description: "City name"),
        .optional("unit", type: .string, description: "celsius or fahrenheit")
    ]
) { input in
    // Fetch weather...
    return WeatherOutput(temperature: 72, condition: "sunny")
}
```

### ToolParameter Types

```swift
// Required parameters
ToolParameter.required("count", type: .int, description: "Number of items")
ToolParameter.required("price", type: .double, description: "Price in USD")
ToolParameter.required("enabled", type: .bool, description: "Enable feature")
ToolParameter.required("name", type: .string, description: "User name")

// Optional parameters
ToolParameter.optional("tags", type: .array(elementType: .string), description: "List of tags")
ToolParameter.optional("config", type: .object(properties: [...]), description: "Config object")
```

### Custom Schema

```swift
let tool = Tool<Input, Output>(
    schema: [
        "type": "function",
        "function": [
            "name": "search",
            "description": "Search the database",
            "parameters": [
                "type": "object",
                "properties": [
                    "query": ["type": "string", "description": "Search query"]
                ],
                "required": ["query"]
            ]
        ]
    ],
    handler: { input in
        // ...
    }
)
```

## Passing Tools to Model

```swift
// Include tools in UserInput
let userInput = UserInput(
    prompt: .text("What's the weather in Paris?"),
    tools: [weatherTool.schema, searchTool.schema]
)

// Prepare and generate
let lmInput = try await modelContainer.prepare(input: userInput)
let stream = try await modelContainer.generate(input: lmInput, parameters: params)
```

## Processing Tool Calls

### From Generation Stream

```swift
for await generation in stream {
    switch generation {
    case .chunk(let text):
        print(text, terminator: "")

    case .toolCall(let toolCall):
        // Model wants to call a tool
        print("Tool call: \(toolCall.function.name)")
        print("Arguments: \(toolCall.function.arguments)")

        // Execute the tool
        let result = try await toolCall.execute(with: weatherTool)
        // Send result back to model...

    case .info(let info):
        print("\nDone: \(info.tokensPerSecond) tok/s")
    }
}
```

### Executing Tool Calls

```swift
// Type-safe execution
let toolCall: ToolCall = ...
let result = try await toolCall.execute(with: weatherTool)
// result is WeatherOutput

// Manual execution
let args = toolCall.function.arguments
let location = args["location"]  // JSONValue
```

## ToolCallProcessor

For streaming detection of tool calls:

```swift
let processor = ToolCallProcessor(format: .json)

// Process each chunk
for chunk in generatedChunks {
    if let text = processor.processChunk(chunk) {
        // Regular text output
        print(text, terminator: "")
    }
}

// After generation, check for tool calls
for toolCall in processor.toolCalls {
    print("Detected tool call: \(toolCall.function.name)")
}
```

### Processor with Tool Schemas

```swift
let processor = ToolCallProcessor(
    format: .lfm2,
    tools: [weatherTool.schema, searchTool.schema]  // For type-aware parsing
)
```

## Format Auto-Detection

Formats are auto-detected from model type:

```swift
// Auto-detected based on model_type in config.json
ToolCallFormat.infer(from: "lfm2")     // -> .lfm2
ToolCallFormat.infer(from: "glm4")     // -> .glm4
ToolCallFormat.infer(from: "gemma")    // -> .gemma
ToolCallFormat.infer(from: "llama")    // -> nil (use default .json)
```

### Explicit Format in Configuration

```swift
let config = ModelConfiguration(
    id: "mlx-community/GLM-4-9B-0414-4bit",
    toolCallFormat: .glm4
)
```

## ToolCall Structure

```swift
public struct ToolCall: Hashable, Codable, Sendable {
    public struct Function: Hashable, Codable, Sendable {
        public let name: String
        public let arguments: [String: JSONValue]
    }

    public let function: Function
}

// JSONValue handles various JSON types
public enum JSONValue: Hashable, Codable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])
}
```

## Multi-Turn with Tool Results

```swift
var messages: [Chat.Message] = [
    .user("What's the weather in Paris?")
]

// Use streamDetails to receive tool calls (respond/streamResponse drops them)
var responseText = ""
var detectedToolCall: ToolCall?

for try await generation in session.streamDetails(to: "What's the weather?", images: [], videos: []) {
    switch generation {
    case .chunk(let text):
        responseText += text
    case .toolCall(let toolCall):
        detectedToolCall = toolCall
    case .info:
        break
    }
}

// If tool call detected, execute and add result
if let toolCall = detectedToolCall {
    let result = try await toolCall.execute(with: weatherTool)

    // Add assistant's tool call and result
    messages.append(.assistant(responseText))
    messages.append(.tool("""
        {"temperature": \(result.temperature), "condition": "\(result.condition)"}
        """))

    // Continue conversation with updated history
    let session = ChatSession(modelContainer, history: messages)
    let finalResponse = try await session.respond(to: "")
}
```

## Parser Protocol

Custom formats can implement `ToolCallParser`:

```swift
public protocol ToolCallParser: Sendable {
    var startTag: String? { get }  // nil for inline formats
    var endTag: String? { get }
    func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall?
}
```

## Error Handling

```swift
public enum ToolError: Error {
    case nameMismatch(toolName: String, functionName: String)
}

do {
    let result = try await toolCall.execute(with: myTool)
} catch ToolError.nameMismatch(let expected, let got) {
    print("Tool '\(got)' doesn't match expected '\(expected)'")
}
```

## Best Practices

### DO

```swift
// DO: Check tool call name before executing
guard toolCall.function.name == "get_weather" else {
    // Handle unknown tool
}

// DO: Handle missing optional arguments with pattern matching
let unit: String
if case .string(let value) = toolCall.function.arguments["unit"] {
    unit = value
} else {
    unit = "celsius"  // default
}

// DO: Use specific format for models that need it
let config = ModelConfiguration(
    id: "glm-model",
    toolCallFormat: .glm4  // Required for GLM4
)
```

### DON'T

```swift
// DON'T: Assume format from model family
// Some models in a family may use different formats

// DON'T: Ignore tool call errors
// Always handle potential parsing/execution failures
```

## Deprecated Patterns

No major deprecations in tool calling - this is a newer feature. However, ensure you're using the streaming `Generation.toolCall` pattern rather than post-hoc parsing of raw output.
