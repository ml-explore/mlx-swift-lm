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
| `.xmlFunction` | Nemotron, Qwen3 Coder, Qwen3 Next | `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>` |
| `.qwen35` | Qwen 3.5 | Same `<tool_call>` frame as `.xmlFunction`, but also accepts a framed Qwen/Hermes JSON payload (`<tool_call>{"name":"f","arguments":{...}}</tool_call>`) that Qwen 3.5 sporadically emits instead of XML. Bare (unframed) JSON is not recovered. |
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

### Schema Validation

When you supply tool schemas, the processor does two checks before a call
becomes executable:

1. The function name must be in the declared tools.
2. The arguments must not definitively violate the tool's `parameters` schema.

A call that fails a check is not dispatched. It is reported as a
`RejectedToolCall` with the reason `undeclaredTool` or `invalidArguments`.
The `detail` of a schema violation says where the problem is and what the
schema requires, for example `arguments.limit must be an integer`.

The package validates the JSON Schema keywords that tool declarations commonly use:
`type`, `properties`, `required`, `additionalProperties`, `items`,
`minItems`, `maxItems`, `uniqueItems`, `minLength`, `maxLength`, `minimum`,
`maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `enum`, `const`, `anyOf`,
`oneOf`, and `allOf`. Validation is conservative: an unsupported assertion such
as `pattern` or `$ref` makes the result uncertain and cannot reject a call on
its own. A tool without a `parameters` schema accepts all arguments. Without
tool declarations, all parsed calls pass.

## Format Resolution

When a model is loaded, the factories resolve the format in this order:

1. an explicit value on the `ModelConfiguration` (registry entry or caller),
2. a `ChatConventionsResolving` registered with `ChatConventionsRegistry`
   (keyed on repo id / model type, for conventions the model cannot know),
3. an explicit `tool_parser_type` in the checkpoint's tokenizer files,
4. the selected tool template reconciled with the model's own
   `ChatConventionsProviding` declaration.

Nothing resolved means the default `.json` parser is used.

At step 4, the model declaration remains selected when it can parse the template's
dialect. Otherwise, the template refines it because the template is what teaches
the model what to emit. This lets Qwen 3.5 keep its dual-dialect parser while a
Hermes checkpoint can refine the Llama 3 architecture heuristic from inline JSON
to framed JSON. If the template has no recognizable tool syntax, the model
declaration remains the fallback.

```swift
ToolCallFormat.inferred(fromChatTemplate: template)  // -> .mistral, .glm4, ...
ToolCallFormat(toolParserType: "qwen3_coder")        // -> .xmlFunction
```

Protocols with token-level framing (`.gptOSS`, `.atem`) are deliberately not
inferred from templates — they are selected by the models that own them.

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
