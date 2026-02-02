# Tokenizer & Chat Messages

## Overview

Tokenizers convert text to/from token IDs and apply chat templates for multi-turn conversations. The `Chat.Message` type provides a structured way to represent conversation history.

**Files:**
- `Libraries/MLXLMCommon/Tokenizer.swift`
- `Libraries/MLXLMCommon/Chat.swift`

## Quick Reference

| Type | Purpose |
|------|---------|
| `Tokenizer` | Protocol from swift-transformers |
| `PreTrainedTokenizer` | Standard tokenizer implementation |
| `Chat.Message` | Structured chat message |
| `Chat.Message.Role` | user, assistant, system, tool |
| `MessageGenerator` | Convert Chat.Message to raw dicts |
| `StreamingDetokenizer` | Incremental token-to-text |
| `NaiveStreamingDetokenizer` | Default streaming implementation |

## Tokenizer Loading

### Automatic Loading

Tokenizers are loaded automatically by model factories:

```swift
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: config
)
let tokenizer = await container.tokenizer
```

### Manual Loading

```swift
let tokenizer = try await loadTokenizer(
    configuration: config,
    hub: HubApi()
)
```

### Loading Components

```swift
// Load tokenizer config and data separately
let (tokenizerConfig, tokenizerData) = try await loadTokenizerConfig(
    configuration: config,
    hub: hub
)

let tokenizer = try PreTrainedTokenizer(
    tokenizerConfig: tokenizerConfig,
    tokenizerData: tokenizerData
)
```

## Tokenizer Usage

### Basic Encoding/Decoding

```swift
// Encode text to tokens
let tokens: [Int] = tokenizer.encode(text: "Hello, world!")

// Decode tokens to text
let text: String = tokenizer.decode(tokens: tokens)
```

### Chat Template

Apply model-specific chat formatting:

```swift
let messages: [[String: String]] = [
    ["role": "system", "content": "You are helpful"],
    ["role": "user", "content": "Hello"]
]

let tokens = try tokenizer.applyChatTemplate(messages: messages)
```

### With Tools

```swift
let tokens = try tokenizer.applyChatTemplate(
    messages: messages,
    tools: [weatherTool.schema],
    additionalContext: ["special_key": "value"]
)
```

### Special Tokens

```swift
// EOS token
tokenizer.eosToken        // String, e.g., "</s>"
tokenizer.eosTokenId      // Int?, e.g., 2

// Unknown token
tokenizer.unknownToken    // String
tokenizer.unknownTokenId  // Int?

// Convert between tokens and IDs
let id = tokenizer.convertTokenToId("</s>")  // Int?
let token = tokenizer.convertIdToToken(2)    // String?
```

## EOS Token Handling

EOS tokens come from multiple sources, merged at load time:

1. **generation_config.json** - `eos_token_id` (primary, overrides others)
2. **config.json** - `eos_token_id`
3. **ModelConfiguration** - `extraEOSTokens` (additional tokens)
4. **Tokenizer** - `eosTokenId`

```swift
// Extra EOS tokens in configuration (as strings)
let config = ModelConfiguration(
    id: "model-id",
    extraEOSTokens: ["<|end_of_turn|>", "<|eot_id|>"]
)

// EOS token IDs loaded from JSON (as Int set)
config.eosTokenIds  // Set<Int>, loaded from generation_config.json
```

### Checking for EOS

```swift
// In generation loop
var eosTokenIds = context.configuration.eosTokenIds
if let tokenizerEos = tokenizer.eosTokenId {
    eosTokenIds.insert(tokenizerEos)
}
for token in context.configuration.extraEOSTokens {
    if let id = tokenizer.convertTokenToId(token) {
        eosTokenIds.insert(id)
    }
}

// Check if token is EOS
if eosTokenIds.contains(token) {
    // Stop generation
}
```

## Chat.Message

Structured representation of chat messages:

```swift
public struct Message {
    public var role: Role
    public var content: String
    public var images: [UserInput.Image]
    public var videos: [UserInput.Video]

    public enum Role: String {
        case user
        case assistant
        case system
        case tool
    }
}
```

### Creating Messages

```swift
// Static constructors
let system = Chat.Message.system("You are helpful")
let user = Chat.Message.user("Hello")
let assistant = Chat.Message.assistant("Hi there!")
let tool = Chat.Message.tool("""{"result": 42}""")

// With media (VLM)
let userWithImage = Chat.Message.user(
    "What's in this image?",
    images: [.url(imageURL)]
)

// Direct initialization
let message = Chat.Message(
    role: .user,
    content: "Hello",
    images: [],
    videos: []
)
```

### Using with ChatSession

```swift
// ChatSession handles messages internally
let session = ChatSession(container, instructions: "You are helpful")
let response = try await session.respond(to: "Hello")

// Or restore from history
let history: [Chat.Message] = [
    .system("You are helpful"),
    .user("Previous question"),
    .assistant("Previous answer")
]
let session = ChatSession(container, history: history)
```

## MessageGenerator

Converts `Chat.Message` to raw dictionaries for tokenizer:

```swift
public protocol MessageGenerator {
    func generate(from input: UserInput) -> [Message]
    func generate(messages: [Chat.Message]) -> [Message]
    func generate(message: Chat.Message) -> Message
}

// Message is [String: any Sendable]
```

### Default Generator

```swift
let generator = DefaultMessageGenerator()
let raw = generator.generate(message: .user("Hello"))
// ["role": "user", "content": "Hello"]
```

### No System Generator

For models that don't support system messages:

```swift
let generator = NoSystemMessageGenerator()
let messages = generator.generate(messages: [
    .system("Ignored"),
    .user("Hello")
])
// Only returns user message
```

### Model-Specific Generators

VLMs often have custom generators for image formatting:

```swift
// Qwen2VL uses specific format for images
let generator = Qwen2VLMessageGenerator()
```

## StreamingDetokenizer

Convert tokens to text incrementally during generation:

```swift
public protocol StreamingDetokenizer: IteratorProtocol<String> {
    mutating func append(token: Int)
}
```

### Usage

```swift
var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)

for token in tokenIterator {
    detokenizer.append(token: token)
    if let chunk = detokenizer.next() {
        print(chunk, terminator: "")
    }
}
```

### Handling Incomplete Unicode

The detokenizer handles tokens that produce incomplete unicode:

```swift
// Returns nil if token produces incomplete character (U+FFFD)
if let chunk = detokenizer.next() {
    // Complete character(s) ready
}
// nil means waiting for more tokens
```

## Tokenizer Replacement Registry

Override tokenizer classes for compatibility:

```swift
// Built-in replacements
// "Qwen2Tokenizer" -> "PreTrainedTokenizer"
// "InternLM2Tokenizer" -> "PreTrainedTokenizer"
// etc.

// Add custom replacement
replacementTokenizers["CustomTokenizer"] = "PreTrainedTokenizer"
```

## Deprecated Patterns

### TokenIterator with prompt MLXArray

```swift
// DEPRECATED: TokenIterator with raw prompt tokens
let iterator = try TokenIterator(
    prompt: MLXArray(tokens),  // Old signature
    model: model,
    parameters: params
)

// USE INSTEAD: TokenIterator with LMInput
let input = LMInput(tokens: MLXArray(tokens))
let iterator = try TokenIterator(
    input: input,
    model: model,
    parameters: params
)

// Or better, use prepared input
let input = try await processor.prepare(input: userInput)
let iterator = try TokenIterator(
    input: input,
    model: model,
    parameters: params
)
```

The old signature is marked deprecated and will be removed. The new signature with `LMInput` properly handles multi-modal inputs and is the standard pattern.
