# LoRA Training

## Overview

mlx-swift-lm supports fine-tuning language models using LoRA (Low-Rank Adaptation) through the `LoRATrain` API. Training adapts only a small number of parameters while keeping the base model frozen.

**File:** `Libraries/MLXLLM/LoraTrain.swift`

## Quick Reference

| Type | Purpose |
|------|---------|
| `LoRATrain` | Namespace for training functions |
| `LoRATrain.Parameters` | Training hyperparameters |
| `LoRATrain.Progress` | Progress reporting enum |
| `LoRATrain.ProgressDisposition` | Continue/stop training |
| `LoRABatchIterator` | Internal batch iteration |

## Training Parameters

```swift
public struct Parameters: Sendable {
    public var batchSize: Int = 4
    public var iterations: Int = 1000
    public var stepsPerReport: Int = 10      // Loss reporting frequency
    public var stepsPerEval: Int = 100       // Validation frequency
    public var validationBatches: Int = 10   // 0 = full validation set
    public var saveEvery: Int = 100          // Checkpoint frequency
    public var adapterURL: URL?              // Save location

    public init(
        batchSize: Int = 4,
        iterations: Int = 1000,
        stepsPerReport: Int = 10,
        stepsPerEval: Int = 100,
        validationBatches: Int = 10,
        saveEvery: Int = 100,
        adapterURL: URL? = nil
    )
}
```

## Training Workflow

### 1. Load Model

```swift
import MLXLLM
import MLXLMCommon

// Load base model
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Llama-3.2-3B-Instruct-4bit")
)

// Extract model and tokenizer for training
let model = await container.perform { $0.model }
let tokenizer = await container.tokenizer
```

### 2. Apply LoRA Layers

```swift
// Create LoRA configuration
let loraConfig = LoRAConfiguration(
    numLayers: 16,
    fineTuneType: .lora,
    loraParameters: .init(rank: 8, scale: 10.0)
)

// Apply adapters (freezes base model)
let adapter = try LoRAContainer.from(
    model: model,
    configuration: loraConfig
)
```

### 3. Load Training Data

```swift
// Data format: Array of strings (each is a training example)
let trainData: [String] = [
    "Example prompt and completion 1",
    "Example prompt and completion 2",
    // ...
]

let validData: [String] = [
    "Validation example 1",
    // ...
]
```

### 4. Configure Optimizer

```swift
import MLXOptimizers

let optimizer = Adam(learningRate: 1e-5)

// Or with learning rate schedule
let schedule = LinearSchedule(
    init: 1e-5,
    end: 1e-6,
    steps: 1000
)
let optimizer = Adam(learningRate: schedule)
```

### 5. Train

```swift
let parameters = LoRATrain.Parameters(
    batchSize: 4,
    iterations: 1000,
    stepsPerReport: 10,
    stepsPerEval: 100,
    adapterURL: saveURL
)

try LoRATrain.train(
    model: model,
    train: trainData,
    validate: validData,
    optimizer: optimizer,
    tokenizer: tokenizer,
    parameters: parameters
) { progress in
    switch progress {
    case .train(let iter, let loss, let iterSec, let tokSec):
        print("Iter \(iter): loss=\(loss), \(tokSec) tok/s")

    case .validation(let iter, let valLoss, let time):
        print("Iter \(iter): val_loss=\(valLoss)")

    case .save(let iter, let url):
        print("Saved checkpoint at iteration \(iter)")
    }

    // Return .stop to halt training early
    return .more
}
```

## Progress Reporting

```swift
public enum Progress: CustomStringConvertible {
    case train(
        iteration: Int,
        trainingLoss: Float,
        iterationsPerSecond: Double,
        tokensPerSecond: Double
    )

    case validation(
        iteration: Int,
        validationLoss: Float,
        validationTime: Double
    )

    case save(
        iteration: Int,
        url: URL
    )
}

public enum ProgressDisposition {
    case stop   // Halt training
    case more   // Continue training
}
```

## Data Formats

### Plain Text (.txt)

One example per line:

```
First training example here.
Second training example here.
Third training example here.
```

### JSONL (.jsonl)

JSON Lines format with "text" field:

```json
{"text": "First training example here."}
{"text": "Second training example here."}
{"text": "Third training example here."}
```

### Loading Data

```swift
func loadLoRAData(directory: URL, name: String) throws -> [String] {
    let txtURL = directory.appending(component: "\(name).txt")
    let jsonlURL = directory.appending(component: "\(name).jsonl")

    if FileManager.default.fileExists(atPath: txtURL.path) {
        let content = try String(contentsOf: txtURL)
        return content.components(separatedBy: .newlines)
            .filter { !$0.isEmpty }
    } else if FileManager.default.fileExists(atPath: jsonlURL.path) {
        let content = try String(contentsOf: jsonlURL)
        return content.components(separatedBy: .newlines)
            .compactMap { line -> String? in
                guard !line.isEmpty,
                      let data = line.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let text = json["text"] as? String
                else { return nil }
                return text
            }
    }
    throw NSError(domain: "LoRA", code: 1, userInfo: [
        NSLocalizedDescriptionKey: "No training data found"
    ])
}
```

## Loss Function

Default cross-entropy loss with length masking:

```swift
public static func loss(
    model: Module,
    inputs: MLXArray,
    targets: MLXArray,
    lengths: MLXArray
) -> (MLXArray, MLXArray) {
    let logits = model(inputs, cache: nil)
    let lengthMask = MLXArray(0..<inputs.dim(1))[.newAxis, 0...] .< lengths[0..., .newAxis]
    let ntoks = lengthMask.sum()
    let ce = (crossEntropy(logits: logits, targets: targets) * lengthMask).sum() / ntoks
    return (ce, ntoks)
}
```

### Custom Loss

```swift
func customLoss(
    model: Module,
    inputs: MLXArray,
    targets: MLXArray,
    lengths: MLXArray
) -> (MLXArray, MLXArray) {
    // Custom loss implementation
    // Return (loss, tokenCount)
}

try LoRATrain.train(
    model: model,
    train: trainData,
    validate: validData,
    optimizer: optimizer,
    loss: customLoss,  // Custom loss function
    tokenizer: tokenizer,
    parameters: parameters
) { progress in
    return .more
}
```

## Evaluation

Evaluate model on test data:

```swift
let testLoss = LoRATrain.evaluate(
    model: model,
    dataset: testData,
    tokenizer: tokenizer,
    batchSize: 4,
    batchCount: 0  // 0 = full dataset
)
print("Test loss: \(testLoss)")
```

## Saving Weights

### Automatic Checkpointing

Set `adapterURL` in parameters for automatic saves:

```swift
let parameters = LoRATrain.Parameters(
    saveEvery: 100,
    adapterURL: URL(filePath: "adapter.safetensors")
)
```

### Manual Saving

```swift
try LoRATrain.saveLoRAWeights(model: model, url: weightsURL)
```

### Loading Saved Weights

```swift
// Via LoRAContainer
let adapter = try LoRAContainer.from(directory: adapterDir)
try adapter.load(into: model)
```

## Memory Optimization

### Sequence Length

Long sequences increase memory usage:

```swift
// Warning printed if sequences > 2048 tokens
// Consider pre-splitting data for long documents
```

### Batch Size

Reduce batch size if running out of memory:

```swift
let parameters = LoRATrain.Parameters(
    batchSize: 1  // Minimum batch size
)
```

### Gradient Checkpointing

Not currently implemented in Swift - use smaller models or reduce batch size.

## Complete Example

```swift
import Foundation
import MLXLLM
import MLXLMCommon
import MLXOptimizers

func trainAdapter() async throws {
    // Load model
    let container = try await LLMModelFactory.shared.loadContainer(
        configuration: .init(id: "mlx-community/Llama-3.2-1B-Instruct-4bit")
    )

    let model = await container.perform { SendableBox($0.model) }.consume()
    let tokenizer = await container.tokenizer

    // Apply LoRA
    let adapter = try LoRAContainer.from(
        model: model as! LanguageModel,
        configuration: LoRAConfiguration(
            numLayers: 8,
            loraParameters: .init(rank: 8)
        )
    )

    // Load data
    let trainData = try loadData(name: "train")
    let validData = try loadData(name: "valid")

    // Train
    let optimizer = Adam(learningRate: 1e-5)
    let params = LoRATrain.Parameters(
        batchSize: 2,
        iterations: 500,
        adapterURL: URL(filePath: "adapter.safetensors")
    )

    try LoRATrain.train(
        model: model,
        train: trainData,
        validate: validData,
        optimizer: optimizer,
        tokenizer: tokenizer,
        parameters: params
    ) { progress in
        print(progress)
        return .more
    }

    print("Training complete!")
}
```

## Deprecated Patterns

No major deprecations in training API. The `LoRATrain` namespace provides a stable interface for training workflows.
