# LoRA Adapters

## Overview

mlx-swift-lm supports LoRA (Low-Rank Adaptation) and DoRA (Weight-Decomposed Low-Rank Adaptation) for parameter-efficient fine-tuning. Adapters can be loaded, applied, fused, or removed from models at runtime.

**Files:**
- `Libraries/MLXLMCommon/Adapters/LoRA/LoRAContainer.swift`
- `Libraries/MLXLMCommon/Adapters/LoRA/LoRAModel.swift`
- `Libraries/MLXLMCommon/Adapters/LoRA/LoRA+Layers.swift`
- `Libraries/MLXLMCommon/Adapters/LoRA/DoRA+Layers.swift`

## Quick Reference

| Type | Purpose |
|------|---------|
| `LoRAConfiguration` | Configuration for LoRA/DoRA adapters |
| `LoRAContainer` | Load, apply, fuse adapters |
| `LoRAModel` | Protocol for LoRA-compatible models |
| `LoRALayer` | Protocol for adapted layers |
| `LoRALinear` | LoRA adapter for Linear layers |
| `QLoRALinear` | LoRA adapter for QuantizedLinear |
| `DoRALinear` | DoRA adapter for Linear layers |
| `QDoRALinear` | DoRA adapter for QuantizedLinear |

## LoRAConfiguration

```swift
public struct LoRAConfiguration: Sendable, Codable {
    public let numLayers: Int           // Number of layers to adapt
    public let fineTuneType: FineTuneType  // .lora or .dora
    public let loraParameters: LoRAParameters

    public struct LoRAParameters: Sendable, Codable {
        public let rank: Int         // Low-rank dimension (default: 8)
        public let scale: Float      // Scaling factor (default: 10.0)
        public let keys: [String]?   // Layer keys to adapt (nil = all Linear)
    }

    public enum FineTuneType: String, Codable {
        case lora
        case dora
    }
}
```

### Creating Configuration

```swift
// Basic LoRA
let config = LoRAConfiguration(
    numLayers: 16,
    fineTuneType: .lora,
    loraParameters: .init(rank: 8, scale: 10.0)
)

// DoRA with custom keys
let config = LoRAConfiguration(
    numLayers: 28,
    fineTuneType: .dora,
    loraParameters: .init(
        rank: 16,
        scale: 20.0,
        keys: ["self_attn.q_proj", "self_attn.v_proj"]
    )
)
```

### adapter_config.json Format

```json
{
  "fine_tune_type": "lora",
  "num_layers": 28,
  "lora_parameters": {
    "rank": 16,
    "scale": 20.0,
    "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
  }
}
```

## Loading Adapters

### From Directory

```swift
// Directory contains:
// - adapter_config.json
// - adapters.safetensors

let adapter = try LoRAContainer.from(directory: adapterURL)

// Apply to model
try adapter.load(into: model)
```

### From Model (Create New Adapter)

```swift
// Create adapter configuration from a model
let adapter = try LoRAContainer.from(
    model: model,
    configuration: LoRAConfiguration(
        numLayers: 16,
        loraParameters: .init(rank: 8)
    )
)

// Model is now ready for training
// adapter.parameters contains trainable weights
```

## Applying Adapters

### Load Adapter Weights

```swift
let adapter = try LoRAContainer.from(directory: adapterDir)

// Load into model (replaces Linear with LoRALinear)
try adapter.load(into: model)

// Model now uses adapter for inference
```

### Fuse Adapter (Permanent)

```swift
// Permanently merge adapter weights into model
try adapter.fuse(with: model)

// Model is back to standard Linear layers
// with adapter weights baked in
```

### Unload Adapter

```swift
// Remove adapter, restore original layers
adapter.unload(from: model)
```

## LoRA Layer Types

### LoRALinear

For regular `Linear` layers:

```swift
public class LoRALinear: Linear, LoRALayer {
    let scale: Float
    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    // Forward: y = Wx + scale * (x @ A @ B)
}
```

### QLoRALinear

For `QuantizedLinear` layers (quantized base model):

```swift
public class QLoRALinear: QuantizedLinear, LoRALayer {
    // Same interface as LoRALinear
    // Works with quantized weights
}
```

### DoRALinear

DoRA adds magnitude tracking:

```swift
public class DoRALinear: Linear, LoRALayer {
    let scale: Float
    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray
    @ParameterInfo(key: "m") var magnitude: MLXArray

    // Forward includes weight normalization
}
```

## LoRAModel Protocol

Models must conform to `LoRAModel` for adapter support:

```swift
public protocol LoRAModel {
    /// Layers to apply adapters to
    var loraLayers: [Module] { get }

    /// Default keys (nil = all Linear layers)
    var loraDefaultKeys: [String] { get }
}
```

### Default Implementation

```swift
extension LoRAModel {
    // By default, adapt all Linear layers in loraLayers
    public var loraDefaultKeys: [String] {
        loraLayers.flatMap { $0.namedModules() }
            .compactMap { key, module in
                module is Linear ? key : nil
            }
    }
}
```

## LoRALayer Protocol

```swift
public protocol LoRALayer: Module {
    /// Merge adapter weights into base layer
    func fused() -> Module

    /// Remove adapter, return original layer
    func reverted() -> Module
}
```

## Usage Examples

### Inference with Adapter

```swift
// Load base model
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(id: "mlx-community/Llama-3.2-3B-Instruct-4bit")
)

// Load and apply adapter
// Note: adapter.load() throws, but container.update() closure is non-throwing
let adapter = try LoRAContainer.from(directory: adapterDir)
await container.update { context in
    do {
        try adapter.load(into: context.model)
    } catch {
        // Handle adapter loading error
        print("Failed to load adapter: \(error)")
    }
}

// Generate with adapted model
let session = ChatSession(container)
let response = try await session.respond(to: "Hello")
```

### Fuse for Deployment

```swift
// Fuse adapter for faster inference
// Note: adapter.fuse() throws, handle errors in closure
await container.update { context in
    do {
        try adapter.fuse(with: context.model)
    } catch {
        print("Failed to fuse adapter: \(error)")
    }
}

// Save fused model weights if desired
// Model no longer has LoRA overhead
```

### Hot-swap Adapters

```swift
// Remove current adapter (unload is non-throwing)
await container.update { context in
    adapter1.unload(from: context.model)
}

// Apply different adapter
let adapter2 = try LoRAContainer.from(directory: adapter2Dir)
await container.update { context in
    do {
        try adapter2.load(into: context.model)
    } catch {
        print("Failed to load adapter: \(error)")
    }
}
```

## Memory Considerations

| Approach | Memory Impact |
|----------|---------------|
| LoRALinear | +2 small matrices per layer |
| QLoRALinear | Same, works with quantized base |
| Fused | No overhead (merged into weights) |

### Typical LoRA Memory

For rank=8 on a 7B model:
- ~10-20MB additional memory
- Much smaller than full fine-tuning

## Saving Adapter Weights

After training, save adapter weights:

```swift
// Get trainable parameters
let params = model.trainableParameters()

// Save to safetensors
try save(arrays: params.flattened(), url: weightsURL)

// Save configuration
let encoder = JSONEncoder()
encoder.outputFormatting = .prettyPrinted
let configData = try encoder.encode(config)
try configData.write(to: configURL)
```

## Deprecated Patterns

No major deprecations in LoRA system. However, note:

### Old LoRATrain.convert() pattern

The newer `LoRAContainer.from(model:configuration:)` is preferred over the older training flow that directly converted layers. The container approach:
- Provides cleaner state management
- Handles loading/unloading consistently
- Works with both LoRA and DoRA
