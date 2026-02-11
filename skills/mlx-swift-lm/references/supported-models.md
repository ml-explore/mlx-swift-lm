# Supported Models

## Overview

mlx-swift-lm supports a wide range of LLM and VLM architectures through type registries that map `model_type` values from config.json to Swift implementations.

**Note:** Model support changes frequently. Check `LLMTypeRegistry` and `VLMTypeRegistry` in source for the latest list.

## Quick Reference

| Registry | Purpose | File |
|----------|---------|------|
| `LLMTypeRegistry` | LLM model type -> initializer | `MLXLLM/LLMModelFactory.swift` |
| `VLMTypeRegistry` | VLM model type -> initializer | `MLXVLM/VLMModelFactory.swift` |
| `LLMRegistry` | Pre-configured LLM models | `MLXLLM/LLMModelFactory.swift` |
| `VLMRegistry` | Pre-configured VLM models | `MLXVLM/VLMModelFactory.swift` |

## LLM Families

### Llama / Mistral

```swift
// model_type: "llama", "mistral"
LLMRegistry.llama3_2_3B_4bit      // mlx-community/Llama-3.2-3B-Instruct-4bit
LLMRegistry.llama3_2_1B_4bit      // mlx-community/Llama-3.2-1B-Instruct-4bit
LLMRegistry.llama3_8B_4bit        // mlx-community/Meta-Llama-3-8B-Instruct-4bit
LLMRegistry.llama3_1_8B_4bit      // mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
LLMRegistry.mistral7B4bit         // mlx-community/Mistral-7B-Instruct-v0.3-4bit
LLMRegistry.mistralNeMo4bit       // mlx-community/Mistral-Nemo-Instruct-2407-4bit
```

### Qwen

```swift
// model_type: "qwen2", "qwen3", "qwen3_moe"
LLMRegistry.qwen2_5_7b            // mlx-community/Qwen2.5-7B-Instruct-4bit
LLMRegistry.qwen2_5_1_5b          // mlx-community/Qwen2.5-1.5B-Instruct-4bit
LLMRegistry.qwen3_4b_4bit         // mlx-community/Qwen3-4B-4bit
LLMRegistry.qwen3_8b_4bit         // mlx-community/Qwen3-8B-4bit
LLMRegistry.qwen3MoE_30b_a3b_4bit // mlx-community/Qwen3-30B-A3B-4bit
```

### Gemma

```swift
// model_type: "gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n"
LLMRegistry.gemma2bQuantized      // mlx-community/quantized-gemma-2b-it
LLMRegistry.gemma_2_2b_it_4bit    // mlx-community/gemma-2-2b-it-4bit
LLMRegistry.gemma_2_9b_it_4bit    // mlx-community/gemma-2-9b-it-4bit
LLMRegistry.gemma3_1B_qat_4bit    // mlx-community/gemma-3-1b-it-qat-4bit
LLMRegistry.gemma3n_E4B_it_lm_4bit // mlx-community/gemma-3n-E4B-it-lm-4bit
```

### Phi

```swift
// model_type: "phi", "phi3", "phimoe"
LLMRegistry.phi4bit               // mlx-community/phi-2-hf-4bit-mlx
LLMRegistry.phi3_5_4bit           // mlx-community/Phi-3.5-mini-instruct-4bit
LLMRegistry.phi3_5MoE             // mlx-community/Phi-3.5-MoE-instruct-4bit
```

### DeepSeek

```swift
// model_type: "deepseek_v3"
LLMRegistry.deepSeekR1_7B_4bit    // mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit
LLMRegistry.deepseek_r1_4bit      // mlx-community/DeepSeek-R1-4bit
```

### GLM

```swift
// model_type: "glm4", "glm4_moe", "glm4_moe_lite"
LLMRegistry.glm4_9b_4bit          // mlx-community/GLM-4-9B-0414-4bit
// Note: GLM4 uses special tool call format
```

### Other Supported LLM Types

| model_type | Description |
|------------|-------------|
| `starcoder2` | StarCoder 2 |
| `cohere` | Cohere models |
| `openelm` | OpenELM |
| `internlm2` | InternLM 2 |
| `granite` | IBM Granite |
| `granitemoehybrid` | Granite MoE |
| `mimo`, `mimo_v2_flash` | MiMo |
| `minimax` | MiniMax |
| `bitnet` | BitNet |
| `smollm3` | SmolLM 3 |
| `ernie4_5` | ERNIE 4.5 |
| `lfm2`, `lfm2_moe` | LFM 2 |
| `exaone4` | EXAONE 4 |
| `olmo2`, `olmo3`, `olmoe` | OLMo |
| `falcon_h1` | Falcon H1 |
| `jamba_3b` | Jamba |
| `apertus` | Apertus |
| `nanochat` | NanoChat |
| `nemotron_h` | Nemotron H |
| `afmoe` | AfMoE |
| `bailing_moe` | Bailing MoE |
| `gpt_oss` | GPT OSS |
| `minicpm` | MiniCPM |

## VLM Families

### Qwen VL

```swift
// model_type: "qwen2_vl", "qwen2_5_vl", "qwen3_vl"
VLMRegistry.qwen2VL2BInstruct4Bit   // mlx-community/Qwen2-VL-2B-Instruct-4bit
VLMRegistry.qwen2_5VL3BInstruct4Bit // mlx-community/Qwen2.5-VL-3B-Instruct-4bit
VLMRegistry.qwen3VL4BInstruct4Bit   // lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit
```

### Gemma Vision

```swift
// model_type: "gemma3" (VLM context)
VLMRegistry.gemma3_4B_qat_4bit    // mlx-community/gemma-3-4b-it-qat-4bit
VLMRegistry.gemma3_12B_qat_4bit   // mlx-community/gemma-3-12b-it-qat-4bit
VLMRegistry.gemma3_27B_qat_4bit   // mlx-community/gemma-3-27b-it-qat-4bit
```

### PaliGemma

```swift
// model_type: "paligemma"
VLMRegistry.paligemma3bMix448_8bit // mlx-community/paligemma-3b-mix-448-8bit
```

### Other VLM Types

| model_type | Description |
|------------|-------------|
| `idefics3` | IDEFICS 3 |
| `smolvlm` | SmolVLM |
| `fastvlm`, `llava_qwen2` | FastVLM |
| `pixtral` | Pixtral |
| `mistral3` | Mistral 3 VLM |
| `lfm2_vl`, `lfm2-vl` | LFM2 VL |

## Loading Any Model

Models not in registries can be loaded by ID:

```swift
// Any mlx-community model
let config = ModelConfiguration(id: "mlx-community/SomeModel-4bit")
let container = try await LLMModelFactory.shared.loadContainer(
    configuration: config
)

// Specific revision
let config = ModelConfiguration(
    id: "mlx-community/Model",
    revision: "v1.0"
)

// Local model
let config = ModelConfiguration(
    directory: URL(filePath: "/path/to/model")
)
```

## Model-Specific Configurations

### Extra EOS Tokens

Some models need additional stop tokens:

```swift
// Gemma 3
let config = ModelConfiguration(
    id: "...",
    extraEOSTokens: ["<end_of_turn>"]
)

// Phi 3.5
let config = ModelConfiguration(
    id: "...",
    extraEOSTokens: ["<|end|>"]
)

// Llama 3 (handled automatically via generation_config.json)
```

### Tool Call Formats

```swift
// GLM4 requires special format
let config = ModelConfiguration(
    id: "mlx-community/GLM-4-9B-0414-4bit",
    toolCallFormat: .glm4
)

// LFM2
let config = ModelConfiguration(
    id: "mlx-community/LFM2-1.2B-4bit",
    toolCallFormat: .lfm2
)

// Most models use default .json format (auto-detected)
```

### Tokenizer Overrides

```swift
// Override tokenizer class
let config = ModelConfiguration(
    id: "...",
    overrideTokenizer: "PreTrainedTokenizer"
)

// Use tokenizer from different model
let config = ModelConfiguration(
    id: "model-without-tokenizer",
    tokenizerId: "different-model-with-tokenizer"
)
```

## Adding New Model Types

Extend the type registry for new architectures:

```swift
// Custom model type
LLMTypeRegistry.shared.register(
    modelType: "custom_model",
    creator: { configData in
        let config = try JSONDecoder().decode(CustomConfig.self, from: configData)
        return CustomModel(config)
    }
)
```

## Checking Supported Types

```swift
// List registered types (via source inspection)
// LLMTypeRegistry.shared supports:
// "llama", "mistral", "qwen2", "qwen3", "gemma", "phi", ...

// VLMTypeRegistry.shared supports:
// "qwen2_vl", "paligemma", "gemma3", "smolvlm", ...
```

## Memory Requirements

Approximate VRAM/RAM usage for quantized models:

| Model Size | 4-bit | 8-bit |
|------------|-------|-------|
| 1B params | ~0.5GB | ~1GB |
| 3B params | ~1.5GB | ~3GB |
| 7B params | ~3.5GB | ~7GB |
| 13B params | ~6.5GB | ~13GB |
| 70B params | ~35GB | ~70GB |

## Notes

- Model support depends on config.json `model_type` matching a registry entry
- Check source files for latest supported types
- MLX community models are pre-quantized and optimized
- Local models must have proper config.json and weights
