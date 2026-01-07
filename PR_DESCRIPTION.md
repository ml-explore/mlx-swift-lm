# Fix: Support per-layer intermediate_size array for Gemma 3n

## Problem

Gemma 3n models from HuggingFace (e.g., `mlx-community/gemma-3n-E2B-it-4bit`) fail to load with the following error:

```
configurationDecodingError("config.json", ...,
  Swift.DecodingError.typeMismatch(Swift.Int,
    Swift.DecodingError.Context(
      codingPath: [VLMCodingKeys(stringValue: "text_config", intValue: nil),
                   CodingKeys(stringValue: "intermediate_size", intValue: nil)],
      debugDescription: "Expected to decode Int but found an array instead."
    )
  )
)
```

## Root Cause

Gemma 3n models specify `intermediate_size` as a **per-layer array** in their `config.json`:

```json
"intermediate_size": [8192, 8192, 8192, ...],  // 30 values for 30 layers
```

However, `Gemma3nTextConfiguration` expects a single `Int`:

```swift
let intermediateSize: Int  // ❌ Fails with array
```

## Solution

Introduced an `IntOrArray` type that can decode either format:

```swift
public struct IntOrArray: Codable {
    public let values: [Int]

    // Decodes both Int and [Int]
    public init(from decoder: Decoder) throws { ... }

    // Access by layer index
    public subscript(layerIdx: Int) -> Int { ... }
}
```

This maintains **full backwards compatibility** with models using a single value while adding support for the per-layer array format.

## Changes

- Added `IntOrArray` type for flexible decoding
- Changed `intermediateSize` type from `Int` to `IntOrArray`
- Updated `Gemma3nMLP.init` to use layer-indexed access: `config.intermediateSize[layerIdx]`

## Testing

- ✅ `swift build` compiles successfully
- ✅ Existing models with single `intermediate_size` value continue to work
- ✅ Gemma 3n models with array format can now be loaded

## Affected Models

This fix enables loading of:
- `mlx-community/gemma-3n-E2B-it-4bit`
- `mlx-community/gemma-3n-E4B-it-4bit`
- `mlx-community/gemma-3n-E2B-it-bf16`
- `mlx-community/gemma-3n-E4B-it-bf16`
- And other Gemma 3n variants on HuggingFace

