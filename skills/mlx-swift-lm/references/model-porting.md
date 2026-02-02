# Port MLX-LM Python Models to MLX-Swift-LM

Use this skill when adding a new LLM model port from `mlx-lm` (Python) to this repo's Swift implementation.

## Quick start

1. Identify the Python model file, a reference `config.json`, and the weight files (`.safetensors`).
2. Create the Swift model file in `Libraries/MLXLLM/Models/YourModel.swift`.
3. Add a configuration struct that mirrors `config.json` (including defaults).
4. Implement modules using `@ModuleInfo`/`@ParameterInfo` with keys that match **safetensors** keys.
5. Implement the top-level model (`LLMModel`, `KVCacheDimensionProvider`) and `sanitize(weights:)`.
6. Register the model type in `LLMTypeRegistry`, and optionally add a `ModelConfiguration` in `LLMRegistry`.

## References to open

- `Libraries/MLXLMCommon/Documentation.docc/porting.md`
- `Libraries/MLXLLM/Documentation.docc/adding-model.md`
- Example ports:
  - `Libraries/MLXLLM/Models/Qwen2.swift`
  - `Libraries/MLXLLM/Models/Llama.swift`
  - `Libraries/MLXLLM/Models/MiniCPM.swift`
  - `Libraries/MLXLLM/Models/GPTOSS.swift`

## File structure mapping (Python -> Swift)

Naming is conventional, not required. Helpers can be `private` and do not need a model-name prefix.

| Python pattern | Swift pattern | Example (Qwen2) |
|---|---|---|
| `@dataclass class ModelArgs` | `struct {ModelName}Configuration: Codable, Sendable` | `Qwen2Configuration` |
| `class Attention(nn.Module)` | `class {ModelName}Attention: Module` | `Qwen2Attention` |
| `class MLP(nn.Module)` | `class {ModelName}MLP: Module, UnaryLayer` | `Qwen2MLP` |
| `class TransformerBlock(nn.Module)` | `class {ModelName}TransformerBlock: Module` | `Qwen2TransformerBlock` |
| `class Model(nn.Module)` | `class {ModelName}: Module, LLMModel, KVCacheDimensionProvider` | `Qwen2Model` |

## Configuration mapping

- Mirror `config.json` fields and map snake_case keys with `CodingKeys`.
- Use `decodeIfPresent` for optional fields and provide defaults.
- For GQA models: default `kvHeads` to `attentionHeads` if missing.
- Bias flags and rope settings vary by model; read them from config when present.

Example pattern:

```swift
public struct YourModelConfiguration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var intermediateSize: Int
    var vocabularySize: Int
    var rmsNormEps: Float
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var tieWordEmbeddings: Bool = true

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
}
```

### RoPE and rope_scaling

- `rope_theta` defaults differ by model (e.g., 10_000 vs 1_000_000).
- `rope_scaling` field names vary (`type` vs `rope_type`). Decode defensively.
- Some models need `ropeTraditional` or custom scaling factors.

## Module and weight key mapping

- `@ModuleInfo(key:)` and `@ParameterInfo(key:)` must match actual weight keys.
- Do **not** assume the Python attribute name is the saved key. Verify against `.safetensors` keys.
- The Swift property name can be anything; the key string controls weight loading.

Common keys (model-dependent):

```swift
@ModuleInfo(key: "q_proj") var wq: Linear
@ModuleInfo(key: "k_proj") var wk: Linear
@ModuleInfo(key: "v_proj") var wv: Linear
@ModuleInfo(key: "o_proj") var wo: Linear
@ModuleInfo(key: "gate_proj") var gate: Linear
@ModuleInfo(key: "up_proj") var up: Linear
@ModuleInfo(key: "down_proj") var down: Linear
@ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
@ModuleInfo(key: "self_attn") var attention: YourModelAttention
```

## Model structure patterns

These are reference patterns; adjust to the Python implementation.

### Attention

```swift
final class YourModelAttention: Module {
    let args: YourModelConfiguration
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(_ args: YourModelConfiguration) {
        self.args = args
        let headDim = args.hiddenSize / args.attentionHeads
        self.scale = pow(Float(headDim), -0.5)
        self.rope = RoPE(
            dimensions: headDim,
            traditional: args.ropeTraditional,
            base: args.ropeTheta
        )

        // Check Python + config for bias settings
        _wq.wrappedValue = Linear(args.hiddenSize, args.attentionHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, args.kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(args.hiddenSize, args.kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(args.attentionHeads * headDim, args.hiddenSize, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}
```

### MLP (SwiGLU)

```swift
final class YourModelMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ args: YourModelConfiguration) {
        _gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        _up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        _down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}
```

### Transformer block

```swift
final class YourModelTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: YourModelAttention
    let mlp: YourModelMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: YourModelConfiguration) {
        _attention.wrappedValue = YourModelAttention(args)
        self.mlp = YourModelMLP(args)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}
```

### Model inner

```swift
public final class YourModelModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [YourModelTransformerBlock]
    let norm: RMSNorm

    init(_ args: YourModelConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )
        self.layers = (0 ..< args.hiddenLayers).map { _ in YourModelTransformerBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // Prefer cache-driven masks when cache is present
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let cache = cache?.first {
            mask = cache.makeMask(n: h.dim(1), windowSize: nil, returnArray: false)
        } else {
            mask = .causal
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}
```

### Top-level model

```swift
public final class YourModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let headDim: IntOrPair

    fileprivate let model: YourModelModelInner

    // Only if tie_word_embeddings is false
    // @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: YourModelConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.headDim = .init(args.hiddenSize / args.attentionHeads)
        self.model = YourModelModelInner(args)

        // Only if tie_word_embeddings is false
        // _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return model.embedTokens.asLinear(out)
        // If tie_word_embeddings is false and you defined lmHead:
        // return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            !key.contains("rotary_emb.inv_freq")
        }
    }
}
```

## Tied embeddings

If `tie_word_embeddings` is true, return `model.embedTokens.asLinear(out)`.
If false, create a separate `lm_head` and return `lmHead(out)`.

## sanitize(weights:)

Use `sanitize(weights:)` to remove or rename keys that should not be loaded or need reshaping:

- Remove precomputed rotary frequencies (key pattern varies, e.g. `rotary_emb.inv_freq`).
- Remove `lm_head.weight` if embeddings are tied.
- Re-root keys if weights are nested (e.g. `language_model.*`, `model.*`).

## LoRA

Models must conform to `LoRAModel` and expose the transformer layers:

```swift
extension YourModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}
```

If you need custom keys, override `loraDefaultKeys`.

## Registration

1. Add model type mapping in `Libraries/MLXLLM/LLMModelFactory.swift`:

```swift
"your_model_type": create(YourModelConfiguration.self, YourModel.init),
```

2. Optional: add a `ModelConfiguration` in `LLMRegistry` (also in `MLXLLM/LLMModelFactory.swift`). If that registry exposes a list (e.g., `all()`), include the new configuration there.

## Common pitfalls

- Weight keys do not always match Python attribute names; verify `.safetensors` keys.
- `rope_scaling` fields differ across models (`type` vs `rope_type`, factor requirements).
- Bias flags are model-specific (check config and Python implementation).
- GQA models require `kvHeads` distinct from `attentionHeads`.
- Sliding-window or special caches may require overriding `newCache` or `prepare`.

## Minimal checklist

- `Libraries/MLXLLM/Models/YourModel.swift` created
- Config struct with correct keys and defaults
- Weight keys verified against safetensors
- `sanitize(weights:)` handles extra keys
- `LoRAModel` conformance (`loraLayers`)
- `LLMTypeRegistry` registration
- Optional `ModelConfiguration` added to `LLMRegistry`
- Smoke test with at least one model ID

## Testing

```swift
let modelContainer = try await LLMModelFactory.shared.loadContainer(
    configuration: ModelConfiguration(id: "mlx-community/YourModel-4bit")
)

let parameters = GenerateParameters()
let input = try await modelContainer.prepare(
    input: UserInput(prompt: "Hello")
)

let stream = try await modelContainer.generate(input: input, parameters: parameters)
for await event in stream {
    if case let .chunk(text) = event {
        print(text, terminator: "")
    }
}
```
