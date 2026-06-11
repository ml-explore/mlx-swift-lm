# Model Compatibility

How to decide whether a Hugging Face model is suitable for this MLX Swift inference engine.

## Overview

mlx-swift-lm loads a model by reading its local or downloaded model directory, decoding
`config.json`, and using the `model_type` field to select a registered Swift model
implementation. A model is suitable when its architecture, tokenizer, processor, and
weights match what the relevant factory can load.

The model repository name is only a hint. Compatibility is decided by files and metadata,
especially `config.json`.

## Compatibility Checklist

A model is suitable for this engine when all of the following are true:

- `config.json` is present.
- `config.json` has a `model_type` that is registered by the relevant factory.
- The fields needed by that Swift configuration type are present, or the Swift
  configuration provides defaults for missing fields.
- The weight files are MLX-loadable `safetensors`, usually `*.safetensors` plus
  `model.safetensors.index.json` for sharded models.
- The weight tensor names and shapes match the Swift model implementation.
- A tokenizer can be loaded through the configured `TokenizerLoader`, usually from
  `tokenizer.json` and `tokenizer_config.json`.
- Generation metadata is present or can be inferred, especially EOS token IDs.
- For VLMs, processor files are present and the processor class is registered.
- The quantization format is supported by the current MLX Swift runtime.
- The model fits the target Apple Silicon memory budget after weights, KV cache,
  processor tensors, and prompt context are included.

If any of these fail, the model may be an MLX model in the broad ecosystem but still not
be loadable by this package.

## Why Registered Models Work

The factories do not inspect arbitrary Python code from a model repository. They map a
small `model_type` string to a Swift implementation.

For LLMs, `LLMTypeRegistry` maps `model_type` values to `LanguageModel` implementations.
For VLMs, `VLMTypeRegistry` maps `model_type` values to multimodal model
implementations, and `VLMProcessorTypeRegistry` maps processor classes to image or video
preprocessors. For text embeddings, `EmbedderTypeRegistry` maps `model_type` values to
embedding model implementations.

This is why an unknown model architecture cannot be made compatible by only converting
weights to `safetensors`. There must also be Swift code that implements the architecture
and a registry entry that selects it.

## Supported LLM Model Types

The current LLM registry supports these `model_type` values:

| `model_type` | Model families that should match |
| --- | --- |
| `llama` | Llama 2, Llama 3, Llama 3.1, Llama 3.2, Llama 3.3, CodeLlama, many Llama-derived fine-tunes |
| `mistral` | Mistral 7B, Mistral Nemo, Codestral, earlier Mistral-family decoder models |
| `mistral3` | Ministral 3, Mistral Small 3.x/4.x, Devstral, newer Mistral-family text models using the Mistral 3 config shape |
| `qwen2` | Qwen1.5, Qwen2, Qwen2.5, Qwen2.5 Coder, QwQ and Qwen-based distillations that keep the Qwen2 config shape |
| `qwen3` | Qwen3 dense text models and Qwen3 embedding models when used through the embedder factory |
| `qwen3_moe` | Qwen3 MoE and Qwen3 Coder MoE families |
| `qwen3_next` | Qwen3 Next and Qwen3 Coder Next families |
| `qwen3_5` | Qwen3.5 and Qwen3.6 dense text models using the Qwen3.5 config shape |
| `qwen3_5_moe` | Qwen3.5 and Qwen3.6 MoE models |
| `qwen3_5_text` | Text-only Qwen3.5 variants with the dedicated text config |
| `gemma` | Gemma 1, CodeGemma, and compatible early Gemma-style decoder models |
| `gemma2` | Gemma 2 models |
| `gemma3`, `gemma3_text` | Gemma 3 text models and text-only slices of Gemma 3 VLM repositories |
| `gemma3n` | Gemma 3n language-model variants |
| `gemma4`, `gemma4_text` | Gemma 4 multimodal or text-only variants, depending on factory |
| `phi` | Phi-2 and some Phi-4 dense variants that use the supported Phi config |
| `phi3` | Phi-3, Phi-3.5 mini, Phi-4 mini variants that keep the Phi3 config shape |
| `phimoe` | Phi-3.5 MoE |
| `deepseek_v3` | DeepSeek V3 and DeepSeek R1 style MoE models using the DeepSeek V3 architecture |
| `glm4` | GLM-4 dense models |
| `glm4_moe` | GLM-4.5, GLM-4.6, GLM-4.7 MoE models |
| `glm4_moe_lite` | GLM-4.x Flash or lite MoE variants |
| `gpt_oss` | GPT-OSS models in compatible MLX format |
| `granite` | IBM Granite dense decoder models |
| `granitemoehybrid` | Granite 4 hybrid MoE models |
| `lfm2` | LFM2 and LFM2.5 text models |
| `lfm2_moe` | LFM2 MoE models |
| `falcon_h1` | Falcon H1 and Falcon H1R models |
| `bitnet` | BitNet and compatible low-bit BitNet-style models |
| `smollm3` | SmolLM3 text models |
| `cohere` | Cohere Command-R, Command-R+, Aya-style compatible models |
| `openelm` | Apple OpenELM |
| `internlm2` | InternLM2 and InternLM2.5 compatible models |
| `minicpm` | MiniCPM v1, v2, and v4 style text models supported by the implementation |
| `starcoder2` | StarCoder2 code models |
| `mimo`, `mimo_v2_flash` | MiMo and MiMo V2 Flash families |
| `minimax` | MiniMax M2 and M2.1 compatible models |
| `acereason` | AceReason models that use the supported Qwen2-compatible path |
| `ernie4_5` | ERNIE 4.5 models |
| `baichuan_m1` | Baichuan M1 |
| `exaone4` | EXAONE 4 |
| `lille-130m` | Lille 130M |
| `olmo2`, `olmo3`, `olmoe` | OLMo 2, OLMo 3, and OLMoE |
| `bailing_moe` | Bailing, Ling, and Ring MoE families using this architecture |
| `nanochat` | NanoChat |
| `nemotron_h` | Nemotron H and compatible Nemotron hybrid families |
| `afmoe` | AfMoE and Trinity-family models using this architecture |
| `jamba` | AI21 Jamba |
| `apertus` | Apertus |

## Supported VLM Model Types

The current VLM registry supports these `model_type` values:

| `model_type` | Model families that should match |
| --- | --- |
| `paligemma` | PaliGemma and PaliGemma 2 variants with compatible processor files |
| `qwen2_vl` | Qwen2-VL |
| `qwen2_5_vl` | Qwen2.5-VL and compatible UI/OCR models using that architecture |
| `qwen3_vl` | Qwen3-VL image-text models |
| `qwen3_5` | Qwen3.5 multimodal repositories with compatible Qwen3.5 config |
| `qwen3_5_moe` | Qwen3.5 or Qwen3.6 multimodal MoE repositories with compatible config |
| `idefics3` | IDEFICS 3 and some SmolVLM repositories tagged with this architecture |
| `gemma3` | Gemma 3 VLM repositories |
| `gemma4` | Gemma 4 VLM repositories |
| `smolvlm` | SmolVLM2 video/image-text repositories |
| `fastvlm` | FastVLM |
| `llava_qwen2` | LLaVA-Qwen2 repositories using the FastVLM-compatible path |
| `pixtral` | Pixtral and compatible multimodal Mistral-style models |
| `mistral3` | Mistral 3 VLM models |
| `lfm2_vl`, `lfm2-vl` | LFM2-VL and LFM2.5-VL |
| `glm_ocr` | GLM OCR |

VLMs also require a registered processor class. The supported processor classes are
`PaliGemmaProcessor`, `Qwen2VLProcessor`, `Qwen2_5_VLProcessor`, `Qwen3VLProcessor`,
`Idefics3Processor`, `Gemma3Processor`, `Gemma4Processor`, `SmolVLMProcessor`,
`FastVLMProcessor`, `PixtralProcessor`, `Mistral3Processor`, `Lfm2VlProcessor`, and
`Glm46VProcessor`.

## Supported Embedder Model Types

The embedder registry supports these `model_type` values:

| `model_type` | Model families that should match |
| --- | --- |
| `bert` | BERT-based embedding models such as BGE small/base/large and MiniLM-style repositories that expose a BERT config |
| `roberta` | RoBERTa-based embedding models |
| `xlm-roberta` | XLM-R based multilingual embedding models such as BGE-M3 and multilingual E5 variants |
| `distilbert` | DistilBERT embedding models |
| `nomic_bert` | Nomic Embed Text v1 and v1.5 |
| `qwen3` | Qwen3 embedding models |
| `gemma3`, `gemma3_text`, `gemma3n` | Gemma-based embedding models such as EmbeddingGemma when the config shape matches |

## Known Preconfigured Models

The registries include convenience `ModelConfiguration` values for commonly tested
models. These are not the only compatible models; any repository with a matching
`model_type`, compatible files, and loadable weights can be loaded by ID or local
directory.

Representative preconfigured LLMs include:

- `mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX`
- `mlx-community/Mistral-Nemo-Instruct-2407-4bit`
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit`
- `mlx-community/phi-2-hf-4bit-mlx`
- `mlx-community/Phi-3.5-mini-instruct-4bit`
- `mlx-community/Phi-3.5-MoE-instruct-4bit`
- `mlx-community/gemma-2-2b-it-4bit`
- `mlx-community/gemma-2-9b-it-4bit`
- `mlx-community/gemma-3-1b-it-qat-4bit`
- `mlx-community/gemma-3n-E2B-it-lm-4bit`
- `mlx-community/gemma-3n-E4B-it-lm-4bit`
- `mlx-community/gemma-4-e2b-it-4bit`
- `mlx-community/gemma-4-e4b-it-4bit`
- `mlx-community/Qwen1.5-0.5B-Chat-4bit`
- `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- `mlx-community/Qwen2.5-7B-Instruct-4bit`
- `mlx-community/Qwen3-0.6B-4bit`
- `mlx-community/Qwen3-1.7B-4bit`
- `mlx-community/Qwen3-4B-4bit`
- `mlx-community/Qwen3-8B-4bit`
- `mlx-community/Qwen3-30B-A3B-4bit`
- `mlx-community/Qwen3.5-2B-4bit`
- `mlx-community/Qwen3.6-27B-4bit`
- `mlx-community/OpenELM-270M-Instruct`
- `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
- `mlx-community/Meta-Llama-3-8B-Instruct-4bit`
- `mlx-community/Llama-3.2-1B-Instruct-4bit`
- `mlx-community/Llama-3.2-3B-Instruct-4bit`
- `mlx-community/DeepSeek-R1-4bit`
- `mlx-community/granite-3.3-2b-instruct-4bit`
- `mlx-community/MiMo-7B-SFT-4bit`
- `mlx-community/GLM-4-9B-0414-4bit`
- `mlx-community/AceReason-Nemotron-7B-4bit`
- `mlx-community/bitnet-b1.58-2B-4T-4bit`
- `mlx-community/Baichuan-M1-14B-Instruct-4bit-ft`
- `mlx-community/SmolLM3-3B-4bit`
- `mlx-community/ERNIE-4.5-0.3B-PT-bf16-ft`
- `mlx-community/LFM2-1.2B-4bit`
- `mlx-community/exaone-4.0-1.2b-4bit`
- `mlx-community/lille-130m-instruct-bf16`
- `mlx-community/OLMoE-1B-7B-0125-Instruct-4bit`
- `mlx-community/OLMo-2-1124-7B-Instruct-4bit`
- `mlx-community/Ling-mini-2.0-2bit-DWQ`
- `mlx-community/Granite-4.0-H-Tiny-4bit-DWQ`
- `mlx-community/LFM2-8B-A1B-3bit-MLX`
- `dnakov/nanochat-d20-mlx`
- `mlx-community/gpt-oss-20b-MXFP4-Q8`
- `mlx-community/AI21-Jamba-Reasoning-3B-4bit`

Representative preconfigured VLMs include:

- `mlx-community/paligemma-3b-mix-448-8bit`
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`
- `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`
- `lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit`
- `mlx-community/Qwen3-VL-4B-Instruct-8bit`
- `mlx-community/SmolVLM-Instruct-4bit`
- `mlx-community/LFM2.5-VL-1.6B-4bit`
- `mlx-community/LFM2-VL-1.6B-4bit`
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`
- `mlx-community/gemma-3-4b-it-qat-4bit`
- `mlx-community/gemma-3-12b-it-qat-4bit`
- `mlx-community/gemma-3-27b-it-qat-4bit`
- `mlx-community/gemma-4-e2b-it-4bit`
- `mlx-community/gemma-4-e4b-it-4bit`
- `mlx-community/gemma-4-26b-a4b-it-4bit`
- `mlx-community/gemma-4-31b-it-4bit`
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx`
- `mlx-community/FastVLM-0.5B-bf16`

Representative preconfigured embedders include:

- `TaylorAI/bge-micro-v2`
- `TaylorAI/gte-tiny`
- `sentence-transformers/all-MiniLM-L6-v2`
- `Snowflake/snowflake-arctic-embed-xs`
- `sentence-transformers/all-MiniLM-L12-v2`
- `BAAI/bge-small-en-v1.5`
- `intfloat/multilingual-e5-small`
- `BAAI/bge-base-en-v1.5`
- `nomic-ai/nomic-embed-text-v1`
- `nomic-ai/nomic-embed-text-v1.5`
- `BAAI/bge-large-en-v1.5`
- `Snowflake/snowflake-arctic-embed-l`
- `BAAI/bge-m3`
- `mixedbread-ai/mxbai-embed-large-v1`
- `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`

## Models That Are Usually Not Suitable

These may be valid MLX ecosystem models, but they are outside the LLM, VLM, and embedder
factories described here:

- Speech recognition models, such as Whisper or Parakeet.
- Text-to-speech models, such as Kokoro or Fish Audio.
- Text-to-image or diffusion models, such as FLUX.
- Audio-to-audio models.
- Models whose `model_type` is missing or not registered.
- Models that require custom Python-side preprocessing not implemented in Swift.
- Models with compatible architecture names but incompatible tensor names or config fields.

## How To Evaluate A New Candidate

1. Inspect `config.json`.
2. Check that `model_type` is listed in this document.
3. For VLMs, inspect `processor_config.json` or `preprocessor_config.json` and check that
   the processor class is registered.
4. Confirm the repository has `safetensors` weights and any shard index file.
5. Confirm tokenizer files are available or configure a separate tokenizer source.
6. Check for special stop tokens or tool-call formats.
7. Try a small prompt with a short generation limit before relying on long-context usage.

For model families with tool calling or nonstandard chat templates, compatibility with
raw inference does not automatically mean compatibility with every chat or tool-calling
workflow. Those behaviors depend on tokenizer chat templates, generation config, and
package-level tool call formatting.

## Common Fixes For Otherwise Compatible Models

- Add `extraEOSTokens` when a model uses additional textual stop markers.
- Set `eosTokenIds` when EOS IDs are missing or wrong in generation metadata.
- Use `tokenizerId` or `tokenizerSource` when the weights repository does not contain
  usable tokenizer files.
- Use `overrideTokenizer` when a tokenizer config names a class that needs to be mapped
  to an available tokenizer implementation.
- Set a tool-call format override for families that need one, such as GLM4 or LFM2.
- Use a smaller quantization or shorter context when the model loads but runs out of
  memory during generation.

## Sources Of Truth

Use the local source first:

- `Libraries/MLXLLM/LLMModelFactory.swift`
- `Libraries/MLXVLM/VLMModelFactory.swift`
- `Libraries/MLXEmbedders/ModelFactory.swift`

Then validate candidate repositories on Hugging Face by checking their `config.json`,
model card tags, tokenizer files, processor files, and weight files.
