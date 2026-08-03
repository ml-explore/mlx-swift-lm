# Model Compatibility

How to decide whether a model repository can be loaded by mlx-swift-lm.

## Overview

mlx-swift-lm loads a model by reading its model directory, decoding
`config.json`, and using metadata such as `model_type` to select a Swift model
implementation. The repository name is only a hint. Compatibility is decided by
the model files, tokenizer files, optional processor files, and the registered
Swift implementation.

A Hugging Face repository can contain valid MLX weights and still be outside
the scope of this package. The package does not execute arbitrary Python model
code from a repository. It loads model families that have Swift implementations
and registry entries.

## Compatibility Checklist

A repository is a good candidate when the following are true:

- `config.json` is present.
- `config.json` has a `model_type` handled by the relevant model factory.
- The required configuration fields are present, or the Swift configuration type
  provides compatible defaults.
- The weights are MLX-loadable `safetensors`. Sharded weights should include the
  corresponding shard index.
- Tensor names and shapes match the Swift model implementation.
- Tokenizer files can be loaded by the configured ``TokenizerLoader``.
- Generation metadata, including EOS token IDs, is present or can be supplied by
  the ``ModelConfiguration``.
- Vision-language models include processor or preprocessor metadata handled by a
  registered processor.
- The quantization format is supported by the current MLX Swift runtime.
- The model fits the target machine's memory budget, including weights, KV
  cache, processor tensors, and prompt context.

## Finding The Source Of Truth

The model factories and registries are the authoritative compatibility list.
Use them instead of copying a static list into application code or
documentation:

- `LLMModelFactory` and `LLMRegistry` for language models.
- `VLMModelFactory`, `VLMRegistry`, and processor registries for
  vision-language models.
- `MLXEmbedders` model factories and registries for embedding models.

Preconfigured ``ModelConfiguration`` values are convenience entries for models
that are expected to work well, but they are not the full compatibility surface.
A repository with the same architecture, compatible files, and matching weights
can also be loaded with an explicit configuration.

## Evaluating A New Repository

1. Inspect `config.json` and identify `model_type`.
2. Check the appropriate factory for a matching Swift model implementation.
3. For VLMs, inspect `processor_config.json`, `preprocessor_config.json`, or
   equivalent processor metadata and check for a registered processor.
4. Confirm that `safetensors` weights and any shard index are present.
5. Confirm tokenizer files are available, or provide a separate tokenizer source.
6. Check generation metadata for EOS tokens, chat templates, and special stop
   strings.
7. Load the model and run a short prompt before relying on long context or large
   batches.

Raw model loading is separate from higher-level behavior. Tool calling, chat
formatting, image preprocessing, video sampling, and stop-string handling may
require package support in addition to compatible weights.

## Common Fixes

Some repositories are compatible after small configuration changes:

- Set ``ModelConfiguration/tokenizerSource`` when the weights repository does
  not contain tokenizer files.
- Use a compatible ``TokenizerLoader`` when tokenizer metadata needs to map to
  an available Swift tokenizer implementation.
- Set ``ModelConfiguration/eosTokenIds`` or
  ``ModelConfiguration/extraEOSTokens`` when generation metadata is missing or
  incomplete.
- Choose a shorter context length or smaller quantization when the model loads
  but exceeds memory during generation.

## Usually Out Of Scope

The following repositories are usually not loadable by this package unless a
matching Swift implementation is added:

- Speech recognition models.
- Text-to-speech models.
- Text-to-image or diffusion models.
- Audio-to-audio models.
- Repositories without a supported `model_type`.
- Repositories that require custom Python preprocessing not implemented in Swift.

For implementing a new architecture, see <doc:porting> and <doc:developing>.
