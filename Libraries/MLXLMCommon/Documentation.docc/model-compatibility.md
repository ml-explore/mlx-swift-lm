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

## Checkpoint Components And Name Mappings

``BaseLanguageModel/prepareCheckpoint(_:)`` runs before quantization and strict
parameter validation. Its ``ModelCheckpoint`` value keeps tensor names, per-file
metadata, and per-layer quantization settings together. The loading sequence is:

1. Read tensors with the metadata of the file that supplied each tensor.
2. Select the requested component and normalize its names and tensor layout.
3. Quantize runtime modules using the normalized layer settings.
4. Update the model with strict parameter validation, then prepare it for inference.

``CheckpointComponent`` declares accepted serialized namespaces, a runtime
destination, and known sibling namespaces to exclude. It resolves exactly one
embedded component. A root component requires an explicit standalone declaration
from the model's configuration. Unknown paths remain for strict validation;
missing or competing components fail before quantization.

``CheckpointNameMapping`` applies ordered prefix replacements and module
exclusions at complete path boundaries. Use ``ModelCheckpoint/mapNames(using:)``
to apply these rules to tensors, source metadata, and layer settings together.
``ModelCheckpoint/mapNames(_:)`` supports custom mappings with the same collision
checks. Aliases cannot overwrite a tensor or a precision declaration.

The shared loader has no architecture-specific namespace or conversion rules.
Models describe their layouts through these value types and implement numeric
transformations in ``BaseLanguageModel/prepareCheckpoint(_:)``. Gemma3n uses the
same mapping API to remove its serialized wrapper while preserving layer
precision. Qwen uses component selection for its MTP predictor. Existing models
still use their sanitizer through the default implementation; migrate renaming
sanitizers to the checkpoint hook so their precision declarations follow renames.

Qwen3.5 targets exclude the MTP component. Drafters accept embedded `mtp.*` and
converted `language_model.mtp.*` components, and unprefixed heads declared as
`qwen3_5_mtp`. An ordinary target without MTP weights is not a standalone head.
Unknown tensors and missing parameters still fail strict loading.

New Swift MTP conversions write `mlx_swift_lm.qwen_mtp.norm_convention=scale`.
`offset` declares upstream zero-centered RMSNorm weights that require adding one;
`scale` declares the runtime multiplicative weights. Explicit declarations take
precedence. For existing exports, standalone heads and `language_model.mtp.*`
use scales; `mtp.*` uses offsets unless its own file declares `format=mlx` or the
caller sets `preconvertedNorms`. Metadata from a target shard does not classify
a separate MTP shard. Conflicting norm declarations fail before model update.

## Components That Load On First Use

Some models have large parts that many sessions never use, such as the vision
encoder of a VLM. Loading them is lazy, like `MLXArray` evaluation: nothing is
read until an input needs it, and apps need no configuration.

A model declares its parts through ``ModelComponentsProviding``. For each
``ModelComponent``, it lists ``CheckpointComponent`` values whose destination is
an optional module and whose namespaces cover every serialized name of that
module. It lists the parts that can load later as ``OnDemandComponent`` values,
which live outside the module tree.

Before reading the checkpoint, the loader removes the modules of each on-demand
component and does not read its tensors. The first input that needs the
component reads only its namespaces. Then the component goes through the same
``BaseLanguageModel/prepareCheckpoint(_:)``, quantization, and strict validation
as the model, and is evaluated. All of this happens under a lock, and the
component is published only after that. The model's module tree does not change
after the load, so sessions that share the model stay safe. Concurrent first uses
share one load. The checkpoint files must stay readable until then.

Model conversion loads with ``ComponentLoading/immediate`` because it saves every
weight. It rebuilds modules that an earlier on-demand load removed. List a
component in ``ModelConfiguration/excludedComponents`` to never load it. Inputs
that need it then fail with an error.

``BaseLanguageModel/excludedCheckpointNamespaces`` is the same read filter for
namespaces that a model never loads. An embedded Qwen MTP drafter excludes the
target namespaces that its component selection drops, so it reads only its own
tensors from a shared checkpoint. A file whose tensors are all excluded still
supplies its metadata.

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
