# Guided generation

Constrain MLX model output to a JSON Schema using xgrammar.

## Overview

When you pass a `FoundationModels.GenerationSchema` to
`LanguageModelSession.respond(to:schema:)`, the framework asks the
underlying model to emit text conforming to that schema. For the system
language model, schema enforcement is built in. For an MLX model, the
schema is enforced by `MLXFoundationModels` via the vendored xgrammar
library: at every sampling step, xgrammar computes the set of
grammar-legal next tokens and a logit mask is applied so the sampler
cannot drift outside the grammar.

The resulting text is guaranteed to be valid JSON instance of the schema,
not just probably-valid: even with temperature > 0 the model cannot emit
a token that would break the structure.

## Where the engine lives

Schema enforcement is implemented by the separate `MLXGuidedGeneration`
library (the vendored xgrammar engine). `MLXFoundationModels` depends on it
whenever the FoundationModels adapter is compiled in (the
`FoundationModelsIntegration` trait, default-on), so guided output is always
available alongside the adapter. The FoundationModels-to-grammar glue lives in
`Libraries/MLXFoundationModels/GuidedGeneration/SchemaConverter.swift`.

To use guided generation without FoundationModels (older OS floors),
depend on the `MLXGuidedGeneration` product directly.

## Cold-compile latency and `@MainActor`

> Warning: `GuidedGenerationLoop.run` may block for hundreds of
> milliseconds on cold grammar compile — the first call for a given
> schema/grammar on a given tokenizer compiles the grammar and builds
> an adaptive token mask, and neither step yields. Do not invoke from
> `@MainActor`; wrap the call in `Task.detached` or dispatch onto a
> background executor. Subsequent calls against the same compiled
> grammar + tokenizer pair reuse the cached matcher state and do not
> pay the compile cost again.
>
> Pre-warming an expected schema with a throwaway `GrammarConstraint` from a
> background task before the user-visible request lands eliminates the
> blocking window entirely.

## When does this matter?

Schema enforcement is most valuable when:

- The downstream code parses the model's output as JSON. Without
  enforcement you must defend against partial JSON, trailing text, fenced
  code blocks, and the rest of the failure modes that come with
  free-form generation.
- The schema has tight constraints (enums with a small candidate set,
  `minItems`/`maxItems`, length bounds). The constraint search rules out
  large swaths of the vocabulary, often improving both quality and speed.
- Tool calling. `MLXFoundationModels` builds a `oneOf`-style envelope
  schema from the developer's tool definitions; the model can only emit
  a structurally-valid tool call.

For pure chat / completion with no schema, guided generation doesn't change
output behavior. To skip compiling the xgrammar source tree entirely, don't
link `MLXFoundationModels` or `MLXGuidedGeneration`.
