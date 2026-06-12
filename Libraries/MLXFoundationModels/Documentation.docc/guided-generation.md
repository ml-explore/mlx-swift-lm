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

## The `GuidedGenerationSupport` package trait

xgrammar is opt-in at the package-trait level:

```swift
.package(
    url: "https://github.com/ml-explore/mlx-swift-lm",
    from: "3.32.0",
    traits: ["GuidedGenerationSupport"]   // default ON
)
```

The trait is enabled by default. With it enabled, `MLXFoundationModels`
compiles the vendored xgrammar C++ sources and exposes the
schema-enforcement path. With it disabled (`--disable-default-traits`),
`MLXFoundationModels` still builds and provides chat / tool calling, but
schema-driven respond() calls return unconstrained text.

The trait gate lives in `Libraries/MLXFoundationModels/GuidedGeneration/`:
every file there is wrapped in `#if GuidedGenerationSupport`, so symbols
literally vanish from the binary when the trait is off.

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

For pure chat / completion with no schema, the trait doesn't change
output behavior; you can disable it to skip compiling the xgrammar
source tree.
