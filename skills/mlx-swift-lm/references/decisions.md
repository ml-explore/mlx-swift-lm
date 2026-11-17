# Decision Models

## Overview

A decision model answers typed questions about a state in one forward pass. It returns the
probability of each answer option and generates no tokens. The state can be text, JSON, or
images, depending on the model. Everything is in the `MLXDecisions` library
product. The DocC page `Libraries/MLXDecisions/Documentation.docc/Documentation.md` is the
full reference, with the precision notes.

## Quick Reference

| Type | Purpose |
|------|---------|
| `DecisionModelFactory` | Loads a decision checkpoint (use `.shared`) |
| `DecisionContainer` | Thread-safe model wrapper; validates and runs requests with `predict` |
| `DecisionQuestion` | `.choice`, `.score`, or `.bool` question |
| `DecisionOption` | Choice option: label plus optional description |
| `DecisionState` | `.text(String)` or `.json(DecisionJSON)`; a string literal is text |
| `DecisionJSON` | JSON whose objects keep member order |
| `DecisionRequest` | State, named questions, images |
| `DecisionExecutionOptions` | `tokenBudget` (default 16,384) |
| `DecisionResponse` | `answers` in request order, `inputTokens`, `modelID`, `response["name"]` |
| `DecisionAnswer` | `value`, `probabilities` in option order, `confidence` |
| `DecisionCapabilities` | Question kinds, media, and score level limit of a loaded model |
| `DecisionError` | Validation and model errors |

## Models

| Checkpoint | Inputs | Size |
|------------|--------|------|
| `LiquidAI/d1-3B` (bfloat16) | text, JSON, images | 6.2 GB |
| `LiquidAI/d1-3B-MLX-4bit` | text, JSON, images | 2.4 GB |
| `LiquidAI/d1-omni-600M` (float32) | text, JSON, images | 2.3 GB |
| `LiquidAI/d1-omni-600M-MLX-8bit` | text, JSON, images | 0.7 GB |

- On an iPhone, use `LiquidAI/d1-3B-MLX-4bit`: the bfloat16 d1-3B does not fit in an app's
  memory. d1-omni-600M runs in float32 or 8-bit.
- LiquidAI also publishes `d1-3B-MLX-8bit`, `d1-3B-MLX-fp16` and `d1-omni-600M-MLX-fp16`.
  Avoid 4-bit d1-omni conversions: they change many answers.
- The factory routes by `config.json`, so any `mlx_vlm.convert` conversion loads.
  `VLMModelFactory` loads d1-3B as a plain LFM2-VL chat model.

## Loading

Add the `MLXDecisions` library product to your target, next to `MLXLMCommon`.

```swift
import HuggingFace
import MLXDecisions
import MLXHuggingFace  // macros: #hubDownloader / #huggingFaceTokenizerLoader
import MLXLMCommon
import Tokenizers

let decider = try await DecisionModelFactory.shared.loadContainer(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    id: "LiquidAI/d1-3B"
)
```

`loadContainer(from: directoryURL, using:)` loads a model directory that is already on disk.

## Asking Questions

```swift
let response = try await decider.predict(
    state: "I was charged twice, please refund one.",
    questions: [
        "refund": .bool("Is the customer asking for a refund?"),
        "team": .choice(
            "Which team should handle this?",
            options: [
                DecisionOption("billing", description: "Charges, refunds"),
                DecisionOption("technical", description: "App faults"),
            ]),
        "urgency": .score(
            "How urgent is this?",
            levels: ["Can wait", "Today", "Blocking the customer now"]),
    ]
)

if case .choice(let team) = response["team"]?.value {
    print(team)  // "billing"
}
```

- Answers keep the order of the `questions:` literal. Names must be unique.
- Choice needs at least 2 options with unique, non-empty labels. Score takes 2 to 10 levels,
  lowest first.
- To build questions at run time, pass `DecisionRequest(state:questions:images:)` with an
  array of `DecisionRequest.NamedQuestion` to `predict(_:options:)`.

| Kind | `value` | `probabilities` |
|------|---------|-----------------|
| Choice | Most probable label | One per option, option order |
| Score | Expected level `Σ i × p(i)`, levels from 0 | One per level, lowest first |
| Bool | `p(true) >= 0.5` | `[p(false), p(true)]` |

`confidence` is the largest probability. `inputTokens` counts input positions as the model's
reference counts them.

## States and Images

- JSON states keep member order, which changes answers:
  `.json(["order": 1042, "message": "The mug arrived broken."])`. A request needs a state or
  images.
- Images: file or `data:` URLs, `CIImage`, or `MLXArray`. Files are turned upright by their
  EXIF orientation.
- The library does not download media. Download remote files first: other URLs throw
  `invalidImage`.

## Memory and Token Budget

- `predict(..., options: DecisionExecutionOptions(tokenBudget: 8_192))` bounds the padded token
  positions in one model batch. A lower budget lowers peak memory for requests with many
  questions, at the cost of more batches. Budgets under a few hundred tokens make long d1-3B
  states slow.
- Set `Memory.cacheLimit` (the example apps use 20 MB) in iOS apps and long-running macOS
  processes. Without a limit, MLX's buffer cache can hold gigabytes of freed memory.

## Errors and Concurrency

- `predict` throws `DecisionError`. The container checks the request before the model runs,
  for example `duplicateQuestionName`, `tooFewCriteria`, `missingState`, `unsupportedMedia`,
  or `invalidImage` for a remote URL. The models throw `mediaTooLong`, `optionsDoNotFit`, and
  `invalidImage` for images they cannot read.
- Loading a checkpoint that is not a decision model throws `DecisionError.unsupportedModel`.
- `DecisionContainer` is `Sendable`. Share one container across tasks; requests run one at a
  time. `DecisionRequest` is not `Sendable`, and `predict` consumes it.
- Cancelling the calling task stops the request between model batches and throws
  `CancellationError`.

## Precision

Answers can differ from the Python models. See the DocC page for details.

- The models run in the precision of their weights. Against float32, the bfloat16 d1-3B moves
  probabilities by a few hundredths and the 4-bit d1-3B by up to about 0.3.
- swift-transformers before 1.4.0 tokenizes some d1-3B text differently from Python, which
  changes a few answers. Use 1.4.0 or later.
- Images with a color profile other than sRGB are converted to sRGB; Python reads their stored
  values. JPEG decodes a few levels away from Python.
