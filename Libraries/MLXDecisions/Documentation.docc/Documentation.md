# ``MLXDecisions``

Load decision models that answer typed questions about a state in one forward pass.

## Overview

A decision model reads a state and a set of named questions. It returns the probability of
each answer option without generating tokens. The state can be text, JSON, or images,
depending on the model.

``DecisionModelFactory`` reads the checkpoint's `config.json` and picks the matching
implementation, so the calling code is the same for every supported model. It returns a
``DecisionContainer``, which checks each request against the model's capabilities before the
model runs. Callers also import MLXLMCommon, for `UserInput` media and the downloader and
tokenizer loader protocols.

## Loading A Decision Model

Pass the same downloader and tokenizer loader the other mlx-swift-lm model factories take.
With the MLXHuggingFace macros, these are `#hubDownloader()` and
`#huggingFaceTokenizerLoader()`.

```swift
import Foundation
import MLXDecisions
import MLXLMCommon

let decider = try await DecisionModelFactory.shared.loadContainer(
    from: downloader,
    using: tokenizerLoader,
    id: "LiquidAI/d1-3B"
)
```

To load a model directory that is already on disk, pass its URL:

```swift
let decider = try await DecisionModelFactory.shared.loadContainer(
    from: URL(filePath: "/models/d1-3B"),
    using: tokenizerLoader
)
```

`decider.capabilities` lists the question kinds and media the loaded model accepts.

With `.vision` in `ModelConfiguration.excludedComponents`, d1-3B loads without its vision
tower. `capabilities` still lists images, and a request with images throws
`OnDemandComponentError.unavailable(.vision)`.

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
    print(team)
}
```

Each question has a name, unique within the request. There are three kinds of question:

- **Choice** picks one of at least two options. The option label is the answer. Labels must
  be unique and not empty. An option can also have a description.
- **Score** rates on 2 to 10 levels, lowest first. The model reads the level descriptions.
- **Bool** answers yes or no. `whenTrue` and `whenFalse` optionally say what each answer
  means.

d1-3B reads each option's description, or the label with `_` read as a space when there is
no description.

To build questions at run time, create a ``DecisionRequest`` from an array of named questions:

```swift
let labels = ["billing", "technical", "fraud"]
let request = DecisionRequest(
    state: "I was charged twice, please refund one.",
    questions: [
        DecisionRequest.NamedQuestion(
            name: "team",
            question: .choice(
                "Which team should handle this?",
                options: labels.map { DecisionOption($0) }))
    ])
let response = try await decider.predict(request)
```

## States

A state is `.text` or `.json`. A string literal is a text state.

```swift
let state = DecisionState.json([
    "order": 1042,
    "total": 25.0,
    "message": "The mug arrived broken.",
])
```

A `.json` state holds a `DecisionJSON`, whose objects keep their members in the order you
write them, as a Python dict does. Key order changes the answers, so write the keys in the
order the model should read them.

Numbers are written as their case says: `.double(25.0)` as `25.0` and `.int(25)` as `25`.
A JSON `null` counts as no state, and a JSON string is read as text.

A request needs a state or images. With images, the state is optional.

## Images

```swift
let response = try await decider.predict(
    questions: [
        "cats": .choice("How many cats are there?", options: ["one", "two", "more"])
    ],
    images: [.url(URL(filePath: "cats.png"))]
)
```

Images are read in order. An image can be:

- a file or `data:` URL. Files are turned upright by their EXIF orientation.
- a `CIImage`, read in its color space.
- an `MLXArray`, used as given. Arrays can be `HW`, `HWC`, or `CHW`, with 1 to 4 channels.
  Float arrays whose largest value is at most 1 are scaled by 255.

The library does not download media. Download remote files first: other URLs throw
`invalidImage` before the request waits for the model.

The models read the stored 8-bit pixels of sRGB and untagged images, as transformers'
`load_image` does. Images with another RGB color profile, such as Display P3, Adobe RGB, or a
camera profile, are converted to sRGB first, while `load_image` reads their stored values (see
Precision). d1-3B first scales each image down to at most 1024 × 1024 pixels in area
with Pillow's bicubic filter, as its reference does. Large images are then split into
512-pixel tiles plus a thumbnail, as transformers' LFM2-VL processor does.

## Answers

Answers come back in request order. `response["name"]` finds one by its question name.

| Kind | `value` | `probabilities` |
|---|---|---|
| Choice | `.choice(label)`: the most probable option | One per option, in option order |
| Score | `.score(level)`: the expected level, `Σ level × probability`, levels counted from 0 | One per level, lowest first |
| Bool | `.bool(value)`: `true` when the probability of `true` is at least 0.5 | `[P(false), P(true)]` |

`confidence` is the probability of the most probable option. Probabilities are not rounded.
`response.inputTokens` counts input positions as the model's reference counts them. d1-3B
counts the shared state and images once.

## Token Budget

`DecisionExecutionOptions.tokenBudget` bounds the padded token positions in one model batch,
media included. The default is 16,384. It must be positive.

```swift
let response = try await decider.predict(
    state: "I was charged twice, please refund one.",
    questions: ["refund": .bool("Is the customer asking for a refund?")],
    options: DecisionExecutionOptions(tokenBudget: 8_192)
)
```

Peak memory grows with the budget. A smaller budget lowers it when a request has many
questions, at the cost of more batches, so use one on devices with less memory. A question
that does not fit the budget on its own still runs, in a batch by itself. Very small budgets
make long states slow.

MLX also keeps freed buffers in a cache, which counts toward the process's memory. Cap it with
`Memory.cacheLimit` in iOS apps and in long-running macOS processes, as the example apps do
with 20 MB.

d1-3B batches questions by length. A batch starts with the longest question left, then takes
the next longest ones while they fit the budget and are at least half as long as the first.
So one long question does not pad many short ones. Answers still come back in request order.
In float32 the batch a question runs in changes its probabilities only by rounding, about
1e-5. In bfloat16 it moves them by up to about 0.05.

d1-3B reads the state and images once, then continues each question from that cached prefix.
When they hold over 2,048 tokens, or more than the budget when it is lower, it reads them in
chunks, which bounds the memory they take.

Requests from several tasks run one at a time. Cancelling a task stops its request between
batches and throws `CancellationError`.

## Errors

`predict` throws `DecisionError`. The container checks these before the model runs:

- `unsupportedMedia`: the model does not read images.
- `noQuestions`, `duplicateQuestionName`.
- `unsupportedQuestionKind`, `tooFewCriteria`, `tooManyScoreLevels` (d1-3B accepts at most
  10 levels), `emptyLabel`, `duplicateLabel`.
- `missingState`: there is no state or image.
- `invalidExecutionOption`: the token budget is not positive.
- `invalidImage`: an image URL is not a file or `data:` URL.

The models throw these:

- `invalidImage`: an image cannot be read or used.
- `imagePlaceholderInText`: with images, d1-3B reads the text `<image>` in the state, or in
  the only question, as an image.
- `noSingleTokenAlias`, `missingToken`: the tokenizer lacks a token the model needs, such as
  an answer or an image marker.

The container throws `invalidProbabilities` if a model returns malformed probabilities.

Loading throws `DecisionError.unsupportedModel` for a checkpoint that is not a supported
decision model.

## Supported Models

| Checkpoint | `config.json` | Inputs | Weights |
|---|---|---|---|
| [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) | `model_type` `lfm2_vl` with an `auto_map.AutoModel` that ends in `D1Model` | text, JSON, images | bfloat16 |

d1-3B answers choice, score, and bool questions. The factory selects a model by
`config.json`, not by repository name, so conversions made with `mlx_vlm.convert` load the
same way, quantized ones included. LiquidAI publishes MLX conversions:
`LiquidAI/d1-3B-MLX-4bit`, `LiquidAI/d1-3B-MLX-8bit` and `LiquidAI/d1-3B-MLX-fp16`.

On an iPhone, use `LiquidAI/d1-3B-MLX-4bit` (2.4 GB). The bfloat16 d1-3B (6.2 GB) does not
fit in an app's memory, even with the increased memory limit entitlement.

d1-3B is an LFM2.5-VL fine-tune. `VLMModelFactory` loads it as an LFM2-VL model, so it also
works for chat.

## Precision

The model runs in the precision of the checkpoint's weights.

- d1-3B ships in bfloat16, which moves probabilities by up to about 0.05 from float32. 8-bit
  conversions stay about as close; 4-bit moves probabilities by up to about 0.3 and changes
  some top answers. For float32, convert it with
  `python -m mlx_vlm convert --hf-path LiquidAI/d1-3B --dtype float32 --mlx-path d1-3B-float32`.
  If it fails with "Permission denied" because the Hugging Face cache is read-only, copy the
  model snapshot to a writable directory, make its files writable with `chmod -R u+w`, and
  pass that directory as `--hf-path`.
- In float32, d1-3B matches its PyTorch reference within 1e-5 in probability (2e-6 measured)
  on text and on sRGB or untagged PNG images.
- Images with an RGB color profile other than sRGB are converted to sRGB, while the Python
  reference reads their stored values, so answers about such photos can differ from Python's.
- JPEG files, MPO and Ultra HDR included, decode through Core Image, up to about 4 levels away
  from libjpeg-turbo, which PIL and transformers use. This can move probabilities by a few
  hundredths. Transparent GIF pixels decode as black, and CMYK images can differ by more than
  100 levels.
- swift-transformers before 1.4.0 encodes some d1-3B text differently from Python's
  `tokenizers`, for example a word followed by a blank line, which every d1-3B choice and
  score prompt contains. This can change answers. Use 1.4.0 or later, which has the fixes
  (huggingface/swift-transformers#397 and #398).

## Topics

### Model Loading

- ``DecisionModelFactory``
- ``DecisionContainer``

### Requests

- ``DecisionRequest``
- ``DecisionQuestion``
- ``DecisionQuestionKind``
- ``DecisionOption``
- ``DecisionState``
- ``DecisionJSON``
- ``DecisionMediaKind``
- ``DecisionExecutionOptions``

### Answers

- ``DecisionResponse``
- ``DecisionAnswer``
- ``DecisionCapabilities``

### Errors

- ``DecisionError``
