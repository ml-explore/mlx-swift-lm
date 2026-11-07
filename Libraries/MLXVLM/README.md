# MLXVLM

# Documentation

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm) -- large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm) -- vision language model example implementations

# Quick Start

Using LLMs and VLMs from MLXLMCommon is as easy as:

```swift
import Foundation
import MLXVLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

let model = try await loadModel(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    id: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
)
let session = ChatSession(model)

let answer1 = try await session.respond(
    to: "what kind of creature is in the picture?",
    image: .url(URL(fileURLWithPath: "support/test.jpg"))
)
print(answer1)

// we can ask a followup question referring back to the previous image
let answer2 = try await session.respond(
    to: "What is behind the dog?"
)
print(answer2)
```

For more information see 
[Evaluation](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/evaluation)
or [Using Models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using-model)
for more advanced API.

# Contents

This is a port of several models from:

- https://github.com/Blaizzy/mlx-vlm

Tokenization is provided via the `TokenizerLoader` protocol – see the main
[README](../../README.md) for available integration packages.

The [VLMModelFactory.swift](VLMModelFactory.swift) provides minor overrides and customization --
if you require overrides for the tokenizer or prompt customizations they can be
added there.

This is set up to load models from Hugging Face, e.g. https://huggingface.co/mlx-community

The following models have been tried:

- mlx-community/paligemma-3b-mix-448-8bit
- mlx-community/Qwen2-VL-2B-Instruct-4bit
- mlx-community/Qwen2.5-VL-3B-Instruct-4bit
- lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit
- mlx-community/Qwen3-VL-4B-Instruct-8bit
- mlx-community/SmolVLM-Instruct-4bit
- mlx-community/gemma-3-4b-it-qat-4bit
- mlx-community/gemma-3-12b-it-qat-4bit
- mlx-community/gemma-3-27b-it-qat-4bit
- HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx
- mlx-community/Muse-Glimmer-30B-4bit

Currently supported model types are:

- paligemma
- qwen2_vl
- qwen2_5_vl
- qwen3_vl
- idefics3
- gemma3
- smolvlm
- muse_glimmer

See [llm-tool](../../Tools/llm-tool)

# Adding a Model

If the model follows the typical VLM pattern:

- `config.json`, `tokenizer.json`, and `tokenizer_config.json`
- `*.safetensors`

You can follow the pattern of the models in the [Models](Models) directory
and create a `.swift` file for your new model:

## Create a Model Configuration

Create a configuration struct for both the Text and Vision models
that matches the structure in `config.json`. A struct like this
is recommended:

```swift
public struct YourModelConfiguration: Codable, Sendable {
    public struct TextConfiguration: Codable, Sendable {
        public let hiddenSize: Int

        // use this pattern for values that need defaults
        public let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }
        
        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case _layerNormEps = "layer_norm_eps"
        }
    }
    
    public struct VisionConfiguration: Codable, Sendable {
        ...
    }
    
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let vocabularySize: Int

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case vocabularySize = "vocab_size"
    }
}
```

## Create a Processor Configuration

VLMs also require a image/video preprocessor. Create a configuration to match 
the `preprocessor_config.json` file:

```swift
public struct YourModelProcessorConfiguration: Codable, Sendable {

    public struct Size: Codable, Sendable {
        public let width: Int
        public let height: Int

        var cgSize: CGSize { .init(width: width, height: height) }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
    }
}
```

this will be consumed by:

```swift
public struct YourModelProcessor: UserInputProcessor {
...
```

discussed later.

## Create the Vision, Text and VLM Classes

VLMs have language and vision models that are aggregated into a single
top level model.

For purposes of name spacing you might put the Language and Vision
models into an `enum` to create something structured like this:

```swift
// MARK: - Language

private enum Language {

    fileprivate class Attention: Module {
        ...
    }
    
    ...
    
    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: YourModel

        var kvHeads: [Int]
        var headDim: MLX.IntOrPair

        public init(_ args: YourModelConfiguration.TextConfiguration) {
            self.model = YourModel(args)

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
            }

        public func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            mask: MLXArray? = nil
        ) -> LMOutput {
            ...
            return LMOutput(logits: ...)
        }
    }
}
```

Similarly the Vision model can go into an `enum` namespace:

```swift
// MARK: - Vision

private enum Vision {

    fileprivate class Attention: Module {
        ...
    }
    
    fileprivate class VisionModel: Module {

        @ModuleInfo(key: "vision_model") var visionModel: InternalVisionModel

        public init(_ config: YourModelConfiguration.VisionConfiguration) {
            self._visionModel.wrappedValue = InternalVisionModel(config)
        }

        public func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, MLXArray?
        ) {
            visionModel(x, outputHiddenStates: outputHiddenStates)
        }
    }
}
```

The exact signatures on the `init()` and `callAsFunction()` can vary as needed --
these models are not exposed to callers.

The top level model is the only piece of the model with public API and it
should implement `VLMModel` (aka `LanguageModel`). Here is an outline of how
the top level model might work:

```swift
public class YourModel: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel

    public let config: YourModelConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var headDim: MLX.IntOrPair { languageModel.headDim }

    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public init(_ config: YourModelConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws
        -> PrepareResult
    {
        // Merge image and text embeddings, then prefill the KV cache in
        // chunks. Single-pass prefill allocates transient buffers
        // proportional to prompt length and causes OOM on long prompts.
        guard let image = input.image else { throw VLMError.imageRequired }
        guard let mask = input.text.mask else { throw VLMError.maskRequired }
        var inputIds = input.text.tokens
        if inputIds.ndim == 1 { inputIds = inputIds.expandedDimensions(axis: 0) }

        let allEmbeds = inputEmbeddings(
            inputIds: inputIds, pixelValues: image.pixels, mask: mask)

        let totalPositions = allEmbeds.dim(1)
        let processed = try prefill.forEachChunk(total: totalPositions) { range in
            _ = languageModel(inputIds[0..., range], cache: cache,
                              inputEmbedding: allEmbeds[0..., range, 0...], mask: mask)
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        let result = languageModel(
            inputIds[0..., processed...], cache: cache,
            inputEmbedding: allEmbeds[0..., processed..., 0...], mask: mask)
        prefill.progress?(totalPositions, totalPositions)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        // TODO evaluate a step in the language model
        languageModel(inputs, cache: cache).logits
    }
}
```

## Create the UserInputProcessor

VLMs require custom `UserInputProcessor` instances to manipulate the prompts and
media as needed. For example it might:

- apply resampling and normalization to the images
- convert the images into an `MLXArray` and build a `THW` struct describing the layout
- modify the prompt by injecting `<image>` tokens that the model expects

In the python implementations, much of this code typically lives in the `transformers`
package from Hugging Face -- inspection will be required to determine which code
is called and what it does. You can examine the processors in the `Models` directory:
they reference the files and functions that they are based on.

The `UserInputProcessor` is initialized with the `ProcessorConfiguration` (defined above)
and has a prepare method:

```swift
public func prepare(input: UserInput) throws -> LMInput
```

This is a slight paraphrase of the `PaligemmaUserInputProcessor` as an example:

```swift
public struct YourModelProcessor: UserInputProcessor {

    private let config: YourModelProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: YourModelProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func prepare(image: CIImage, processing: UserInput.Processing?) -> MLXArray {
        // based on image_processing_siglip from transformers
        var image = image

        // we want to do all of the image processing in an sRGB tone curve
        // rather than a linear space as that is what transformers / torch_vision
        // do (implicitly by using sRGB rasters directly)
        image = MediaProcessing.inSRGBToneCurveSpace(image)

        // apply user instructions
        image = MediaProcessing.apply(image, processing: processing)

        image = MediaProcessing.resampleBicubic(image, to: config.size.cgSize)
        image = MediaProcessing.normalize(
            image, mean: config.imageMeanTuple, std: config.imageStdTuple)

        return MediaProcessing.asMLXArray(image)
    }

    public func prepare(input: UserInput) throws -> LMInput {
        switch input.images.count {
        case 0: throw VLMError.imageRequired
        case 1: break
        default: throw VLMError.singleImageAllowed
        }

        // this doesn't have a chat template so just use the last message.
        var prompt = input.prompt.asMessages().last?["content"] ?? ""

        // based on transformers/processing_paligemma
        let count = input.images.count * config.imageSequenceLength
        prompt =
            Array(repeating: "<image>", count: count).joined() + (tokenizer.bosToken ?? "") + prompt
            + "\n"

        let promptTokens = try tokenizer.encode(text: prompt)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray)

        let pixels = try prepare(image: input.images[0].asCIImage(), processing: input.processing)

        return LMInput(text: .init(tokens: promptArray, mask: mask), image: .init(pixels: pixels))
    }

}
```

Note that the python code may rely on the chat template to inject the image tokens
(paligemma does not). This may have to be expressed in swift code as the current
interface does not support the structured parameters used for this (see Qwen2VL 
processor for an example).

## Register the Model

In [VLMModelFactory.swift](VLMModelFactory.swift) register the model type itself
(this is independent of the model id):

```swift
public class VLMTypeRegistry: @unchecked Sendable {
...
    private var creators: [String: @Sendable (URL) throws -> any LanguageModel] = [
        "yourModel": create(YourModelConfiguration.self, YourModel.init),
```

Similarly, register the UserInputProcessor type (`preprocessor_config.json`):

```swift
public class VLMProcessorTypeRegistry: @unchecked Sendable {
...
    private var creators:
        [String: @Sendable (URL, any Tokenizer) throws -> any UserInputProcessor] = [
            "YourModelProcessor": create(
                YourModelProcessorConfiguration.self, YourModelProcessor.init),
```

### Processor metadata supplied by an external package

Some repositories omit processor metadata, or declare a generic processor that is not the
right implementation for their model type. An external model package can handle either case
without changing `VLMModelFactory`:

```swift
struct YourModelProcessorLoadingResolver: VLMProcessorLoadingResolver {
    func fallbackProcessorConfiguration(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? {
        guard context.modelType == "your_model" else { return nil }

        let model = try JSONDecoder().decode(
            YourModelConfiguration.self, from: context.configurationData)
        let processor = YourModelProcessorConfiguration(model: model)
        return VLMProcessorConfiguration(
            data: try JSONEncoder().encode(processor),
            processorType: "YourModelProcessor")
    }

    func processorType(
        for context: VLMProcessorLoadingContext,
        declaredProcessorType: String?
    ) throws -> String? {
        context.modelType == "your_model" ? "YourModelProcessor" : nil
    }
}

VLMProcessorLoadingRegistry.shared.register(YourModelProcessorLoadingResolver())
```

Implement only the hook your model needs. A checkpoint's `preprocessor_config.json` or
`processor_config.json` always wins over a generated fallback. Processor type resolution is
applied afterward and can supply a missing `processor_class` or correct an incorrect one.
For isolated applications and tests, pass a separate
`VLMProcessorLoadingRegistry` to `VLMModelFactory` instead of registering globally.

Add a constant for the model in the VLMRegistry (not strictly required but useful
for callers to refer to it in code):

```swift
public class VLMRegistry: @unchecked Sendable {
...
    static public let yourModel_4bit = ModelConfiguration(
        id: "mlx-community/YourModel-4bit",
        defaultPrompt: "Describe the image in English"
    )
```

and finally add it to the all list -- this will let users find the model
configuration by id:

```swift
    private static func all() -> [ModelConfiguration] {
        [
            paligemma3bMix4488bit,
...
            yourModel_4bit,
```

# Using a Model

See [MLXLMCommon/README.md](../MLXLMCommon/README.md#using-a-model).

## LFM3-Omni

`LiquidAI/LFM3-Omni-2B` accepts text, images, timestamped video and 16 kHz audio.
It can produce text, 24 kHz speech, or interleaved text and speech. Text responses
use `VLMModelFactory` and `ModelContainer` as usual, including tool-call handling.
Speech responses use `LFM3OmniSession`, which owns the model and its bundled
LiquidCodec weights. The model does not support thinking mode.
The session decodes speech with float32 codec weights and resets the codec at
each end-of-audio marker. MLX's reduced-precision TF32 arithmetic can still cause
differences between streaming and batched decoding; use `MLX_ENABLE_TF32=0` when
comparing numerical results with the full-precision reference.

Download the complete checkpoint, including `audio_tokenizer/config.json` and
`audio_tokenizer/model.safetensors`. The original BF16 checkpoint is supported;
no conversion or tokenizer code from the checkpoint runs in the Swift loader.
Use a tokenizer with the fixes in swift-transformers 1.4.0 or newer. Loading
validates the tokenizer's LFM3 behavior and modality token IDs.

```swift
import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers

let loader: any TokenizerLoader = #huggingFaceTokenizerLoader()
let context = try await VLMModelFactory.shared.load(
    from: URL(fileURLWithPath: "/path/to/LFM3-Omni-2B"), using: loader)
let session = try LFM3OmniSession(context: context)
let response = try await session.respond(
    to: UserInput(chat: [
        .system("Perform TTS in American English using a female voice."),
        .user("The weather is sunny today."),
    ]), parameters: .init(mode: .audio, seed: 42)
) { event in
    switch event {
    case .text(let fragment): print(fragment, terminator: "")
    case .audio(let samples):
        // Feed these mono Float samples to a 24 kHz audio player.
        print("Received \(samples.count) samples")
    }
}
// response.audio contains the complete waveform; response.sampleRate is 24000.
```

Use `.interleaved` with the system prompt `Respond with interleaved text and audio.`
for voice chat. `.automatic` allows the model to choose the response modality;
`.text` forces text output. `maxSteps` counts both text tokens and speech frames.
The optional seed controls codebook reveal order and sampling. Cancellation
throws `CancellationError`; each response has independent language and codec caches.

For subsequent turns, append `response.content` as the assistant message and pass
its `response.audioCodes` through `audioOutputCodes`. Concatenate codebook rows in
message order when retaining multiple spoken turns. Keep the modality markers
and terminal codebook rows: the model needs them to reconstruct speech embeddings.
`session.encodeAudio` accepts mono 24 kHz samples and creates codebook rows for
recorded speech; `session.decodeAudio` reconstructs a waveform from those rows.

Audio URLs are decoded and resampled to mono 16 kHz. For `.array` audio, set
`input.processing.audio.sampleRate = 16000` and supply mono samples already at that
rate. Input audio can accompany images or video. To include a video's soundtrack,
pass its URL as both video and audio and enable timestamp-based interleaving:

```swift
let input = UserInput(
    prompt: "Describe what is happening and what is being said.",
    videos: [.url(videoURL)], audios: [.url(videoURL)],
    additionalContext: ["use_audio_in_video": true])
let response = try await session.respond(to: input, parameters: .init(mode: .text))
```

When there are multiple videos, each video pairs with the next audio placeholder
in the conversation. Standalone audio clips remain separate. Explicit frame inputs
must carry their real timestamps so soundtrack tokens align with frame groups.

### Omni validation

Unit tests generate their configurations and weights in memory. Real-checkpoint
and PyTorch comparison tests are in `IntegrationTesting/IntegrationTestingTests/LFM3OmniIntegrationTests.swift`.
The reference generator is `tools/lfm3-omni-checkpoint-reference.py`; it executes
the downloaded checkpoint's Python modules, so use a trusted snapshot. The tested
upstream snapshot is `21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`.

```sh
python tools/lfm3-omni-checkpoint-reference.py /path/to/checkpoint /tmp/codec.safetensors --component codec
python tools/lfm3-omni-checkpoint-reference.py /path/to/checkpoint /tmp/audio.safetensors --component audio
python tools/lfm3-omni-checkpoint-reference.py /path/to/checkpoint /tmp/depth.safetensors --component depth

TEST_RUNNER_MLX_ENABLE_TF32=0 \
TEST_RUNNER_MLX_LFM3_OMNI_CHECKPOINT=/path/to/checkpoint \
TEST_RUNNER_MLX_LFM3_OMNI_CODEC_REFERENCE=/tmp/codec.safetensors \
TEST_RUNNER_MLX_LFM3_OMNI_AUDIO_REFERENCE=/tmp/audio.safetensors \
TEST_RUNNER_MLX_LFM3_OMNI_DEPTH_REFERENCE=/tmp/depth.safetensors \
xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting -destination 'platform=macOS' \
  -skipPackagePluginValidation \
  -only-testing:IntegrationTestingTests/LFM3OmniIntegrationTests
```

`MLX_LFM3_OMNI_AUDIO_SAMPLE` optionally supplies a recording saying “The weather is
sunny today” for transcription and synchronized video/audio checks. Pass it with
the `TEST_RUNNER_` prefix when invoking `xcodebuild`. `MLX_LFM3_OMNI_VIDEO_SAMPLE`
optionally supplies a red video with that phrase as its soundtrack to test the
file loading path. `MLX_LFM3_OMNI_SPEECH_OUTPUT`
optionally saves the speech test's generated samples and codebook rows outside
the repository. Full-precision reference comparisons disable reduced-precision
TF32; ordinary generation also runs with MLX's default precision settings.
