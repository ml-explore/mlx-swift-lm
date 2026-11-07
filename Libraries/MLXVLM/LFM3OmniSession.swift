import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// A text fragment or a mono 24 kHz audio fragment from an Omni response.
public enum LFM3OmniEvent: Sendable {
    case text(String)
    case audio([Float])
}

/// A complete Omni response, including the codes needed to replay spoken history.
public struct LFM3OmniResponse: Sendable {
    public let text: String
    public let audio: [Float]
    public let sampleRate: Int
    /// Assistant content with modality markers, suitable for the next chat prompt.
    public let content: String
    /// One row per speech placeholder in `content`, including terminal frames.
    public let audioCodes: [[Int]]
    public let stopReason: GenerateStopReason
}

/// Serializes access to an LFM3-Omni model and its LiquidCodec audio tokenizer.
///
/// Load a `ModelContext` with `VLMModelFactory`, then transfer it to this session.
/// Ordinary text generation can also use the factory's `ModelContainer` directly.
public final class LFM3OmniSession: Sendable {
    private struct State {
        let context: ModelContext
        let model: LFM3Omni
        let codec: LiquidCodec
    }
    private let state: SerialAccessContainer<State>

    /// Takes ownership of the context and loads the bundled `audio_tokenizer` weights.
    public init(context: consuming ModelContext, codecDirectory: URL? = nil) throws {
        guard let model = context.model as? LFM3Omni else {
            throw VLMError.processing("LFM3OmniSession requires an LFM3-Omni model")
        }
        let directory =
            try codecDirectory
            ?? context.configuration.modelDirectory.appending(path: "audio_tokenizer")
        let config = try JSONDecoder().decode(
            LiquidCodecConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "config.json")))
        try config.validate()
        guard config.codebooks == model.config.codebooks,
            config.levels.reduce(1, *) == model.config.audioEndCode,
            model.config.codebookSize == model.config.audioEndCode + 1
        else { throw VLMError.processing("LiquidCodec codebooks do not match the Omni model") }
        let codec = try LiquidCodec(config)
        // Keep codec arithmetic in float32 across streaming and offline decoding.
        let weights = codec.sanitize(
            weights: try loadArrays(url: directory.appending(path: "model.safetensors"))
                .mapValues { $0.asType(.float32) })
        try codec.update(parameters: .unflattened(weights), verify: .all)
        eval(codec)
        state = SerialAccessContainer(State(context: context, model: model, codec: codec))
    }

    /// Encodes mono 24 kHz samples into speech-history codebook rows.
    public func encodeAudio(_ samples: [Float], sampleRate: Int = 24_000) async throws -> [[Int]] {
        guard sampleRate == 24000 else {
            throw VLMError.processing("LiquidCodec requires 24 kHz audio")
        }
        return try await state.read { state in
            try Task.checkCancellation()
            let codes = try state.codec.encode(MLXArray(samples).reshaped(1, -1))[0].transposed()
            return (0 ..< codes.dim(0)).map { codes[$0].asArray(Int.self) }
        }
    }

    /// Decodes speech-history rows, omitting terminal frames, into mono 24 kHz samples.
    public func decodeAudio(_ codes: [[Int]]) async throws -> [Float] {
        try await state.read { state in
            try Task.checkCancellation()
            let c = state.model.config
            guard
                codes.allSatisfy({
                    $0.count == c.codebooks && $0.allSatisfy { (0 ..< c.codebookSize).contains($0) }
                })
            else {
                throw VLMError.processing("Invalid Omni audio codebook rows")
            }
            guard
                codes.allSatisfy({
                    !$0.contains(c.audioEndCode) || $0.allSatisfy { $0 == c.audioEndCode }
                })
            else {
                throw VLMError.processing("A terminal Omni speech frame must end every codebook")
            }
            var samples = [Float]()
            for frames in codes.split(whereSeparator: { $0.allSatisfy { $0 == c.audioEndCode } }) {
                let array = MLXArray(frames.flatMap { $0 }).reshaped(frames.count, c.codebooks)
                    .transposed().expandedDimensions(axis: 0)
                samples.append(
                    contentsOf: try state.codec.decode(array).reshaped(-1).asArray(Float.self))
            }
            return samples
        }
    }

    /// Generates text and speech, delivering fragments as they become available.
    ///
    /// For spoken history, append earlier responses' `content` as assistant messages and
    /// concatenate their `audioCodes` in message order. Cancellation throws `CancellationError`.
    public func respond(
        to input: consuming sending UserInput,
        audioOutputCodes: [[Int]] = [], parameters: LFM3OmniGenerationParameters = .init(),
        onEvent: @escaping @Sendable (LFM3OmniEvent) -> Void = { _ in }
    ) async throws -> LFM3OmniResponse {
        let input = SendableBox(input)
        return try await state.read { state in
            try Task.checkCancellation()
            let c = state.model.config
            guard audioOutputCodes.allSatisfy({ $0.count == c.codebooks }) else {
                throw VLMError.processing("Omni speech history has an invalid codebook count")
            }
            let history =
                audioOutputCodes.isEmpty
                ? nil
                : MLXArray(audioOutputCodes.flatMap { $0 }).reshaped(-1, c.codebooks)
            let prepared = try await state.context.processor.prepare(input: input.consume())
            var codecState = try state.codec.newStreamingState()
            var text = ""
            var audio = [Float]()
            var detokenizer = state.context.tokenizer.makeStreamingDetokenizer()
            var stop = state.context.configuration.eosTokenIds
            if let eos = state.context.tokenizer.eosTokenId { stop.insert(eos) }
            let markers = Set([c.audioStartToken, c.mixedStartToken, c.textEndToken]).union(stop)
            func emitAudio(_ values: MLXArray) {
                let samples = values.reshaped(-1).asArray(Float.self)
                if !samples.isEmpty {
                    audio.append(contentsOf: samples)
                    onEvent(.audio(samples))
                }
            }
            let result = try state.model.generateOmni(
                input: prepared, audioOutputCodes: history,
                parameters: parameters, stopTokens: stop
            ) { event in
                switch event {
                case .token(let token):
                    if !markers.contains(token) {
                        detokenizer.append(token: token)
                        if let chunk = detokenizer.next(), !chunk.isEmpty {
                            text += chunk
                            onEvent(.text(chunk))
                        }
                    }
                case .audio(let codes):
                    emitAudio(
                        try state.codec.decode(
                            codes.reshaped(1, -1, 1), state: codecState, finalize: false))
                case .audioEnd:
                    emitAudio(
                        try state.codec.decode(
                            MLXArray.zeros([1, c.codebooks, 0], dtype: .int32), state: codecState))
                    codecState = try state.codec.newStreamingState()
                }
            }
            emitAudio(
                try state.codec.decode(
                    MLXArray.zeros([1, c.codebooks, 0], dtype: .int32), state: codecState))
            if let tail = detokenizer.finish(), !tail.isEmpty {
                text += tail
                onEvent(.text(tail))
            }
            let codes = (0 ..< result.codes.dim(0)).map { result.codes[$0].asArray(Int.self) }
            return LFM3OmniResponse(
                text: text, audio: audio, sampleRate: state.codec.config.sampleRate,
                content: state.context.tokenizer.decode(
                    tokenIds: result.tokens.filter { !stop.contains($0) || $0 == c.textEndToken },
                    skipSpecialTokens: false),
                audioCodes: codes, stopReason: result.stopReason)
        }
    }
}
