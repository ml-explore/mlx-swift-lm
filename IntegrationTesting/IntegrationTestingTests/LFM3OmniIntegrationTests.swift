import CoreImage
import CoreMedia
import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest

@testable import MLXVLM

final class LFM3OmniIntegrationTests: XCTestCase {
    private func checkpoint() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["MLX_LFM3_OMNI_CHECKPOINT"] else {
            throw XCTSkip("Set MLX_LFM3_OMNI_CHECKPOINT to the official Omni checkpoint directory")
        }
        return URL(fileURLWithPath: path)
    }

    private func config() throws -> LFM3OmniConfiguration {
        try JSONDecoder().decode(
            LFM3OmniConfiguration.self,
            from: Data(contentsOf: checkpoint().appending(path: "config.json")))
    }

    private func weights(prefix: String) throws -> [String: MLXArray] {
        let all = try loadArrays(url: checkpoint().appending(path: "model.safetensors"))
        return Dictionary(
            uniqueKeysWithValues: all.compactMap { key, value in
                key.hasPrefix(prefix)
                    ? (String(key.dropFirst(prefix.count)), value.asType(.float32)) : nil
            })
    }

    private func reference(_ suffix: String) throws -> [String: MLXArray] {
        guard
            let path = ProcessInfo.processInfo.environment["MLX_LFM3_OMNI_" + suffix + "_REFERENCE"]
        else {
            throw XCTSkip(
                "Set MLX_LFM3_OMNI_" + suffix + "_REFERENCE to the official PyTorch reference")
        }
        return try loadArrays(url: URL(fileURLWithPath: path))
    }

    func testOfficialTextImageAndAudioGeneration() async throws {
        let directory = try checkpoint()
        let container = try await VLMModelFactory.shared.loadContainer(
            from: directory, using: #huggingFaceTokenizerLoader())
        let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: 256, height: 256))
        var cases: [(UserInput, [String])] = [
            (.init(prompt: "What is two plus two? Answer with one digit."), ["4"]),
            (
                .init(
                    prompt: "What color is this image? Answer with one word.",
                    images: [.ciImage(image)]), ["red"]
            ),
        ]
        let frames = (0 ..< 4).map {
            VideoFrame(
                image: .ciImage(image),
                timeStamp: .init(seconds: Double($0) / 2, preferredTimescale: 600))
        }
        var video = UserInput(
            prompt: "What color is shown in this video? Answer with one word.",
            videos: [.frames(frames)])
        video.processing.video = .init(targetFrames: 4)
        cases.append((video, ["red"]))
        if let path = ProcessInfo.processInfo.environment["MLX_LFM3_OMNI_AUDIO_SAMPLE"] {
            cases.append(
                (
                    .init(
                        prompt: "Transcribe the audio.", audios: [.url(URL(fileURLWithPath: path))]),
                    ["weather", "sunny"]
                ))
            var input = UserInput(
                prompt: "What color is shown, and what words are spoken?",
                videos: [.frames(frames)], audios: [.url(URL(fileURLWithPath: path))],
                additionalContext: ["use_audio_in_video": true])
            input.processing.video = .init(targetFrames: 4)
            cases.append((input, ["red", "weather", "sunny"]))
        }
        if let path = ProcessInfo.processInfo.environment["MLX_LFM3_OMNI_VIDEO_SAMPLE"] {
            let url = URL(fileURLWithPath: path)
            var input = UserInput(
                prompt: "What color is shown, and what words are spoken?",
                videos: [.url(url)], audios: [.url(url)],
                additionalContext: ["use_audio_in_video": true])
            input.processing.video = .init(targetFrames: 4)
            cases.append((input, ["red", "weather", "sunny"]))
        }
        for (input, expected) in cases {
            let prepared = try await container.prepare(input: input)
            let stream = try await container.generate(
                input: prepared, parameters: .init(maxTokens: 48, temperature: 0))
            var answer = ""
            for await event in stream {
                if case .chunk(let text) = event { answer += text }
            }
            print("LFM3-Omni answer (\(expected)): \(answer)")
            for word in expected {
                XCTAssertTrue(
                    answer.lowercased().contains(word), "Expected '\(word)' in '\(answer)'")
            }
        }
    }

    func testOfficialInterleavedSessionAndSpeechHistory() async throws {
        let context = try await VLMModelFactory.shared.load(
            from: checkpoint(), using: #huggingFaceTokenizerLoader())
        let session = try LFM3OmniSession(context: context)
        let response = try await session.respond(
            to: .init(chat: [
                .system("Respond with interleaved text and audio."), .user("Say hello."),
            ]), parameters: .init(mode: .interleaved, maxSteps: 128, textTemperature: 0, seed: 42))
        print(
            "LFM3-Omni interleaved text: \(response.text), audio samples: \(response.audio.count)")
        XCTAssertTrue(response.text.lowercased().contains("hello"))
        XCTAssertGreaterThan(response.audio.count, 2400)
        XCTAssertTrue(response.content.contains("<|text_end|>"))
        XCTAssertEqual(response.stopReason, .stop)
        let next = try await session.respond(
            to: .init(chat: [
                .system("Respond with text."), .user("Say hello."),
                .assistant(response.content),
                .user("What word did you just say? Reply with that word only."),
            ]), audioOutputCodes: response.audioCodes,
            parameters: .init(mode: .text, maxSteps: 48, textTemperature: 0))
        print(
            "LFM3-Omni speech history reply: \(next.text), raw: \(next.content), stop: \(next.stopReason)"
        )
        XCTAssertTrue(next.text.lowercased().contains("hello"))
        XCTAssertTrue(next.audio.isEmpty)
        let decoded = try await session.decodeAudio(response.audioCodes)
        XCTAssertEqual(decoded.count, response.audio.count)
        XCTAssertTrue(decoded.allSatisfy(\.isFinite))
        let residual = MLXArray(decoded) - MLXArray(response.audio)
        let replayError = abs(residual).max().item(Float.self)
        print("Session codec replay max error: \(replayError)")
        // TF32 uses different rounding for single-frame and batched matrix products.
        if ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "0" {
            XCTAssertLessThan(replayError, 0.00001)
        }
        let encoded = try await session.encodeAudio(Array(response.audio.prefix(1920)))
        XCTAssertEqual(encoded.count, 1)
        XCTAssertEqual(encoded.first?.count, 12)
        XCTAssertTrue(encoded.flatMap { $0 }.allSatisfy { (0 ..< 2016).contains($0) })
        let twice = try await session.decodeAudio(response.audioCodes + response.audioCodes)
        XCTAssertEqual(twice.count, 2 * decoded.count)
        XCTAssertEqual(Array(twice.prefix(decoded.count)), decoded)
        XCTAssertEqual(Array(twice.suffix(decoded.count)), decoded)
        let cancelled = Task {
            try await session.respond(to: .init(prompt: "Hello"))
        }
        cancelled.cancel()
        do {
            _ = try await cancelled.value
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }

    }

    func testOfficialToolCalling() async throws {
        let container = try await VLMModelFactory.shared.loadContainer(
            from: checkpoint(), using: #huggingFaceTokenizerLoader())
        let configuration = await container.configuration
        XCTAssertEqual(configuration.toolCallFormat, .xmlFunction)
        try await ToolCallTests.lfm2EndToEndGeneration(container: container)
    }

    func testOfficialAudioToSpeech() async throws {
        guard let path = ProcessInfo.processInfo.environment["MLX_LFM3_OMNI_AUDIO_SAMPLE"] else {
            throw XCTSkip("Set MLX_LFM3_OMNI_AUDIO_SAMPLE to the weather recording")
        }
        let context = try await VLMModelFactory.shared.load(
            from: checkpoint(), using: #huggingFaceTokenizerLoader())
        let session = try LFM3OmniSession(context: context)
        let response = try await session.respond(
            to: .init(chat: [
                .system("Respond with interleaved text and audio."),
                .user(
                    "What is the weather in this recording?",
                    audios: [.url(URL(fileURLWithPath: path))]),
            ]), parameters: .init(mode: .interleaved, maxSteps: 160, textTemperature: 0, seed: 42))
        print("LFM3-Omni audio-to-speech: \(response.text), samples: \(response.audio.count)")
        XCTAssertTrue(response.text.lowercased().contains("sunny"))
        XCTAssertGreaterThan(response.audio.count, 2400)
        XCTAssertTrue(response.audio.allSatisfy(\.isFinite))
        XCTAssertEqual(response.stopReason, .stop)
    }

    func testOfficialSpeechGeneration() async throws {
        let context = try await VLMModelFactory.shared.load(
            from: checkpoint(), using: #huggingFaceTokenizerLoader())
        let session = try LFM3OmniSession(context: context)
        let response = try await session.respond(
            to: .init(chat: [
                .system("Perform TTS in American English using a female voice."),
                .user("The weather is sunny today."),
            ]), parameters: .init(mode: .audio, maxSteps: 128, textTemperature: 0, seed: 42))
        let samples = MLXArray(response.audio).reshaped(1, -1)
        print(
            "LFM3-Omni speech frames: \(response.audioCodes.count), samples: \(samples.size), stop: \(response.stopReason)"
        )
        XCTAssertGreaterThan(samples.size, 2400)
        XCTAssertTrue(all(isFinite(samples)).item(Bool.self))
        XCTAssertGreaterThan(abs(samples).max().item(Float.self), 0.01)
        XCTAssertEqual(response.sampleRate, 24000)
        XCTAssertEqual(response.stopReason, .stop)
        if let path = ProcessInfo.processInfo.environment["MLX_LFM3_OMNI_SPEECH_OUTPUT"] {
            let codes = MLXArray(response.audioCodes.flatMap { $0 }).reshaped(-1, 12)
            try MLX.save(
                arrays: ["samples": samples, "codes": codes], url: URL(fileURLWithPath: path))
        }
    }

    func testOfficialAudioEncoderAndProjector() throws {
        let c = try config()
        let encoder = LFM3OmniAudioEncoder(c.audioEncoder)
        try encoder.update(
            parameters: .unflattened(
                encoder.sanitize(weights: weights(prefix: "model.audio_encoder."))), verify: .all)
        let projector = LFM3OmniAudioProjector(
            inputSize: c.audioEncoder.hiddenSize, outputSize: c.visionLanguage.text.hiddenSize)
        try projector.update(
            parameters: .unflattened(weights(prefix: "model.audio_projector.")), verify: .all)
        let reference = try reference("AUDIO")
        let encoded = try encoder(
            XCTUnwrap(reference["features"]), lengths: XCTUnwrap(reference["lengths"]))
        let projected = projector(encoded)
        let features = try XCTUnwrap(reference["features"])[0 ..< 1, 0..., 0...]
        for chunkSize in [1, 7, 8, 31, 80] {
            let state = try encoder.newStreamingState()
            var chunks = [MLXArray]()
            for offset in stride(from: 0, to: features.dim(2), by: chunkSize) {
                chunks.append(
                    try encoder.stream(
                        features[0..., 0..., offset ..< min(offset + chunkSize, features.dim(2))],
                        state: state))
            }
            chunks.append(
                try encoder.stream(MLXArray.zeros([1, 128, 0]), state: state, finalize: true))
            let streamed = concatenated(chunks, axis: 1)[0]
            let expected = try XCTUnwrap(reference["encoded"])[..<43]
            XCTAssertEqual(streamed.shape, expected.shape)
            let error = abs(streamed - expected).max().item(Float.self)
            print("LFM3-Omni streamed encoder chunk \(chunkSize) error: \(error)")
            XCTAssertTrue(allClose(streamed, expected, rtol: 0.0001, atol: 0.0001).item(Bool.self))
            XCTAssertThrowsError(try encoder.stream(features, state: state))
        }
        for (name, actual) in [("encoded", encoded), ("projected", projected)] {
            let expected = try XCTUnwrap(reference[name])
            XCTAssertEqual(actual.shape, expected.shape)
            let error = abs(actual - expected).max().item(Float.self)
            print("LFM3-Omni \(name) max error: \(error)")
            XCTAssertTrue(allClose(actual, expected, rtol: 0.0001, atol: 0.0001).item(Bool.self))
        }
    }

    func testOfficialDepthDecoder() throws {
        let decoder = LFM3OmniDepthDecoder(try config())
        try decoder.update(
            parameters: .unflattened(weights(prefix: "model.depth_decoder.")), verify: .all)
        let reference = try reference("DEPTH")
        let logits = try decoder(
            hidden: XCTUnwrap(reference["hidden"]), codes: XCTUnwrap(reference["codes"]),
            masked: XCTUnwrap(reference["masked"]))
        let expected = try XCTUnwrap(reference["logits"])
        XCTAssertEqual(logits.shape, expected.shape)
        print("LFM3-Omni depth logits max error: \(abs(logits - expected).max().item(Float.self))")
        XCTAssertTrue(allClose(logits, expected, rtol: 0.0001, atol: 0.0001).item(Bool.self))
    }

    func testOfficialCodecReferenceAndStreaming() throws {
        let directory = try checkpoint().appending(path: "audio_tokenizer")
        let config = try JSONDecoder().decode(
            LiquidCodecConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "config.json")))
        let model = try LiquidCodec(config)
        let weights = model.sanitize(
            weights: try loadArrays(url: directory.appending(path: "model.safetensors")))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.apply { $0.asType(.float32) }
        eval(model)
        guard let path = ProcessInfo.processInfo.environment["MLX_LFM3_OMNI_CODEC_REFERENCE"] else {
            throw XCTSkip("Set MLX_LFM3_OMNI_CODEC_REFERENCE to the PyTorch codec reference")
        }
        let reference = try loadArrays(url: URL(fileURLWithPath: path))
        let samples = try XCTUnwrap(reference["samples"])
        let codes = try XCTUnwrap(reference["codes"])
        let expected = try XCTUnwrap(reference["audio"])
        let encoded = try model.encode(samples)
        print("LiquidCodec codes: \(encoded.asArray(Int.self))")
        XCTAssertTrue(all(encoded .== codes).item(Bool.self))
        let audio = try model.decode(codes)
        XCTAssertEqual(audio.shape, expected.shape)
        let error = abs(audio - expected).max().item(Float.self)
        print("LiquidCodec PyTorch max waveform error: \(error)")
        XCTAssertLessThan(error, 0.00001)
        for chunkSize in [1, 2, 3] {
            let state = try model.newStreamingState()
            var chunks = [MLXArray]()
            for offset in stride(from: 0, to: codes.dim(2), by: chunkSize) {
                chunks.append(
                    try model.decode(
                        codes[0..., 0..., offset ..< min(offset + chunkSize, codes.dim(2))],
                        state: state, finalize: false))
            }
            chunks.append(
                try model.decode(
                    MLXArray.zeros([codes.dim(0), config.codebooks, 0], dtype: .int32),
                    state: state))
            let streamed = concatenated(chunks, axis: 1)
            XCTAssertEqual(streamed.shape, audio.shape)
            let streamingError = abs(streamed - audio).max().item(Float.self)
            print("LiquidCodec streaming chunk \(chunkSize) error: \(streamingError)")
            XCTAssertLessThan(streamingError, 0.00001)
            XCTAssertThrowsError(try model.decode(codes, state: state))
        }
    }
}
