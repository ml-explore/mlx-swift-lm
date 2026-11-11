import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class LFM3OmniComponentTests: XCTestCase {
    private func model() throws -> LFM3Omni {
        let json = """
            {"text_config": {"model_type":"lfm3", "vocab_size":32,
              "hidden_size":16, "num_hidden_layers":2, "num_attention_heads":2,
              "num_key_value_heads":1, "head_dim":8, "intermediate_size":24,
              "layer_types":["conv","full_attention"]},
             "vision_config": {"hidden_size":16, "intermediate_size":24,
              "num_attention_heads":2, "num_key_value_heads":1, "patch_size":2,
              "temporal_patch_size":2, "spatial_merge_size":2, "in_channels":3,
              "conv_kernel_size":3, "norm_eps":0.00001, "num_hidden_layers":2,
              "layer_types":["conv","full_attention"],
              "rope_parameters":{"rope_type":"axial","rope_theta":1000000}},
             "image_token_id":10, "video_token_id":11, "downsample_factor":2,
             "projector_hidden_size":64, "projector_bias":true,
             "projector_layer_norm_eps":0.00001, "projector_hidden_act":"gelu"}
            """
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        object["audio_encoder_config"] = [
            "conv_channels": 4, "hidden_size": 16, "num_mel_bins": 16, "sliding_window": 4,
            "text_config": [
                "model_type": "lfm2", "vocab_size": 32, "hidden_size": 16,
                "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                "layer_types": ["conv", "full_attention"], "norm_eps": 0.00001,
                "block_ff_dim": 24, "block_auto_adjust_ff_dim": false,
            ] as [String: Any],
        ]
        object["depth_config"] =
            [
                "model_type": "lfm3", "vocab_size": 1, "hidden_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 8, "intermediate_size": 24, "layer_types": ["full_attention"],
            ] as [String: Any]
        for (key, value) in [
            "audio_token_id": 12, "audio_out_token_id": 13, "audio_start_token_id": 14,
            "mixed_start_token_id": 15, "text_end_token_id": 16, "audio_eos_code": 6,
            "codebook_size": 7, "num_codebooks": 2, "interleaved_n_text": 2,
            "interleaved_n_audio": 2,
        ] { object[key] = value }
        object["maskgit_schedule"] = [1, 1]
        let config = try JSONDecoder().decode(
            LFM3OmniConfiguration.self, from: JSONSerialization.data(withJSONObject: object))
        let codec = LiquidCodecConfiguration(
            hiddenSize: 16, channels: 4, fftSize: 16, hopSize: 4, windowSize: 16,
            frameStack: 4, quantizerSize: 8, levels: [2, 3], codebooks: 2,
            decoderWindow: 4, sampleRate: 24000)
        return try LFM3Omni(config, codecConfiguration: codec)
    }

    private func checkpoint(convertedCodec: Bool = false) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(
            at: directory.appending(path: "audio_tokenizer"), withIntermediateDirectories: true)
        let source = try model()
        var main = [String: MLXArray]()
        var codec = [String: MLXArray]()
        for (index, entry) in source.parameters().flattened().sorted(by: { $0.0 < $1.0 })
            .enumerated()
        {
            let (name, value) = entry
            let weight =
                name == "audio_tokenizer.decoder.istft.window"
                ? value
                : (sin(MLXArray(0 ..< value.size).asType(.float32) * 0.13 + Float(index)) * 0.1)
                    .reshaped(value.shape)
            if name.hasPrefix("audio_tokenizer.") {
                let key = String(name.dropFirst("audio_tokenizer.".count))
                if convertedCodec {
                    codec[name] = weight
                } else {
                    let transpose =
                        key.hasSuffix(".conv.weight")
                        || (key.hasPrefix("frontend.") && weight.ndim == 3)
                    codec[key] = transpose ? weight.transposed(0, 2, 1) : weight
                }
            } else {
                main[name] = weight
            }
        }
        try MLX.save(arrays: main, url: directory.appending(path: "model.safetensors"))
        try MLX.save(
            arrays: codec, url: directory.appending(path: "audio_tokenizer/model.safetensors"))
        return directory
    }

    private func hidden(_ model: LFM3Omni, component: ModelComponent? = nil) throws -> MLXArray {
        let input: LMInput
        switch component {
        case .vision:
            input = LMInput(
                text: .init(tokens: MLXArray([1, 10, 2])),
                image: .init(pixels: MLXArray.ones([4, 24]), frames: [THW(1, 2, 2)]))
        case .audio:
            input = LMInput(
                text: .init(tokens: MLXArray([1, 12, 12, 2])),
                audio: .init(features: MLXArray.ones([1, 16, 9]), mask: MLXArray.ones([1, 9])))
        default:
            input = LMInput(text: .init(tokens: MLXArray([1, 2, 3])))
        }
        let result = try model.embeddings(input)
        eval(result)
        return result
    }

    func testComponentsLoadIndependentlyAndMatchImmediateLoading() throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let lazy = try model()
        let eager = try model()
        try loadWeights(modelDirectory: directory, model: lazy)
        try loadWeights(modelDirectory: directory, model: eager, componentLoading: .immediate)
        let tree = lazy.namedModules().map(\.0)
        XCTAssertTrue(lazy.onDemandComponents.allSatisfy { !$0.isLoaded })
        XCTAssertTrue(allClose(try hidden(lazy), try hidden(eager)).item(Bool.self))
        XCTAssertTrue(try lazy.memory().components.isEmpty)
        for component in [ModelComponent.vision, .audio] {
            XCTAssertTrue(
                allClose(
                    try hidden(lazy, component: component), try hidden(eager, component: component)
                ).item(Bool.self))
            XCTAssertTrue(lazy.onDemandComponents.first { $0.component == component }!.isLoaded)
            XCTAssertFalse(lazy.onDemandComponents.first { $0.component == .speech }!.isLoaded)
        }
        let holder = try XCTUnwrap(lazy.onDemandComponents.first { $0.component == .speech })
        final class Loads: @unchecked Sendable {
            let lock = NSLock()
            var identities = Set<ObjectIdentifier>()
            var errors = [String]()
        }
        let loads = Loads()
        DispatchQueue.concurrentPerform(iterations: 8) { _ in
            do {
                let identity = ObjectIdentifier(try holder.loadModule())
                loads.lock.withLock { _ = loads.identities.insert(identity) }
            } catch {
                loads.lock.withLock { loads.errors.append(String(describing: error)) }
            }
        }
        XCTAssertTrue(loads.errors.isEmpty, loads.errors.description)
        XCTAssertEqual(loads.identities.count, 1)
        let speech = try lazy.speech()
        let codes = MLXArray.zeros([1, 2, 2], dtype: .int32)
        XCTAssertTrue(
            allClose(
                try speech.codec.decode(codes), try eager.speech().codec.decode(codes), rtol: 0.001,
                atol: 0.0001
            ).item(Bool.self))
        let memory = try lazy.memory()
        XCTAssertEqual(Set(memory.components.map(\.kind)), [.vision, .audio, .speech])
        XCTAssertTrue(memory.components.allSatisfy { $0.bytes > 0 })
        XCTAssertEqual(lazy.namedModules().map(\.0), tree)
    }

    func testExcludedComponentsSupportTextAndRejectTheirInputs() throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try model()
        try FileManager.default.removeItem(
            at: directory.appending(path: "audio_tokenizer/model.safetensors"))
        try loadWeights(
            modelDirectory: directory, model: model, excludedComponents: [.vision, .audio, .speech])
        XCTAssertNoThrow(try hidden(model))
        for component in [ModelComponent.vision, .audio, .speech] {
            XCTAssertThrowsError(
                try component == .speech
                    ? model.speech().embedding(MLXArray([0])) : hidden(model, component: component)
            ) { error in
                guard case OnDemandComponentError.unavailable(component) = error else {
                    return XCTFail("Unexpected error: \(error)")
                }
            }
        }
        let processor = TestInputProcessor()
        let context = ModelContext(
            configuration: .init(id: "test"), model: model,
            processor: processor, tokenizer: processor.tokenizer)
        XCTAssertThrowsError(try LFM3OmniSession(context: context)) { error in
            guard case OnDemandComponentError.unavailable(.speech) = error else {
                return XCTFail("Unexpected session error: \(error)")
            }
        }
        XCTAssertTrue(model.onDemandComponents.allSatisfy { !$0.isLoaded })
    }

    func testConvertedCodecLayoutMatchesTheOriginalCheckpoint() throws {
        let raw = try checkpoint()
        let converted = try checkpoint(convertedCodec: true)
        defer {
            try? FileManager.default.removeItem(at: raw)
            try? FileManager.default.removeItem(at: converted)
        }
        let originalModel = try model()
        let convertedModel = try model()
        try loadWeights(modelDirectory: raw, model: originalModel)
        try loadWeights(modelDirectory: converted, model: convertedModel)
        let codes = MLXArray.zeros([1, 2, 2], dtype: .int32)
        XCTAssertTrue(
            allClose(
                try originalModel.speech().codec.decode(codes),
                try convertedModel.speech().codec.decode(codes), rtol: 0.001, atol: 0.0001
            ).item(Bool.self))
    }

    func testCodecRenamingPreservesPackedWeightsAndLayerPrecision() throws {
        let model = try model()
        let checkpoint = ModelCheckpoint(
            weights: ["to_quantizer.weight": MLXArray.zeros([8, 2], dtype: .uint32)],
            perLayerQuantization: .init(
                quantization: .init(groupSize: 64, bits: 4),
                perLayerQuantization: ["to_quantizer": .skip]))
        let prepared = try model.prepareCheckpoint(checkpoint)
        XCTAssertEqual(prepared.weights["audio_tokenizer.to_quantizer.weight"]?.dtype, .uint32)
        guard
            case .skip = prepared.perLayerQuantization?.perLayerQuantization[
                "audio_tokenizer.to_quantizer"]
        else {
            return XCTFail("Codec precision must follow its renamed module")
        }
    }

    func testSessionValidatesTheBundledCodecConfiguration() throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try model()
        try loadWeights(modelDirectory: directory, model: model)
        let c = try model.speech().codec.config
        var json: [String: Any] = [
            "encoder_dim": c.hiddenSize, "encoder_conv_dim": c.channels, "stft_n_fft": c.fftSize,
            "stft_hop_length": c.hopSize, "stft_win_length": c.windowSize,
            "stft_frame_stack": c.frameStack, "quantizer_dim": c.quantizerSize,
            "fsq_levels": c.levels, "n_q": c.codebooks, "decoder_window": c.decoderWindow,
            "sampling_rate": c.sampleRate,
        ]
        let url = directory.appending(path: "audio_tokenizer/config.json")
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        func context() -> ModelContext {
            let processor = TestInputProcessor()
            return ModelContext(
                configuration: .init(directory: directory), model: model,
                processor: processor, tokenizer: processor.tokenizer)
        }
        XCTAssertNoThrow(try LFM3OmniSession(context: context()))
        json["stft_hop_length"] = c.hopSize + 1
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        XCTAssertThrowsError(try LFM3OmniSession(context: context())) { error in
            XCTAssertTrue(
                error.localizedDescription.contains("Unsupported LiquidCodec configuration"))
        }
    }

    func testTextAndAudioDoNotReadMissingSpeechWeightsAndFailedLoadCanRetry() throws {
        let directory = try checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let mainURL = directory.appending(path: "model.safetensors")
        let weights = try MLX.loadArrays(url: mainURL)
        eval(Array(weights.values))
        try MLX.save(
            arrays: weights.filter {
                !$0.key.hasPrefix("depth_decoder.") && !$0.key.hasPrefix("codebook_embedding.")
            }, url: mainURL)
        let model = try model()
        try loadWeights(modelDirectory: directory, model: model)
        _ = try hidden(model, component: .audio)
        XCTAssertThrowsError(try model.speech())
        XCTAssertFalse(model.onDemandComponents.first { $0.component == .speech }!.isLoaded)
        try MLX.save(arrays: weights, url: mainURL)
        XCTAssertNoThrow(try model.speech())
    }
}
