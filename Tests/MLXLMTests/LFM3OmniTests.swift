import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

final class LFM3OmniTests: XCTestCase {
    private func audioConfig() throws -> LFM3OmniAudioConfiguration {
        try JSONDecoder().decode(
            LFM3OmniAudioConfiguration.self,
            from: Data(
                """
                {"conv_channels":4,"hidden_size":16,"num_mel_bins":16,"sliding_window":4,
                 "text_config":{"model_type":"lfm2","vocab_size":32,"hidden_size":16,
                 "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
                 "norm_eps":0.00001,"layer_types":["conv","full_attention"],
                 "block_ff_dim":24,"block_auto_adjust_ff_dim":false}}
                """.utf8))
    }

    private func codecConfig() throws -> LiquidCodecConfiguration {
        try JSONDecoder().decode(
            LiquidCodecConfiguration.self,
            from: Data(
                """
                {"encoder_dim":16,"encoder_conv_dim":4,"stft_n_fft":16,"stft_hop_length":4,
                 "stft_win_length":16,"stft_frame_stack":4,"quantizer_dim":8,"fsq_levels":[2,3],
                 "n_q":2,"decoder_window":4,"sampling_rate":24000}
                """.utf8))
    }

    private func initialize(_ model: Module) throws {
        let entries = model.parameters().flattened().sorted { $0.0 < $1.0 }
        var weights = [String: MLXArray]()
        for (index, entry) in entries.enumerated() {
            let (key, original) = entry
            if key == "decoder.istft.window" { continue }
            weights[key] =
                (sin(MLXArray(0 ..< original.size).asType(.float32) * 0.13 + Float(index)) * 0.1)
                .reshaped(original.shape)
        }
        try model.update(
            parameters: .unflattened(weights), verify: [.noUnusedKeys, .shapeMismatch])
        eval(model)
    }

    func testAudioPaddingDoesNotChangeValidTokens() throws {
        let encoder = LFM3OmniAudioEncoder(try audioConfig())
        try initialize(encoder)
        let features = sin(MLXArray(0 ..< 2 * 16 * 49).asType(.float32) * 0.017).reshaped(2, 16, 49)
        let batch = try encoder(features, lengths: MLXArray([49, 17]))
        let alone = try encoder(features[1 ..< 2, 0..., ..<17], lengths: MLXArray([17]))
        XCTAssertEqual(batch.shape, [10, 16])
        XCTAssertTrue(allClose(batch[7...], alone, rtol: 1e-3, atol: 1e-4).item(Bool.self))
    }

    func testEncoderStreamingAndStateValidation() throws {
        let encoder = LFM3OmniAudioEncoder(try audioConfig())
        try initialize(encoder)
        let features = sin(MLXArray(0 ..< 16 * 49).asType(.float32) * 0.017).reshaped(1, 16, 49)
        let expected = try encoder(features, lengths: MLXArray([49]))
        let state = try encoder.newStreamingState()
        XCTAssertThrowsError(try encoder.stream(MLXArray.zeros([1, 15, 8]), state: state))
        var chunks = [MLXArray]()
        for index in stride(from: 0, to: 49, by: 3) {
            chunks.append(
                try encoder.stream(features[0..., 0..., index ..< min(index + 3, 49)], state: state)
            )
        }
        chunks.append(try encoder.stream(MLXArray.zeros([1, 16, 0]), state: state, finalize: true))
        let result = concatenated(chunks, axis: 1)[0]
        print("Tiny encoder stream error \(abs(result - expected).max().item(Float.self))")
        XCTAssertTrue(allClose(result, expected, rtol: 1e-3, atol: 1e-4).item(Bool.self))
        XCTAssertThrowsError(try encoder.stream(features, state: state))
        let other = LFM3OmniAudioEncoder(try audioConfig())
        XCTAssertThrowsError(try other.stream(features, state: encoder.newStreamingState()))
    }

    func testAudioRejectsInvalidFeatureLengths() throws {
        let encoder = LFM3OmniAudioEncoder(try audioConfig())
        let features = MLXArray.zeros([1, 16, 8])
        for lengths in [MLXArray([0]), MLXArray([9]), MLXArray([Float(3.5)]), MLXArray([1, 2])] {
            XCTAssertThrowsError(try encoder(features, lengths: lengths))
        }
        XCTAssertThrowsError(try encoder(features + Float.nan, lengths: MLXArray([8])))
    }

    func testCodecStreamingOverlapAndTerminalState() throws {
        let codec = try LiquidCodec(codecConfig())
        try initialize(codec)
        let codes = MLXArray([0, 1, 2, 3, 4, 5, 2, 1]).reshaped(1, 2, 4)
        let expected = try codec.decode(codes)
        XCTAssertEqual(expected.shape, [1, 64])
        let state = try codec.newStreamingState()
        var chunks = [MLXArray]()
        for index in 0 ..< 4 {
            chunks.append(
                try codec.decode(
                    codes[0..., 0..., index ..< index + 1], state: state, finalize: false))
        }
        chunks.append(try codec.decode(MLXArray.zeros([1, 2, 0], dtype: .int32), state: state))
        let streamed = concatenated(chunks, axis: 1)
        print("Tiny codec stream error \(abs(streamed - expected).max().item(Float.self))")
        XCTAssertTrue(allClose(streamed, expected, rtol: 1e-3, atol: 1e-4).item(Bool.self))
        XCTAssertThrowsError(try codec.decode(codes, state: state))
        XCTAssertThrowsError(try codec.decode(codes + 6))
        XCTAssertThrowsError(try codec.decode(codes.asType(.float32)))
        let encoded = try codec.encode(MLXArray.zeros([1, 29]))
        XCTAssertEqual(encoded.shape, [1, 2, 2])
        XCTAssertTrue(all((encoded .>= 0) .&& (encoded .< 6)).item(Bool.self))
        XCTAssertFalse(codec.parameters().flattened().contains { $0.0 == "quantizer.levels" })
    }

    func testCodecInvalidatesStateAfterNonFiniteOutput() throws {
        let codec = try LiquidCodec(codecConfig())
        try initialize(codec)
        let codes = MLXArray.zeros([1, 2, 1], dtype: .int32)
        let state = try codec.newStreamingState()
        let weights = codec.parameters().flattened()
        let bias = try XCTUnwrap(weights.first { $0.0 == "decoder.lin.bias" })
        try codec.update(
            parameters: .unflattened([
                bias.0: MLXArray.full(bias.1.shape, values: MLXArray(Float.nan))
            ]))
        XCTAssertThrowsError(try codec.decode(codes, state: state, finalize: false))
        try initialize(codec)
        XCTAssertThrowsError(try codec.decode(codes, state: state))
        XCTAssertNoThrow(try codec.decode(codes, state: codec.newStreamingState()))
    }
}
