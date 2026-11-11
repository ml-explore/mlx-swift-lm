import MLX
import XCTest

@testable import MLXVLM

final class LFM3OmniAudioFeaturesTests: XCTestCase {
    func testStreamingPreservesWindowBoundariesAndFinalPadding() throws {
        let frontend = LFM3OmniAudioFeatures()
        var state: UInt32 = 17
        let samples = (0 ..< 1733).map { _ -> Float in
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(state >> 8) / Float(1 << 24) - 0.5
        }
        let input = MLXArray(samples)
        let expected = try frontend(input)
        for chunkSize in [1, 159, 160, 399, 401, 800] {
            let state = LFM3OmniFeatureStreamingState()
            var chunks = [MLXArray]()
            for offset in stride(from: 0, to: input.size, by: chunkSize) {
                chunks.append(
                    try frontend.stream(
                        input[offset ..< min(offset + chunkSize, input.size)], state: state))
            }
            chunks.append(try frontend.stream(MLXArray.zeros([0]), state: state, finalize: true))
            let actual = concatenated(chunks, axis: 1)
            print(
                "Feature stream chunk \(chunkSize) max error \(abs(actual - expected).max().item(Float.self))"
            )
            XCTAssertEqual(actual.shape, expected.shape)
            XCTAssertTrue(allClose(actual, expected, rtol: 0.001, atol: 0.001).item(Bool.self))
            XCTAssertThrowsError(try frontend.stream(input, state: state))
        }
    }

    func testStreamingLowEnergyBandsPreserveMelPower() throws {
        let frontend = LFM3OmniAudioFeatures()
        let input = sin(MLXArray(0 ..< 1733).asType(.float32) * 0.13)
        let expected = try frontend(input)
        let state = LFM3OmniFeatureStreamingState()
        var chunks = [MLXArray]()
        for offset in stride(from: 0, to: input.size, by: 159) {
            chunks.append(
                try frontend.stream(input[offset ..< min(offset + 159, input.size)], state: state))
        }
        chunks.append(try frontend.stream(MLXArray.zeros([0]), state: state, finalize: true))
        let actual = concatenated(chunks, axis: 1)
        // Logarithms magnify FFT rounding near silence; compare the underlying band power.
        let actualPower = pow(MLXArray(Float(10)), actual * 4 - 4)
        let expectedPower = pow(MLXArray(Float(10)), expected * 4 - 4)
        print(
            "Mel power absolute error \(abs(actualPower - expectedPower).max().item(Float.self)), reference max \(expectedPower.max().item(Float.self)), relative \((abs(actualPower - expectedPower) / maximum(expectedPower, 1e-8)).max().item(Float.self))"
        )
        XCTAssertTrue(allClose(actualPower, expectedPower, rtol: 0.002, atol: 1e-9).item(Bool.self))
    }

    func testSilenceAndTailPadding() throws {
        let frontend = LFM3OmniAudioFeatures()
        for (samples, frames) in [(1, 1), (400, 1), (401, 2), (560, 2), (561, 3)] {
            let features = try frontend(MLXArray.zeros([samples]))
            XCTAssertEqual(features.shape, [128, frames])
            XCTAssertEqual(abs(features + 2).max().item(Float.self), 0)
        }
    }

    func testAllBandsMatchIndependentReference() throws {
        // Generated with tools/lfm3-omni-features-reference.py.
        let expected: [Float] = [
            0.770608578, 0.868173009, -0.78642446, -0.820879235, -0.648801809, -0.71188519,
            -0.564136885, -0.54460696,
            -0.521366282, -0.395932702, -0.482613765, -0.259522308, -0.257909811, -0.173415634,
            -0.03337907, -0.079233377,
            0.21393286, 0.362197333, 0.521486669, 1.20016602, 1.21231486, 1.37177794, 1.24999612,
            1.21280798,
            0.347609155, 0.419591892, 0.123380969, 0.0658722732, -0.0435682177, -0.267482396,
            -0.251482677, -0.585307095,
            -0.584187077, -0.437138638, -0.424770663, -0.18376802, -0.0947792667, -0.0348627453,
            0.191242514, 0.121746063,
            0.570708338, 1.00256268, 1.12522571, 1.18774662, 1.13820959, 1.01418603, 0.339080525,
            0.321906165,
            0.115127477, -0.122981327, -0.188007487, -0.315632312, -0.563416547, -0.478139765,
            -0.28316155, -0.129303367,
            0.0311105868, 0.190615854, 0.560242518, 1.03072155, 1.0887642, 0.962133905, 0.698461446,
            0.197070496,
            -0.0539830889, -0.255231855, -0.489464603, -0.442250005, -0.176131443, 0.0819541872,
            0.488154746, 0.931822303,
            1.01513482, 0.824765556, 0.200038257, -0.0825941922, -0.326941337, -0.55310358,
            -0.196411094, 0.0994886001,
            0.785582106, 0.953763693, 0.8287442, 0.220894832, -0.192834165, -0.520523853,
            -0.248735336, 0.35470547,
            0.834942512, 0.893937571, 0.379998274, -0.215347355, -0.555642282, -0.174785644,
            0.647900614, 0.87200781,
            0.673806392, -0.157258328, -0.54639229, -0.0933151921, 0.771956702, 0.799282721,
            0.0550365999, -0.556692745,
            -0.210173393, 0.765663873, 0.742805864, -0.208990755, -0.760219808, 0.631359891,
            0.775081372, 0.397948596,
            -0.673675718, 0.626012846, 0.749229124, 0.296261467, -0.222291243, 0.710262718,
            0.65625954, -0.491246216,
            0.616565484, 0.706906071, 0.00865944379, 0.574696897, 0.704586634, 0.128725626,
            0.615480911, 0.674073794,
            0.770382852, 0.867947282, -0.674554264, -0.709009039, -0.584483744, -0.667522199,
            -0.528277179, -0.525416963,
            -0.502176285, -0.386354749, -0.474092829, -0.255250357, -0.256352975, -0.171858797,
            -0.0330144868, -0.0788687938,
            0.213916896, 0.362156274, 0.521444821, 1.20016878, 1.21231762, 1.37177539, 1.24999399,
            1.21281651,
            0.347994656, 0.419977393, 0.125452987, 0.067944291, -0.0356906964, -0.252348254,
            -0.221392989, -0.417136159,
            -0.416016141, -0.339084271, -0.380969117, -0.165118668, -0.0902216726, -0.0303051511,
            0.192218887, 0.122722436,
            0.57081922, 1.00256024, 1.12522315, 1.18774533, 1.13820848, 1.01419785, 0.339435414,
            0.322283224,
            0.11715444, -0.114918292, -0.1761228, -0.283704495, -0.382379507, -0.38135836,
            -0.256904365, -0.12074188,
            0.0338958025, 0.191716506, 0.560370459, 1.03071463, 1.08876426, 0.962137681, 0.69848096,
            0.197715793,
            -0.050032006, -0.237098093, -0.393621738, -0.365934649, -0.164194359, 0.0839006558,
            0.488312574, 0.931813922,
            1.01513514, 0.824771218, 0.200464562, -0.0790244963, -0.300313211, -0.436940299,
            -0.184399617, 0.101056439,
            0.785584165, 0.953759827, 0.828750258, 0.221065176, -0.186991751, -0.43911884,
            -0.235865308, 0.354910099,
            0.834937205, 0.893938632, 0.379992027, -0.211251753, -0.484030229, -0.169068061,
            0.64791761, 0.872003318,
            0.673817995, -0.156393772, -0.502923628, -0.0910103978, 0.771956155, 0.799282762,
            0.0548499758, -0.541310718,
            -0.207267283, 0.76566393, 0.742805779, -0.209868727, -0.746647025, 0.631364099,
            0.775079599, 0.397976069,
            -0.690921289, 0.626014733, 0.749228085, 0.296304548, -0.222590154, 0.710262526,
            0.65626005, -0.479933285,
            0.616565188, 0.70690578, 0.00899805407, 0.574697736, 0.704585648, 0.128892328,
            0.615480809, 0.674073324,
            0.770373868, 0.867938298, -0.669924703, -0.704379478, -0.583848093, -0.668934203,
            -0.530696862, -0.52949957,
            -0.506258892, -0.39085733, -0.478505568, -0.259293782, -0.259444119, -0.174949941,
            -0.0349978804, -0.0808521873,
            0.212932698, 0.361845995, 0.521155654, 1.20018105, 1.21232989, 1.37176636, 1.24998632,
            1.21284349,
            0.34913073, 0.421113467, 0.131186442, 0.0736777457, -0.0161170697, -0.218467559,
            -0.165049828, -0.290155421,
            -0.289035402, -0.231619977, -0.308620292, -0.124456731, -0.0778809295, -0.017964408,
            0.195104057, 0.125607607,
            0.571170718, 1.00255055, 1.12521306, 1.18774369, 1.13820727, 1.01422528, 0.340320849,
            0.323225677,
            0.122227001, -0.0962016538, -0.14984802, -0.227582472, -0.256993986, -0.277965518,
            -0.207095154, -0.100363926,
            0.0411713589, 0.194688154, 0.560724975, 1.03069509, 1.08876488, 0.962147819, 0.69853132,
            0.19939929,
            -0.0402492602, -0.200281036, -0.293045017, -0.27570578, -0.138947981, 0.0884999127,
            0.488674089, 0.931797786,
            1.01513298, 0.824789481, 0.201807562, -0.0694326523, -0.250481616, -0.330325512,
            -0.160106956, 0.104444192,
            0.785592797, 0.95375073, 0.828760518, 0.221760084, -0.171164455, -0.345083218,
            -0.210653876, 0.355308831,
            0.834932059, 0.893935832, 0.380138415, -0.198549234, -0.395113073, -0.157548171,
            0.647950756, 0.871995801,
            0.673830945, -0.152068069, -0.43528371, -0.0867103595, 0.77195804, 0.799280135,
            0.0550247856, -0.502464917,
            -0.202212656, 0.765665392, 0.742804076, -0.210191301, -0.706343546, 0.631370114,
            0.775077631, 0.398007213,
            -0.712965115, 0.626015991, 0.749227313, 0.296363946, -0.22209794, 0.710261123,
            0.656262564, -0.454875602,
            0.616562784, 0.706905844, 0.00975541774, 0.57469805, 0.704583842, 0.129276426,
            0.615480279, 0.67407225,
            0.774935093, 0.872499523, 0.569070007, 0.534615232, 0.45125214, 0.286660641,
            0.340844887, 0.359873219,
            0.383113897, 0.508211252, 0.415573481, 0.611515542, 0.578456656, 0.662950834,
            0.713945111, 0.668090804,
            0.856563703, 0.863838716, 1.01363762, 1.18222329, 1.19437212, 1.2985242, 1.18310753,
            1.2429321,
            0.904493135, 0.976475872, 0.887255896, 0.829747199, 0.859435207, 0.714549077,
            0.81521957, 0.733725218,
            0.734845237, 0.764655716, 0.655209645, 0.778884494, 0.673318178, 0.7332347, 0.77255408,
            0.70305763,
            0.926078928, 0.99553222, 1.11488496, 1.13290214, 1.08594266, 1.06556375, 0.742185516,
            0.735549677,
            0.753687886, 0.667956428, 0.652212061, 0.648776144, 0.680604885, 0.63911498,
            0.650834466, 0.664196116,
            0.681902291, 0.715115817, 0.860333857, 1.01176681, 1.02728935, 0.938579763, 0.759647195,
            0.528069263,
            0.451886049, 0.407408711, 0.415536991, 0.438237759, 0.487694805, 0.543949294,
            0.709319499, 0.896662671,
            0.951780152, 0.841189824, 0.563549337, 0.488322143, 0.430835031, 0.369897173,
            0.386428265, 0.425662761,
            0.77705232, 0.905235436, 0.833390837, 0.57480425, 0.520252296, 0.49622785, 0.491509827,
            0.570404878,
            0.818740492, 0.860655023, 0.588935431, 0.448729728, 0.441147032, 0.480818405,
            0.685341435, 0.826922662,
            0.658220651, 0.268343896, 0.241413424, 0.370492809, 0.733911801, 0.758529064,
            0.462207391, 0.319983223,
            0.338120681, 0.733009418, 0.724105087, 0.446271019, 0.408346216, 0.635022278,
            0.745823573, 0.470669183,
            0.364200122, 0.618929226, 0.70804873, 0.357363419, 0.252516102, 0.669691746,
            0.630220605, 0.310272418,
            0.591047917, 0.685664928, 0.394417315, 0.57651582, 0.67794177, 0.347127516, 0.59364091,
            0.63630927,
        ]
        // GPU FFT rounding is amplified by the logarithm in low-energy bands.
        for (device, tolerance): (Device, Float) in [(.cpu, 1e-4), (.gpu, 1e-3)] {
            try Device.withDefaultDevice(device) {
                let samples = MLXArray((0 ..< 721).map { Float($0 % 31) / 32 - 0.5 })
                let actual = try LFM3OmniAudioFeatures()(samples).transposed()
                XCTAssertEqual(actual.shape, [4, 128])
                let error = abs(actual - MLXArray(expected).reshaped(4, 128)).max().item(Float.self)
                print("LFM3-Omni log-mel maximum error on \(device): \(error)")
                XCTAssertLessThan(error, tolerance)
            }
        }
    }

    func testRejectsSpectrogramOverflow() {
        XCTAssertThrowsError(
            try LFM3OmniAudioFeatures()(
                MLXArray.full([400], values: MLXArray(Float.greatestFiniteMagnitude))))
    }

    func testSampleRateContract() throws {
        let frontend = LFM3OmniAudioFeatures()
        let samples = MLXArray.zeros([400])
        XCTAssertNoThrow(try frontend(samples, sampleRate: 16_000))
        for rate in [0, -1, 8_000, 24_000, 44_100, 48_000] {
            XCTAssertThrowsError(try frontend(samples, sampleRate: rate))
        }
    }

    func testStridedAndReducedPrecisionInput() throws {
        let frontend = LFM3OmniAudioFeatures()
        let storage = MLXArray((0 ..< 1442).map { Float($0 % 31) / 32 - 0.5 })
        let strided = asStrided(storage, [721], strides: [2])
        let contiguous = MLXArray(strided.asArray(Float.self))
        let expected = try frontend(contiguous)
        XCTAssertLessThan(abs(try frontend(strided) - expected).max().item(Float.self), 1e-5)
        for dtype: DType in [.float16, .bfloat16] {
            let actual = try frontend(contiguous.asType(dtype))
            XCTAssertEqual(actual.dtype, .float32)
            XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 1e-5)
        }
    }

    func testRejectsInvalidWaveforms() {
        let frontend = LFM3OmniAudioFeatures()
        XCTAssertThrowsError(try frontend(MLXArray.zeros([0])))
        XCTAssertThrowsError(try frontend(MLXArray.zeros([2, 400])))
        XCTAssertThrowsError(try frontend(MLXArray([Float.nan])))
        XCTAssertThrowsError(try frontend(MLXArray([Float.infinity])))
        XCTAssertThrowsError(try frontend(MLXArray([1, 2, 3])))
    }
}
