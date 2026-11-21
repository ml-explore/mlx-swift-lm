// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXVLM

/// A right-padded row must store the conv state of its last real tokens, as
/// mlx-lm qwen3_5 does with `cache.lengths`, not the state of its pad steps.
final class Qwen35RightPaddedConvStateTests: XCTestCase {

    private let configurationJSON = """
        {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 32,
            "full_attention_interval": 2,
            "num_experts": 0,
            "num_experts_per_tok": 0
        }
        """

    private let lengths = [6, 3]

    /// Runs the rows right-padded in one batch, then each row alone, and
    /// compares the outputs and both cache slots row by row.
    private func assertRowsMatch(
        _ layer: (MLXArray, MLXArray?, MambaCache) -> MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let width = lengths.max()!
        let rows = lengths.map { MLXRandom.normal([1, $0, 64]) }
        let batch = concatenated(
            rows.map { row in
                padded(row, widths: [[0, 0], [0, width - row.dim(1)], [0, 0]])
            })

        let batchedCache = MambaCache()
        batchedCache.prepare(lengths: lengths)
        let batched = layer(batch, createSSMMask(h: batch, cache: batchedCache), batchedCache)
        batchedCache.finalize()

        for (index, row) in rows.enumerated() {
            let aloneCache = MambaCache()
            let alone = layer(row, nil, aloneCache)
            let n = row.dim(1)
            assertClose(
                batched[index ..< (index + 1), ..<n], alone,
                atol: 1e-5, reducedAtol: 1e-3, "\(label) output, row \(index)",
                file: file, line: line)
            for slot in 0 ..< 2 {
                assertClose(
                    batchedCache[slot]![index ..< (index + 1)], aloneCache[slot]!,
                    atol: 1e-5, reducedAtol: 1e-3, "\(label) slot \(slot), row \(index)",
                    file: file, line: line)
            }
        }
    }

    func testLLMRightPaddedRowsKeepTheirOwnConvState() throws {
        MLXRandom.seed(47)
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(configurationJSON.utf8))
        let gdn = Qwen35GatedDeltaNet(configuration)
        assertRowsMatch({ gdn($0, mask: $1, cache: $2) }, "LLM")
    }

    func testVLMRightPaddedRowsKeepTheirOwnConvState() throws {
        MLXRandom.seed(47)
        let configuration = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(configurationJSON.utf8))
        let gdn = Qwen35Language.GatedDeltaNet(configuration)
        assertRowsMatch({ gdn($0, mask: $1, cache: $2) }, "VLM")
    }
}
