// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class UnreadCacheStateTests: XCTestCase {

    private static let tinyConfigJSON = """
        {
          "model_type": "lfm2",
          "vocab_size": 32,
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "num_attention_heads": 16,
          "num_key_value_heads": 1,
          "max_position_embeddings": 128,
          "norm_eps": 1e-5,
          "conv_bias": false,
          "conv_L_cache": 3,
          "block_dim": 64,
          "intermediate_size": 192,
          "block_multiple_of": 8,
          "block_ffn_dim_multiplier": 1.0,
          "block_auto_adjust_ff_dim": false,
          "layer_types": ["conv", "full_attention"],
          "rope_parameters": { "rope_theta": 10000000.0 }
        }
        """

    func testConvolutionCacheUnreadStateIsItsRewindHistory() {
        let cache = RewindableConvolutionCache(stateLength: 2, rollbackCapacity: 4)
        var state = MLXArray.zeros([1, 2, 3])
        for step in 0 ..< 3 {
            let input = MLXArray.ones([1, 1, 3]) * Float(step)
            let next = state + input
            cache.record(input: input, initialState: state, currentState: next)
            state = next
        }

        let unread = (cache as any KVCache).unreadState
        XCTAssertEqual(unread.count, 1)
        XCTAssertTrue(unread.first === cache[1])
        XCTAssertFalse(unread.first === cache[0])
    }

    func testGenerationKeepsConvolutionHistoryEvaluated() throws {
        let model = LFM2Model(
            try JSONDecoder().decode(
                LFM2Configuration.self, from: Data(Self.tinyConfigJSON.utf8)))
        eval(model)
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([1, 2, 3, 4, 5])), model: model,
            parameters: GenerateParameters(maxTokens: 576, temperature: 0))

        var activeMemory = [Int]()
        for count in [64, 512] {
            for _ in 0 ..< count {
                XCTAssertNotNil(iterator.next())
            }
            Stream.defaultStream.synchronize()
            Memory.clearCache()
            activeMemory.append(Memory.activeMemory)
        }
        // 512 tokens add 16 KiB of attention cache. A convolution history left unevaluated
        // keeps each step's input alive and adds 128 KiB more.
        XCTAssertLessThan(activeMemory[1] - activeMemory[0], 48 * 1024)
    }
}
