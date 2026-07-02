// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXEmbedders

struct NomicBertTests {

    @Test
    func testFullRoPESkipsLearnedPositions() throws {
        // NomicEmbedding skips learned position table when rotary_emb_fraction == 1.0
        // nomic-embed-text-v1.5 has rotary_emb_fraction == 1.0 and no learned position_embeddings.weight
        // tensor.
        // Loading the module unconditionally caused weight loading
        // to fail with "Key embeddings.position_embeddings.weight not found".
        let json = """
            {
              "vocab_size": 100,
              "n_embd": 32,
              "n_head": 4,
              "n_layer": 1,
              "max_position_embeddings": 2048,
              "rotary_emb_fraction": 1.0
            }
            """
        let config = try JSONDecoder().decode(
            NomicBertConfiguration.self, from: Data(json.utf8))
        let embedding = NomicEmbedding(config)

        #expect(embedding.positionEmbeddings == nil)
    }

    @Test
    func testPartialOrNoRoPEAllocatesLearnedPositions() throws {
        // NomicEmbedding allocates learned position table when rotary_emb_fraction < 1.0
        let partialRopeJson = """
            {
              "vocab_size": 100,
              "n_embd": 32,
              "n_head": 4,
              "n_layer": 1,
              "max_position_embeddings": 512,
              "rotary_emb_fraction": 0.4
            }
            """
        let config = try JSONDecoder().decode(
            NomicBertConfiguration.self, from: Data(partialRopeJson.utf8))
        let embedding = NomicEmbedding(config)

        #expect(embedding.positionEmbeddings != nil)
    }
}
