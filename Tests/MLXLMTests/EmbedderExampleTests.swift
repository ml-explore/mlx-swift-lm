// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXEmbedders
import Testing
import Tokenizers

struct EmbedderExampleTests {

    private func readeMeExampleResult() async throws -> ([String], [[Float]]) {
        let modelContainer = try await loadModelContainer(configuration: .nomic_text_v1_5)
        let searchInputs = [
            "search_query: Animals in Tropical Climates.",
            "search_document: Elephants",
            "search_document: Horses",
            "search_document: Polar Bears",
        ]

        // Generate embeddings
        let resultEmbeddings = await modelContainer.perform {
            (model: EmbeddingModel, tokenizer: Tokenizer, pooling: Pooling) -> [[Float]] in
            let inputs = searchInputs.map {
                tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            // Pad to longest
            let maxLength = inputs.reduce(into: 16) { acc, elem in
                acc = max(acc, elem.count)
            }

            let padded = stacked(
                inputs.map { elem in
                    MLXArray(
                        elem
                            + Array(
                                repeating: tokenizer.eosTokenId ?? 0,
                                count: maxLength - elem.count))
                })
            let mask = (padded .!= tokenizer.eosTokenId ?? 0)
            let tokenTypes = MLXArray.zeros(like: padded)
            let result = pooling(
                model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
                normalize: true, applyLayerNorm: true
            )
            result.eval()
            return result.map { $0.asArray(Float.self) }
        }

        return (searchInputs, resultEmbeddings)
    }

    @Test("README.md example")
    func testReadMeExample() async throws {
        guard let (searchInputs, resultEmbeddings) = try? await readeMeExampleResult() else {
            throw NSError(
                domain: "EmbedderExampleTests",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to get example results"]
            )
        }

        // Compute similarities
        let searchQueryEmbedding = resultEmbeddings[0]
        let documentEmbeddings = resultEmbeddings[1...]
        let similarities = documentEmbeddings.map { documentEmbedding in
            zip(searchQueryEmbedding, documentEmbedding).map(*).reduce(0, +)
        }

        //        let searchQueryName = searchInputs[0].replacingOccurrences(of: "search_query: ", with: "")
        //        let documentNames = searchInputs[1...].map{ $0.replacingOccurrences(of: "search_document: ", with: "") }
        //
        //        print()
        //        print("Similarities to '\(searchQueryName)':")
        //        for (index, sim) in similarities.enumerated() {
        //            print(" -> \(sim)\tfor Document '\(documentNames[index])'" )
        //        }
        //        print()

        #expect(
            searchInputs.count == 4,
            "If this fails please update the test with new search inputs."
        )
        #expect(
            similarities.count == 3,
            "If this fails please update the test with new search inputs."
        )
        #expect(
            similarities[0] > similarities[1],
            "Elephants should be more similar to the query than Horses."
        )
    }
}
