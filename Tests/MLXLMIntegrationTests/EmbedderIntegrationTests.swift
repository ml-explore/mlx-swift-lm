// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

struct EmbedderIntegrationtests {

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

    @Test("MLXEmbedders README.md example")
    func testReadMeExample() async throws {
        guard let (searchInputs, resultEmbeddings) = try? await readeMeExampleResult() else {
            throw NSError(
                domain: "EmbedderIntegrationtests",
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
        let documentNames = searchInputs[1...].map {
            $0.replacingOccurrences(of: "search_document: ", with: "")
        }
        let expectedSimilarities: [Float] = [
            0.6854175,  // Elephants
            0.6644787,  // Horses
            0.63326025,  // Polar Bears
        ]

        for (index, resultSimilarity) in similarities.enumerated() {
            #expect(
                abs(resultSimilarity - expectedSimilarities[index]) < 0.01,
                "The expected similarity does not match the result similarity for \(documentNames[index])"
            )
        }
    }

    @Test("Gemma 3 Embedder integration")
    func testGemma3Embedder() async throws {
        // Gemma 3 1B model
        let modelId = "mlx-community/gemma-3-1b-it-qat-4bit"
        let modelContainer = try await loadModelContainer(configuration: .init(id: modelId))

        let inputs = [
            "The Coca-Cola Company is a soft drink company based in Atlanta, Georgia, USA.",
            "In the United States, PepsiCo Inc. is a leading soft drink company.",
        ]

        let resultEmbeddings = await modelContainer.perform {
            (model: EmbeddingModel, tokenizer: Tokenizer, pooling: Pooling) -> [[Float]] in
            let encoded = inputs.map {
                tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            // Pad to longest sequence
            let maxLength = encoded.reduce(into: 1) { acc, elem in
                acc = max(acc, elem.count)
            }

            let padded = stacked(
                encoded.map { elem in
                    MLXArray(
                        elem
                            + Array(
                                repeating: tokenizer.eosTokenId ?? 0,
                                count: maxLength - elem.count))
                })

            // Mask out padding tokens
            let mask = (padded .!= (tokenizer.eosTokenId ?? 0))
            let tokenTypes = MLXArray.zeros(like: padded)

            // Generate embeddings. EmbeddingGemma returns a pooledOutput by default.
            let modelOutput = model(
                padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask)

            // Pooling strategy .cls (the default if no pooling config exists)
            // will pick up the pooledOutput from the EmbeddingGemma model.
            let result = pooling(
                modelOutput,
                normalize: true, applyLayerNorm: true
            )
            result.eval()
            return result.map { $0.asArray(Float.self) }
        }

        #expect(resultEmbeddings.count == inputs.count, "Should have one embedding per input")
        for embedding in resultEmbeddings {
            // Gemma 3 1B hidden size is 1152
            #expect(embedding.count == 1152, "Gemma 3 1B embedding size should be 1152")

            // Verify normalization (L2 norm should be ~1.0)
            let l2Norm = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
            #expect(abs(l2Norm - 1.0) < 0.05, "Embeddings should be approximately L2-normalized")
        }

        // Basic semantic check: similarity between related sentences should be positive
        let similarity = zip(resultEmbeddings[0], resultEmbeddings[1]).map(*).reduce(0, +)
        //print("similarity: \(similarity)")
        #expect(similarity > 0.0, "Similarity between related sentences should be positive")
    }

}
