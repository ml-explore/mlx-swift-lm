// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

extension ContinuationAssertions {
    /// Qwen-style M-RoPE positions are built for one sequence. A batch that carries media must
    /// throw instead of placing every row like the last. A text-only batch still prepares.
    func assertBatchedMediaThrows<M: LanguageModel>(
        _ model: M, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        try withRandomState(MLXRandom.RandomState(seed: 71)) {
            let row = concatenated([textTokens(5), imageRun(), textTokens(4, seed: 3)], axis: 1)
            let batch = concatenated([row, row], axis: 0)
            // One image per row, so the feature count matches the placeholders.
            let one = image()
            let two = LMInput.ProcessedImage(
                pixels: concatenated([one.pixels, one.pixels]),
                frames: (one.frames ?? []) + (one.frames ?? []))

            XCTAssertThrowsError(
                try model.prepare(
                    LMInput(text: .init(tokens: batch), image: two),
                    cache: try model.newCache(parameters: nil), state: nil,
                    prefill: .init(chunking: .unchunked)),
                "a batch with media must throw", file: file, line: line
            ) { error in
                guard case VLMError.processing = error else {
                    return XCTFail(
                        "expected VLMError.processing, got \(error)", file: file, line: line)
                }
            }

            let textBatch = concatenated([textTokens(9), textTokens(9, seed: 4)], axis: 0)
            XCTAssertNoThrow(
                try model.prepare(
                    LMInput(text: .init(tokens: textBatch)),
                    cache: try model.newCache(parameters: nil), state: nil,
                    prefill: .init(chunking: .unchunked)),
                "a text-only batch must still prepare", file: file, line: line)
        }
    }
}
