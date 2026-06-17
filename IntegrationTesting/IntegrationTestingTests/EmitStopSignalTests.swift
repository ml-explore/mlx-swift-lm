// Copyright © 2026 Apple Inc.
//
// Regression tests for the emit-callback stop-signal contract in
// `GuidedGenerationLoop.run`. Contract: when the caller's `emit`
// closure returns `false`, the loop must stop generating promptly
// -- no further `emit` invocations, no further model forward passes.
//
// The subtle path: when `emit` returns `false` during fast-forward
// yielding, the loop must still stop promptly. The inner `for` over
// `ffTokens` must propagate the stop signal to the outer `while` so it
// does not sample another token and call `emit` again -- which would
// violate the "emit=false stops generation" contract.
//
// Shape of the failure this test detects: `emit` returning `false`
// on the sampled-token path already breaks the outer `while`
// cleanly, so a test that always returns `false` would exit on the
// first call regardless of the bug. To exercise the FF path
// specifically, the callback returns `true` on the first call
// (which lines up with the first sampled-token emit) and `false`
// thereafter. The second call almost always lands on an FF-yielded
// text because the schema -- a single `const` string field -- forces
// the entire body as FF after the opening `{`.

#if FoundationModelsIntegration

    import Testing
    import Foundation
    import MLX
    import MLXLMCommon
    @testable import MLXFoundationModels
    @testable import MLXGuidedGeneration

    @Suite(.serialized, .timeLimit(.minutes(2)))
    struct EmitStopSignalTests {

        @Test("GuidedGenerationLoop honors emit=false during fast-forward yielding")
        func emitStopSignalHonoredDuringFastForward() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            // Const-string schema: after `{` is sampled, the grammar forces
            // the entire remaining body (`"k":"abcdefghij"}`) as FF. That
            // guarantees the loop enters the FF yield path on its first
            // iteration, which is the only path where the stop-signal bug
            // manifests.
            let schema = """
                {
                  "type": "object",
                  "properties": { "k": { "const": "abcdefghij" } },
                  "required": ["k"],
                  "additionalProperties": false
                }
                """

            let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

            try await container.perform { context in
                let xgTokenizer = try await MLXLanguageModel.makeXGTokenizer(
                    modelID: TestFixtures.defaultModelID,
                    tokenizer: context.tokenizer
                )
                let constraint = try GrammarConstraint(
                    tokenizer: xgTokenizer,
                    jsonSchema: schema,
                    fastForward: true,
                    hostTokenizer: context.tokenizer
                )

                let messages: [[String: any Sendable]] = [
                    ["role": "user", "content": "Emit the schema value."]
                ]
                let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
                let input = LMInput(tokens: MLXArray(tokens))

                var callCount = 0
                var callsAfterFalse = 0
                var firstFalseAt: Int? = nil

                // Return `true` on the first call so the loop enters at
                // least one FF yield pass. Return `false` thereafter. Any
                // call made after `firstFalseAt` is set violates the
                // stop-signal contract.
                let tokensGenerated = try GuidedGenerationLoop.run(
                    input: input,
                    context: context,
                    constraint: constraint,
                    maxTokens: 128,
                    vocabSize: Int(xgTokenizer.vocabSize)
                ) { _ in
                    callCount += 1
                    if firstFalseAt != nil {
                        callsAfterFalse += 1
                    }
                    if callCount >= 2 {
                        if firstFalseAt == nil { firstFalseAt = callCount }
                        return false
                    }
                    return true
                }

                #expect(
                    callsAfterFalse == 0,
                    """
                    emit() returned false on call #\(firstFalseAt ?? -1) but the \
                    loop continued to call emit \(callsAfterFalse) more time(s). \
                    The caller's stop signal must halt generation immediately, \
                    including when it lands during fast-forward yielding. \
                    tokensGenerated=\(tokensGenerated).
                    """
                )
            }
        }
    }

#endif
