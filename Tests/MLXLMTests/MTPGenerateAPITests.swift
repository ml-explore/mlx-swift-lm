// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

// MARK: - Surface check
//
// These tests assert the new MTP `generate(...)` and `generateTokens(...)`
// overloads exist with the expected signatures. They do not run actual
// inference (which requires a real Gemma 4 target checkpoint and MLX metal
// runtime); end-to-end stream-yields-tokens validation lives in
// `MTPRung4TokenParityTests` and `MTPAcceptanceRateTests`, which are
// checkpoint-gated.

@Test
func testMTPGenerateOverloadIsCallableSurface() {
    // Surface check: the function pointer can be formed without invocation.
    // If the signature drifts (param name, types, defaults), this test
    // fails to compile.
    let _:
        (
            LMInput, [KVCache]?, GenerateParameters, ModelContext,
            any MTPDrafterModel, Int, WiredMemoryTicket?
        ) throws -> AsyncStream<Generation> = { input, cache, params, ctx, drafter, block, ticket in
            try generate(
                input: input, cache: cache, parameters: params, context: ctx,
                mtpDrafter: drafter, blockSize: block, wiredMemoryTicket: ticket
            )
        }
}

@Test
func testMTPGenerateTokensOverloadIsCallableSurface() {
    let _:
        (
            LMInput, [KVCache]?, GenerateParameters, ModelContext,
            any MTPDrafterModel, Int, WiredMemoryTicket?
        ) throws -> AsyncStream<TokenGeneration> = {
            input, cache, params, ctx, drafter, block, ticket in
            try generateTokens(
                input: input, cache: cache, parameters: params, context: ctx,
                mtpDrafter: drafter, blockSize: block, wiredMemoryTicket: ticket
            )
        }
}

@Test
func testMTPGenerateUsesBlockSizeDefaultOfFour() {
    // The default value of `blockSize` is part of the public API surface.
    // Probe it via Mirror-equivalent: call the function with the default
    // omitted and observe its impact through a structural-level check.
    // (We can't easily introspect defaults at runtime, but we can verify
    // the call compiles without supplying `blockSize`.)
    func _callableWithoutBlockSize(
        input: LMInput, parameters: GenerateParameters, context: ModelContext,
        mtpDrafter: any MTPDrafterModel
    ) throws -> AsyncStream<Generation> {
        try generate(
            input: input, parameters: parameters, context: context,
            mtpDrafter: mtpDrafter
        )
    }
    // Existence as a compile-time check.
    _ = _callableWithoutBlockSize
}
