// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels

/// Synthetic tool used by MLX's tool-calling path to encode the model's
/// free-text response as a structured tool call.
///
/// MLX constrains tool-calling generation to a JSON schema shaped as
/// `{oneOf: [{name: "T_i", arguments: <T_i.parameters>}, …]}`. The
/// developer's real tools are the `T_1…T_N`; this synthetic tool is the
/// extra `T_{N+1}` whose arguments carry the text (or structured response)
/// the model wants to deliver directly to the user.
///
/// When the model picks this tool at generation time, the executor does not
/// emit a `toolCallDelta` for it -- instead it extracts the `arguments`
/// payload and re-emits it as `textDelta` events, so consumers of the
/// channel see text in the same shape they would for a tools-free response.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum FinalAnswerTool {

    /// Reserved tool name. Developers must not register a real tool with
    /// this name; if they do, resolution silently keeps the synthetic
    /// tool (no auto-renaming).
    static let toolName = "mlx_final_answer"

    /// Human-readable description shown to the model alongside the real
    /// tools' descriptions.
    static let toolDescription = """
        Call this tool to respond directly to the user in natural language. \
        Use it when no other tool is needed, or once information gathered \
        from prior tool calls is sufficient to answer the user's request.
        """

    /// Wrapper schema used when the request has no developer-supplied
    /// response schema. The tool's single argument `response` carries the
    /// free-text response; the executor unwraps it into text deltas.
    @Generable
    struct StringResponse {
        @Guide(description: "The natural-language response to return to the user.")
        var response: String
    }

    /// Builds the `Transcript.ToolDefinition` the model should see in its
    /// prompt, alongside the developer's real tools.
    ///
    /// - Parameter responseSchema: The developer-provided response schema
    ///   for the current request, if any.
    ///   - `nil`: the synthetic tool uses the `StringResponse` wrapper, so
    ///     the tool's arguments are `{"response": "<text>"}`.
    ///   - non-`nil`: the developer's schema is used verbatim as the
    ///     synthetic tool's `arguments` schema. Consumers then decode the
    ///     tool's arguments JSON through their own `GenerationSchema`.
    static func makeToolDefinition(
        responseSchema: GenerationSchema?
    ) -> Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: toolName,
            description: toolDescription,
            parameters: parameterSchema(for: responseSchema)
        )
    }

    /// Selects the schema used for the synthetic tool's `arguments`.
    static func parameterSchema(
        for responseSchema: GenerationSchema?
    ) -> GenerationSchema {
        responseSchema ?? StringResponse.generationSchema
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
