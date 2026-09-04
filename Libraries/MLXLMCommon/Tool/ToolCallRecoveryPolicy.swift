// Copyright © 2026 Apple Inc.

import Foundation

/// Policy governing bounded cross-dialect tool-call recovery.
///
/// The selected `ToolCallFormat` parser is always authoritative. Recovery is a
/// secondary pass that may promote a call written in a *different* common
/// dialect when the model drifts from the selected one. Because recovery turns
/// response text into executable actions, it is deliberately conservative:
/// a missed recovery is preferable to a false execution.
public enum ToolCallRecoveryPolicy: String, Hashable, Sendable, CaseIterable {
    /// Only the selected parser may produce calls. No alternate-dialect
    /// recovery is attempted.
    case disabled

    /// Recover only structurally complete alternate-dialect calls that carry
    /// an explicit protocol marker (`<tool_call>`, `<|tool_call>`,
    /// `<function=`, `[TOOL_CALLS]`) and appear in committed response text —
    /// never inside reasoning spans (`<think>`/`<thinking>`/`[THINK]`),
    /// Markdown code spans or fences, or ordinary JSON data — and only when
    /// the call names an exactly declared tool and satisfies its declared
    /// required arguments.
    ///
    /// Explicit protocol attempts that are malformed, incomplete at end of
    /// stream, or undeclared are surfaced as ``RejectedToolCall`` rather than
    /// leaked as response text. Markerless `name[ARGS]{...}` rehearsals are
    /// never promoted by this policy: without a protocol marker, prose that
    /// merely mentions a declared call is indistinguishable from an intended
    /// one.
    case conservative

    /// Everything ``conservative`` does, plus promotion of exact markerless
    /// `declaredTool[ARGS]{...}` rehearsals in committed response text, and
    /// documented end-of-stream repair of calls whose outer closing marker is
    /// missing while their payload is structurally complete. Every promotion
    /// and repair is recorded in ``ToolCallProcessor/recoveryEvents`` with
    /// its provenance. Markerless candidates that fail validation remain
    /// response text.
    case permissive
}

/// Provenance for a tool call produced by cross-dialect recovery.
///
/// Recovery events feed diagnostics and telemetry without changing the
/// ordinary ``ToolCall`` execution interface.
public struct ToolCallRecoveryEvent: Hashable, Sendable {
    /// The textual dialect the model actually emitted.
    public enum Dialect: String, Hashable, Sendable {
        /// `<tool_call>{...}</tool_call>` or `<tool_call><function=...>...</tool_call>`.
        case toolCallFrame = "tool_call_frame"
        /// `<|tool_call>call:name{...}<tool_call|>`.
        case gemma4
        /// `<function=name><parameter=k>v</parameter></function>`.
        case qwenFunction = "qwen_function"
        /// `[TOOL_CALLS]name[ARGS]{...}`.
        case mistral
        /// Markerless `name[ARGS]{...}` rehearsal syntax (promoted only under
        /// ``ToolCallRecoveryPolicy/permissive``).
        case declaredArgs = "declared_args"
    }

    /// The repair applied to make the call executable.
    public enum Repair: String, Hashable, Sendable {
        /// The payload was structurally complete as emitted; only the dialect
        /// differed from the selected format.
        case alternateDialect = "alternate_dialect"
        /// The outer closing marker was missing at end of stream and the
        /// structurally complete payload was committed (permissive policy).
        case missingOuterClose = "missing_outer_close"
    }

    /// The recovered function name.
    public let toolName: String

    /// The recovered call identifier, when the dialect carried one.
    public let callID: String?

    /// The format selected for this generation (the authoritative parser).
    public let selectedFormat: ToolCallFormat

    /// The dialect the call was actually emitted in.
    public let dialect: Dialect

    /// The repair that was applied.
    public let repair: Repair

    /// Whether the call was committed at end of stream rather than from a
    /// complete in-stream frame.
    public let wasIncompleteAtEOS: Bool

    public init(
        toolName: String,
        callID: String?,
        selectedFormat: ToolCallFormat,
        dialect: Dialect,
        repair: Repair,
        wasIncompleteAtEOS: Bool
    ) {
        self.toolName = toolName
        self.callID = callID
        self.selectedFormat = selectedFormat
        self.dialect = dialect
        self.repair = repair
        self.wasIncompleteAtEOS = wasIncompleteAtEOS
    }
}
