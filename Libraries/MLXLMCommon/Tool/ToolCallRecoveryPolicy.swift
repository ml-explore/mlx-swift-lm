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

    /// Recover only structurally complete alternate-dialect calls that appear
    /// in committed response text — never inside reasoning spans
    /// (`<think>`/`<thinking>`/`[THINK]`), Markdown code spans or fences, or
    /// ordinary JSON data — and only when the call names an exactly declared
    /// tool and satisfies its declared required arguments.
    ///
    /// Explicit protocol attempts that are malformed, incomplete at end of
    /// stream, or undeclared are surfaced as ``RejectedToolCall`` rather than
    /// leaked as response text. Ambiguous markerless candidates
    /// (`name[ARGS]{...}`) that fail validation remain response text.
    case conservative

    /// Everything ``conservative`` does, plus documented end-of-stream repair
    /// of calls whose outer closing marker is missing while their payload is
    /// structurally complete. Every repair is recorded in
    /// ``ToolCallProcessor/recoveryEvents`` with its provenance.
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
        /// Markerless `name[ARGS]{...}` rehearsal syntax.
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
