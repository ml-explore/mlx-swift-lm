// Copyright © 2026 Apple Inc.

/// Typed pieces of the OpenAI Harmony response format used by GPT-OSS.
///
/// Harmony is a framed protocol, not a tool-call syntax. A frame is a header
/// (`channel`, optional `to=` recipient, optional constraint) followed by a
/// payload and a terminator. These types carry that structure only — they do
/// not know about generation streams, chat history, or tool authorization.
package enum HarmonyChannel: String, Sendable, Equatable {
    case analysis
    case commentary
    case final
    /// Unrecognized channel name from a non-conforming generation.
    case other
}

/// How a Harmony frame ended.
package enum HarmonyTerminator: Sendable, Equatable {
    case end
    case call
    case `return`
    /// Generation stopped mid-frame (EOS / cancel / next header without an
    /// explicit terminator).
    case incomplete
}

/// Decoded header fields for one Harmony frame.
package struct HarmonyHeader: Sendable, Equatable {
    var channel: HarmonyChannel
    /// Raw channel text when ``channel`` is ``HarmonyChannel/other``.
    var rawChannel: String?
    /// Recipient of a directed frame, e.g. `functions.get_weather` or `browser`.
    var recipient: String?
    /// Optional constraint tag value, e.g. `json`.
    var constrain: String?
}

/// A completed Harmony frame: header + payload token ids + terminator.
package struct HarmonyFrame: Sendable, Equatable {
    var header: HarmonyHeader
    var payloadTokenIds: [Int]
    var terminator: HarmonyTerminator
}

/// Incremental outcomes from feeding one token into ``HarmonyFrameParser``.
package enum HarmonyParseStep: Sendable, Equatable {
    /// Control or header token consumed; no payload.
    case consumed
    /// Token belongs to the payload of the currently open frame.
    case payload(header: HarmonyHeader, token: Int)
    /// A frame just closed. Payload tokens were already reported via
    /// ``payload`` steps; ``frame`` carries them again for offline consumers.
    case closed(HarmonyFrame)
}
