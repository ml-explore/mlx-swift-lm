// Copyright © 2026 Apple Inc.

/// Incrementally removes configured stop strings from streamed text.
///
/// The filter retains only a suffix that could still become a stop string, so
/// callers can stream all text known to be safe without exposing a partial
/// terminator.
struct StopStringFilter {
    let stopStrings: [String]
    var buffer = ""
    var stopped = false

    init(stopStrings: Set<String>) {
        self.stopStrings = stopStrings.filter { !$0.isEmpty }.sorted {
            if $0.count == $1.count {
                return $0 < $1
            }
            return $0.count > $1.count
        }
    }

    var isEnabled: Bool {
        !stopStrings.isEmpty
    }

    mutating func process(_ chunk: String) -> (text: String?, stopped: Bool) {
        guard !stopped else {
            return (nil, true)
        }
        guard isEnabled else {
            return (chunk.isEmpty ? nil : chunk, false)
        }

        buffer += chunk

        if let stopRange = earliestStopRange(in: buffer) {
            let text = String(buffer[..<stopRange.lowerBound])
            buffer = ""
            stopped = true
            return (text.isEmpty ? nil : text, true)
        }

        let suffixLength = longestStopPrefixSuffixLength(in: buffer)
        let emitEnd = buffer.index(buffer.endIndex, offsetBy: -suffixLength)
        let text = String(buffer[..<emitEnd])
        buffer = String(buffer[emitEnd...])
        return (text.isEmpty ? nil : text, false)
    }

    mutating func finish() -> String? {
        guard isEnabled, !stopped, !buffer.isEmpty else {
            return nil
        }
        let text = buffer
        buffer = ""
        return text
    }

    private func earliestStopRange(in text: String) -> Range<String.Index>? {
        var earliest: Range<String.Index>?
        for stopString in stopStrings {
            guard let range = text.range(of: stopString) else {
                continue
            }
            if let current = earliest {
                if range.lowerBound < current.lowerBound {
                    earliest = range
                }
            } else {
                earliest = range
            }
        }
        return earliest
    }

    private func longestStopPrefixSuffixLength(in text: String) -> Int {
        var longest = 0
        for stopString in stopStrings {
            let maxLength = Swift.min(text.count, stopString.count - 1)
            guard maxLength > longest else {
                continue
            }
            for length in stride(from: maxLength, through: longest + 1, by: -1) {
                if text.suffix(length) == stopString.prefix(length) {
                    longest = length
                    break
                }
            }
        }
        return longest
    }
}
