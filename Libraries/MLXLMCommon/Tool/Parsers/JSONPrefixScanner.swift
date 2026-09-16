// Copyright © 2026 Apple Inc.

/// Incrementally recognizes one JSON value without allocating decoded values.
/// Used for context shielding: an impossible prefix is prose, an incomplete
/// valid prefix is still data. Each byte is examined once, including in strings.
struct JSONPrefixScanner {
    enum Result {
        case incomplete, invalid, depthLimit
        case complete(byteCount: Int)
    }

    private enum Expectation {
        case value, arrayValueOrEnd, keyOrEnd, key, colon
        case arraySeparator, objectSeparator, end
    }

    private enum Token {
        case none
        case string(key: Bool)
        case escape(key: Bool)
        case unicode(remaining: Int, key: Bool)
        case literal(remaining: ArraySlice<UInt8>)
        case number(NumberState)
    }

    private enum NumberState {
        case sign, zero, integer, point, fraction, exponent, exponentSign, exponentDigits

        var canEnd: Bool {
            switch self {
            case .zero, .integer, .fraction, .exponentDigits: true
            default: false
            }
        }

        func next(_ byte: UInt8) -> Self? {
            switch (self, byte) {
            case (.sign, 48): .zero
            case (.sign, 49 ... 57), (.integer, 48 ... 57): .integer
            case (.zero, 46), (.integer, 46): .point
            case (.point, 48 ... 57), (.fraction, 48 ... 57): .fraction
            case (.zero, 69), (.zero, 101), (.integer, 69), (.integer, 101),
                (.fraction, 69), (.fraction, 101):
                .exponent
            case (.exponent, 43), (.exponent, 45): .exponentSign
            case (.exponent, 48 ... 57), (.exponentSign, 48 ... 57),
                (.exponentDigits, 48 ... 57):
                .exponentDigits
            default: nil
            }
        }
    }

    private var expectation: Expectation = .value
    private var parents: [Expectation] = []
    private var token: Token = .none
    private var consumed = 0

    mutating func scan(_ text: some StringProtocol) -> Result {
        let bytes = text.utf8
        var index = bytes.index(bytes.startIndex, offsetBy: consumed)
        while index < bytes.endIndex {
            let byte = bytes[index]
            switch token {
            case .string(let key):
                switch byte {
                case 34:
                    token = .none
                    if key { expectation = .colon } else { completeValue() }
                case 92: token = .escape(key: key)
                case 0 ..< 32: return .invalid
                default: break
                }
            case .escape(let key):
                switch byte {
                case 34, 47, 92, 98, 102, 110, 114, 116: token = .string(key: key)
                case 117: token = .unicode(remaining: 4, key: key)
                default: return .invalid
                }
            case .unicode(let remaining, let key):
                guard
                    (48 ... 57).contains(byte) || (65 ... 70).contains(byte)
                        || (97 ... 102).contains(byte)
                else { return .invalid }
                token =
                    remaining == 1
                    ? .string(key: key) : .unicode(remaining: remaining - 1, key: key)
            case .literal(let remaining):
                guard byte == remaining.first else { return .invalid }
                if remaining.count == 1 {
                    token = .none
                    completeValue()
                } else {
                    token = .literal(remaining: remaining.dropFirst())
                }
            case .number(let state):
                if let next = state.next(byte) {
                    token = .number(next)
                } else {
                    guard state.canEnd else { return .invalid }
                    token = .none
                    completeValue()
                    // The delimiter belongs to the surrounding container.
                    continue
                }
            case .none:
                if expectation == .end { return .complete(byteCount: consumed) }
                if [9, 10, 13, 32].contains(byte) {
                    break
                }
                switch expectation {
                case .value, .arrayValueOrEnd:
                    if expectation == .arrayValueOrEnd, byte == 93 {
                        closeContainer()
                    } else {
                        switch byte {
                        case 123, 91:
                            guard parents.count < 256 else { return .depthLimit }
                            parents.append(byte == 123 ? .objectSeparator : .arraySeparator)
                            expectation = byte == 123 ? .keyOrEnd : .arrayValueOrEnd
                        case 34: token = .string(key: false)
                        case 116: token = .literal(remaining: [114, 117, 101][...])
                        case 102: token = .literal(remaining: [97, 108, 115, 101][...])
                        case 110: token = .literal(remaining: [117, 108, 108][...])
                        case 45: token = .number(.sign)
                        case 48: token = .number(.zero)
                        case 49 ... 57: token = .number(.integer)
                        default: return .invalid
                        }
                    }
                case .key, .keyOrEnd:
                    if expectation == .keyOrEnd, byte == 125 {
                        closeContainer()
                    } else {
                        guard byte == 34 else { return .invalid }
                        token = .string(key: true)
                    }
                case .colon:
                    guard byte == 58 else { return .invalid }
                    expectation = .value
                case .arraySeparator:
                    if byte == 93 {
                        closeContainer()
                    } else if byte == 44 {
                        expectation = .value
                    } else {
                        return .invalid
                    }
                case .objectSeparator:
                    if byte == 125 {
                        closeContainer()
                    } else if byte == 44 {
                        expectation = .key
                    } else {
                        return .invalid
                    }
                case .end: break
                }
            }
            bytes.formIndex(after: &index)
            consumed += 1
            if expectation == .end, case .none = token {
                return .complete(byteCount: consumed)
            }
        }
        return .incomplete
    }

    private mutating func completeValue() {
        // A key's value follows a colon; array values have no key. The
        // container kind is represented by the continuation on the stack.
        expectation = parents.last ?? .end
    }

    private mutating func closeContainer() {
        parents.removeLast()
        completeValue()
    }
}
