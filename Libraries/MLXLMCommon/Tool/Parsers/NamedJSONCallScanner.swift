// Copyright © 2026 Apple Inc.

/// Recognizer for the markerless `name\n{json}` tool-call dialect.
///
/// GLM-4-0414 writes a call as the function name on its own line followed by
/// the JSON arguments object, with no delimiter tokens at all. The tool schema
/// is the only anchor: a candidate is a declared name that begins a line,
/// optional whitespace, then a balanced JSON object. Ordinary `word\n{...}`
/// text never matches because `word` is not a declared tool.
///
/// Every query is a single UTF-8 walk with no allocation, so the streaming
/// lexer can afford to ask at each line start of each chunk.
///
/// Reference: `process_response` in Zhipu's serving demo, which accepts a call
/// only when the first line is a declared tool and the second opens with `{`:
/// https://github.com/zai-org/GLM-4/blob/e3e6de52c45290291f984cfe934839d0954a17ef/basic_demo/glm_server.py
struct NamedJSONCallScanner: Sendable {
    enum Scan: Equatable {
        /// A declared name and the complete arguments object that follows it.
        case complete(name: Substring, arguments: Substring)
        /// A declared name whose arguments object is still open.
        case openArguments(name: Substring)
        /// A prefix of a declared name, or a whole name still awaiting `{`.
        case prefix
        /// The text cannot become a candidate.
        case mismatch
    }

    private enum Arguments {
        case complete(Substring)
        case open
        case awaitingBrace
        case mismatch
    }

    /// Byte trie over the declared names. One walk answers both "is this a
    /// declared name" and "can this still become one", in time proportional
    /// to the name rather than to the number of tools.
    private struct NameTrie: Sendable {
        enum Match: Equatable {
            case name(end: String.Index)
            case prefix
            case mismatch
        }

        private var children: [[UInt8: Int]] = [[:]]
        private var terminal: [Bool] = [false]
        private(set) var maximumNameLength = 0

        init(_ names: Set<String>) {
            for name in names where !name.isEmpty {
                insert(name)
            }
        }

        var isEmpty: Bool { children.count == 1 }

        /// Follows `text` until a separator closes the name or the input ends.
        func match(_ text: Substring) -> Match {
            let bytes = text.utf8
            var node = 0
            var index = bytes.startIndex
            while index < bytes.endIndex {
                let byte = bytes[index]
                if isNameSeparator(byte) {
                    return terminal[node] ? .name(end: index) : .mismatch
                }
                guard let next = children[node][byte] else { return .mismatch }
                node = next
                bytes.formIndex(after: &index)
            }
            return .prefix
        }

        private mutating func insert(_ name: String) {
            var node = 0
            for byte in name.utf8 {
                if let next = children[node][byte] {
                    node = next
                } else {
                    children.append([:])
                    terminal.append(false)
                    children[node][byte] = children.count - 1
                    node = children.count - 1
                }
            }
            terminal[node] = true
            maximumNameLength = max(maximumNameLength, name.utf8.count)
        }
    }

    /// Whitespace allowed between the name and its arguments. The template
    /// renders one space and a newline; the bound keeps an unterminated
    /// candidate from pinning the stream.
    private static let maximumSeparatorLength = 8

    private let names: NameTrie

    /// Longest text, in UTF-8 bytes, that can still be a name awaiting its
    /// arguments.
    var maximumPrefixLength: Int { names.maximumNameLength + Self.maximumSeparatorLength }

    init?(toolNames: Set<String>) {
        let names = NameTrie(toolNames)
        guard !names.isEmpty else { return nil }
        self.names = names
    }

    /// Classifies text that begins at a line start.
    func scan(_ text: Substring) -> Scan {
        switch names.match(text) {
        case .prefix: return .prefix
        case .mismatch: return .mismatch
        case .name(let end):
            let name = text[..<end]
            switch Self.arguments(after: end, in: text) {
            case .complete(let arguments): return .complete(name: name, arguments: arguments)
            case .open: return .openArguments(name: name)
            case .awaitingBrace: return .prefix
            case .mismatch: return .mismatch
            }
        }
    }

    /// Splits a payload that is exactly one `name\n{json}` call, leaving the
    /// caller to anchor the name. Trailing whitespace is tolerated; any other
    /// trailing text disqualifies the payload.
    static func split(_ text: Substring) -> (name: Substring, arguments: Substring)? {
        let bytes = text.utf8
        guard let nameEnd = bytes.firstIndex(where: isNameSeparator), nameEnd > bytes.startIndex,
            case .complete(let arguments) = arguments(after: nameEnd, in: text),
            bytes[arguments.endIndex...].allSatisfy(isWhitespace)
        else { return nil }
        return (text[..<nameEnd], arguments)
    }

    private static func arguments(after nameEnd: String.Index, in text: Substring) -> Arguments {
        let bytes = text.utf8
        var brace = nameEnd
        var separatorLength = 0
        while brace < bytes.endIndex, isWhitespace(bytes[brace]) {
            separatorLength += 1
            guard separatorLength <= maximumSeparatorLength else { return .mismatch }
            bytes.formIndex(after: &brace)
        }
        guard brace < bytes.endIndex else { return .awaitingBrace }
        guard bytes[brace] == UInt8(ascii: "{") else { return .mismatch }

        var json = JSONPrefixScanner()
        switch json.scan(text[brace...]) {
        case .complete(let byteCount):
            return .complete(text[brace ..< bytes.index(brace, offsetBy: byteCount)])
        case .incomplete:
            return .open
        case .invalid, .depthLimit:
            return .mismatch
        }
    }
}

/// ASCII whitespace, as JSON defines it.
private func isWhitespace(_ byte: UInt8) -> Bool {
    switch byte {
    case 9, 10, 13, 32: true
    default: false
    }
}

/// Whitespace or `{` ends a name.
private func isNameSeparator(_ byte: UInt8) -> Bool {
    byte == UInt8(ascii: "{") || isWhitespace(byte)
}
