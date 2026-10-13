// Copyright © 2026 Apple Inc.

import Foundation

/// Writes ``DecisionJSON`` the way Python's `json.dumps(value, ensure_ascii=False)` does, so
/// prompts match the reference token for token.
enum PythonJSON {
    /// `json.dumps(value, ensure_ascii=False, indent=indent)`. With an indent, items are
    /// separated by `,` and a newline; without one, by `, `. Keys are always followed by `: `.
    static func dumps(_ value: DecisionJSON, indent: Int? = nil) -> String {
        var out = ""
        write(value, indent: indent, depth: 0, into: &out)
        return out
    }

    /// Python's `repr(float)`, as `json.dumps` writes it.
    ///
    /// Swift and Python both print the shortest digits that round-trip, but Swift switches
    /// to exponent notation at 2^53 and Python at 1e16, so the digits are laid out again here.
    static func floatRepr(_ value: Double) -> String {
        if value.isNaN { return "NaN" }
        if value.isInfinite { return value < 0 ? "-Infinity" : "Infinity" }
        let sign = value.sign == .minus ? "-" : ""
        let parts = value.magnitude.description.split(separator: "e")
        let mantissa = parts[0]
        let integerDigits =
            mantissa.firstIndex(of: ".").map {
                mantissa.distance(from: mantissa.startIndex, to: $0)
            }
            ?? mantissa.count
        var digits = Array(mantissa.filter { $0 != "." })
        let leadingZeros = digits.prefix { $0 == "0" }.count
        digits.removeFirst(leadingZeros)
        while digits.last == "0" { digits.removeLast() }
        guard !digits.isEmpty else { return sign + "0.0" }

        // The value is d.ddd × 10^exponent.
        let exponent =
            integerDigits - 1 - leadingZeros + (parts.count > 1 ? Int(parts[1]) ?? 0 : 0)
        let text: String
        if exponent >= 16 || exponent < -4 {
            let fraction = digits.count > 1 ? "." + String(digits[1...]) : ""
            let power = String(abs(exponent))
            text =
                "\(digits[0])\(fraction)e\(exponent < 0 ? "-" : "+")"
                + (power.count < 2 ? "0" + power : power)
        } else if exponent < 0 {
            text = "0." + String(repeating: "0", count: -exponent - 1) + String(digits)
        } else if digits.count > exponent + 1 {
            text = String(digits[...exponent]) + "." + String(digits[(exponent + 1)...])
        } else {
            text =
                String(digits) + String(repeating: "0", count: exponent + 1 - digits.count) + ".0"
        }
        return sign + text
    }

    /// A JSON string literal with Python's `ensure_ascii=False` escapes.
    static func quoted(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case _ where scalar.value < 0x20:
                out += String(format: "\\u%04x", scalar.value)
            default:
                out.unicodeScalars.append(scalar)
            }
        }
        return out + "\""
    }

    private static func write(
        _ value: DecisionJSON, indent: Int?, depth: Int, into out: inout String
    ) {
        switch value {
        case .null: out += "null"
        case .bool(let flag): out += flag ? "true" : "false"
        case .int(let number): out += String(number)
        case .double(let number): out += floatRepr(number)
        case .string(let text): out += quoted(text)
        case .array(let items):
            writeContainer(
                open: "[", close: "]", count: items.count, indent: indent, depth: depth, into: &out
            ) { index, out in
                write(items[index], indent: indent, depth: depth + 1, into: &out)
            }
        case .object(let members):
            writeContainer(
                open: "{", close: "}", count: members.count, indent: indent, depth: depth,
                into: &out
            ) { index, out in
                out += quoted(members[index].key) + ": "
                write(members[index].value, indent: indent, depth: depth + 1, into: &out)
            }
        }
    }

    private static func writeContainer(
        open: String, close: String, count: Int, indent: Int?, depth: Int,
        into out: inout String, item: (Int, inout String) -> Void
    ) {
        guard count > 0 else {
            out += open + close
            return
        }
        out += open
        if let indent {
            let inner = "\n" + String(repeating: " ", count: indent * (depth + 1))
            for index in 0 ..< count {
                out += (index == 0 ? "" : ",") + inner
                item(index, &out)
            }
            out += "\n" + String(repeating: " ", count: indent * depth) + close
        } else {
            for index in 0 ..< count {
                if index > 0 { out += ", " }
                item(index, &out)
            }
            out += close
        }
    }
}
