// Copyright © 2026 Apple Inc.

import Foundation

/// The tokens a scripted model emits on each assistant turn, and how it finds the turn.
///
/// A turn starts after each occurrence of ``turnMarker`` (the template's generation
/// prompt, such as `<|assistant|>`). Within a turn the script is indexed by position, not
/// content: the token after position `start + k` is `script[k + 1]`, whatever the input.
/// So a wrong speculative draft does not move the cursor, and a model with a different
/// script (a draft model) keeps following its own.
///
/// Each new turn is the one after the previous turn, unless the prompt ends with one of
/// ``promptEndings``. `ChatSession` drops an empty, cancelled, or rejected turn together
/// with the user message before it, so order alone would replay the dropped turn's script.
/// When several turns match, the first one at or after the next turn wins, otherwise the
/// last one.
///
/// A turn past the last script emits ``endOfSequence`` (or a stop token), so an extra turn
/// ends at once with an empty reply instead of running to `maxTokens`.
///
/// A marker inside the current turn starts a new turn, unless the script itself has the
/// marker at that position. That handles history that renders a turn shorter than its
/// script (reasoning stripped, for example) and scripts that contain the marker.
public struct TurnScripts: Sendable, Equatable {

    /// Where the next token comes from: `scripts[turn][index]`. An `index` equal to the
    /// script's length means the script is used up.
    public struct Cursor: Sendable, Equatable {
        public let turn: Int
        public let index: Int

        public init(turn: Int, index: Int) {
            self.turn = turn
            self.index = index
        }
    }

    /// The tokens each assistant turn emits, in order.
    public let scripts: [[Int]]

    /// The tokens that open an assistant turn.
    public let turnMarker: [Int]

    /// For each script, the tokens that end its prompt (the user message and the
    /// generation prompt), or `nil` to place the turn by order alone.
    public let promptEndings: [[Int]?]

    /// The end-of-sequence token. A turn whose script ends with it, or with a stop token, is
    /// over after the script.
    public let endOfSequence: Int?

    /// Other token ids that stop generation, such as a model configuration's `eosTokenIds`.
    public let stopTokens: Set<Int>

    /// Tokens the model must not emit inside a turn where the script has no token: the stop
    /// tokens and the first marker token. Either one would end the turn early or start the
    /// next one. Stop strings are not covered.
    public var blockedUnscriptedTokens: Set<Int> {
        stopTokens.union([turnMarker[0], endOfSequence].compactMap { $0 })
    }

    public init(
        scripts: [[Int]], turnMarker: [Int], promptEndings: [[Int]?]? = nil,
        endOfSequence: Int? = nil, stopTokens: Set<Int> = []
    ) {
        precondition(!turnMarker.isEmpty, "turnMarker must not be empty")
        let promptEndings = promptEndings ?? Array(repeating: nil, count: scripts.count)
        precondition(
            promptEndings.count == scripts.count, "promptEndings needs one entry per script")
        self.scripts = scripts
        self.turnMarker = turnMarker
        self.promptEndings = promptEndings
        self.endOfSequence = endOfSequence
        self.stopTokens = stopTokens
    }

    /// The scripted token at `cursor`, or `nil` when the turn has no script or it is used up.
    public func token(at cursor: Cursor) -> Int? {
        let script = script(for: cursor.turn)
        return script.indices.contains(cursor.index) ? script[cursor.index] : nil
    }

    /// The turn's script, or a single stop token for a turn past the last script.
    private func script(for turn: Int) -> [Int] {
        if scripts.indices.contains(turn) { return scripts[turn] }
        guard turn >= 0, let stop = endOfSequence ?? stopTokens.min() else { return [] }
        return [stop]
    }

    /// For each position in `tokens`, the cursor for the token that follows it, or `nil`
    /// outside an assistant turn.
    public func cursors(for tokens: [Int]) -> [Cursor?] {
        var result = [Cursor?](repeating: nil, count: tokens.count)
        var turn = -1
        var start: Int?
        var i = 0

        while i < tokens.count {
            let script = script(for: turn)

            // The script position of token i, while inside a scripted turn.
            var k: Int?
            if let start, i - start < script.count {
                k = i - start
            }

            let markerIsScripted = k.map { hasMarker(in: script, at: $0) } ?? false
            if hasMarker(in: tokens, at: i), !markerIsScripted {
                let end = i + turnMarker.count
                turn = nextTurn(after: turn, prompt: tokens[..<end])
                result[end - 1] = Cursor(turn: turn, index: 0)
                start = end
                i = end
                continue
            }

            if let k {
                result[i] = Cursor(turn: turn, index: k + 1)
            } else if start != nil {
                if let last = script.last, last == endOfSequence || stopTokens.contains(last) {
                    start = nil
                } else {
                    // Unscripted tokens after the script, until the next marker.
                    result[i] = Cursor(turn: turn, index: script.count)
                }
            }
            i += 1
        }
        return result
    }

    private func nextTurn(after turn: Int, prompt: ArraySlice<Int>) -> Int {
        let matches = promptEndings.indices.filter { index in
            guard let ending = promptEndings[index] else { return false }
            return prompt.count >= ending.count && prompt.suffix(ending.count).elementsEqual(ending)
        }
        return matches.first { $0 > turn } ?? matches.last ?? turn + 1
    }

    private func hasMarker(in tokens: [Int], at index: Int) -> Bool {
        guard index >= 0, index + turnMarker.count <= tokens.count else { return false }
        return tokens[index ..< index + turnMarker.count].elementsEqual(turnMarker)
    }
}
