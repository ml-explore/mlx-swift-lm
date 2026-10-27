// Copyright © 2026 Apple Inc.

import Foundation
import MLXScriptedLM
import Testing

struct TurnScriptsTests {

    static let marker = 1

    @Test func `positions before the first marker have no cursor`() {
        let scripts = TurnScripts(scripts: [[5, 6, 7]], turnMarker: [Self.marker])
        let cursors = scripts.cursors(for: [9, 8, Self.marker, 5, 6])
        #expect(cursors[0] == nil)
        #expect(cursors[1] == nil)
        #expect(cursors[2] == .init(turn: 0, index: 0))
        #expect(cursors[3] == .init(turn: 0, index: 1))
        #expect(cursors[4] == .init(turn: 0, index: 2))
    }

    @Test func `the script is indexed by position, not content`() {
        let scripts = TurnScripts(scripts: [[5, 6, 7]], turnMarker: [Self.marker])
        let cursors = scripts.cursors(for: [Self.marker, 99, 98, 97])
        #expect(cursors[1] == .init(turn: 0, index: 1))
        #expect(cursors[2] == .init(turn: 0, index: 2))
        #expect(scripts.token(at: cursors[2]!) == 7)
        #expect(scripts.token(at: cursors[3]!) == nil)
    }

    @Test func `rows after the script stay in the turn until the next marker`() {
        let scripts = TurnScripts(scripts: [[5, 6]], turnMarker: [Self.marker])
        let cursors = scripts.cursors(for: [Self.marker, 5, 6, 40, 41])
        #expect(cursors[2] == .init(turn: 0, index: 2))
        #expect(cursors[3] == .init(turn: 0, index: 2))
        #expect(cursors[4] == .init(turn: 0, index: 2))
        #expect(scripts.token(at: cursors[4]!) == nil)
    }

    @Test func `a script that ends with EOS ends its turn`() {
        let eos = 3
        let scripts = TurnScripts(
            scripts: [[5, eos]], turnMarker: [Self.marker], endOfSequence: eos)
        let cursors = scripts.cursors(for: [Self.marker, 5, eos, 40, 41])
        #expect(cursors[2] == .init(turn: 0, index: 2))
        #expect(cursors[3] == nil)
        #expect(cursors[4] == nil)
        #expect(scripts.blockedUnscriptedTokens == [Self.marker, eos])
    }

    @Test func `each marker starts the next turn`() {
        let scripts = TurnScripts(scripts: [[5, 6], [7, 8]], turnMarker: [Self.marker])
        let cursors = scripts.cursors(for: [9, Self.marker, 5, 6, 9, 9, Self.marker, 7])
        #expect(cursors[6] == .init(turn: 1, index: 0))
        #expect(cursors[7] == .init(turn: 1, index: 1))
    }

    @Test func `a marker inside a turn starts a new turn when history is shorter than the script`()
    {
        // History renders turn 0 as one token, though its script has four.
        let scripts = TurnScripts(scripts: [[5, 6, 7, 8], [20]], turnMarker: [Self.marker])
        let cursors = scripts.cursors(for: [Self.marker, 5, Self.marker])
        #expect(cursors[2] == .init(turn: 1, index: 0))
    }

    @Test func `a marker the script contains does not start a turn`() {
        let scripts = TurnScripts(scripts: [[5, Self.marker, 6]], turnMarker: [Self.marker])
        let cursors = scripts.cursors(for: [Self.marker, 5, Self.marker, 6])
        #expect(cursors[2] == .init(turn: 0, index: 2))
        #expect(cursors[3] == .init(turn: 0, index: 3))
    }

    @Test func `a multi-token marker must match in full`() {
        let scripts = TurnScripts(scripts: [[5]], turnMarker: [1, 2])
        let cursors = scripts.cursors(for: [1, 3, 1, 2])
        #expect(cursors[1] == nil)
        #expect(cursors[2] == nil)
        #expect(cursors[3] == .init(turn: 0, index: 0))
    }

    @Test func `turns past the scripts have no token without a stop token`() {
        let scripts = TurnScripts(scripts: [[5]], turnMarker: [Self.marker])
        let cursors = scripts.cursors(for: [Self.marker, 5, Self.marker])
        #expect(cursors[2] == .init(turn: 1, index: 0))
        #expect(scripts.token(at: cursors[2]!) == nil)
    }

    @Test func `turns past the scripts emit EOS and end`() {
        let eos = 3
        let scripts = TurnScripts(
            scripts: [[5, eos]], turnMarker: [Self.marker], endOfSequence: eos)
        let cursors = scripts.cursors(for: [Self.marker, 5, eos, Self.marker, eos, 40])
        #expect(cursors[3] == .init(turn: 1, index: 0))
        #expect(scripts.token(at: cursors[3]!) == eos)
        #expect(cursors[5] == nil)
    }

    @Test func `stop tokens are blocked and can end a turn past the scripts`() {
        let stop = 4
        let scripts = TurnScripts(scripts: [[5]], turnMarker: [Self.marker], stopTokens: [stop])
        #expect(scripts.blockedUnscriptedTokens == [Self.marker, stop])
        let cursors = scripts.cursors(for: [Self.marker, 5, Self.marker])
        #expect(scripts.token(at: cursors[2]!) == stop)
    }

    // MARK: - Prompt endings

    /// User marker 2, user texts 10, 11, 12, then the turn marker.
    static let user = 2
    static func ending(_ text: Int) -> [Int] { [user, text, marker] }

    static let threeTurns = TurnScripts(
        scripts: [[5], [6], [7]], turnMarker: [marker],
        promptEndings: [ending(10), ending(11), ending(12)])

    @Test func `a prompt ending picks the turn after a dropped turn`() {
        // ChatSession dropped turn 1 and its user message 11.
        let cursors = Self.threeTurns.cursors(for: Self.ending(10) + [5] + Self.ending(12))
        #expect(cursors.last == .init(turn: 2, index: 0))
    }

    @Test func `a retried prompt replays its turn`() {
        let cursors = Self.threeTurns.cursors(for: Self.ending(10) + [5] + Self.ending(11))
        #expect(cursors.last == .init(turn: 1, index: 0))
    }

    @Test func `an extra assistant message in history does not shift the turn`() {
        let tokens = Self.ending(10) + [5] + [Self.marker, 9] + Self.ending(11)
        let cursors = Self.threeTurns.cursors(for: tokens)
        #expect(cursors.last == .init(turn: 1, index: 0))
    }

    @Test func `repeated user text resolves by order`() {
        let scripts = TurnScripts(
            scripts: [[5], [6]], turnMarker: [Self.marker],
            promptEndings: [Self.ending(10), Self.ending(10)])
        #expect(scripts.cursors(for: Self.ending(10)).last == .init(turn: 0, index: 0))
        let second = scripts.cursors(for: Self.ending(10) + [5] + Self.ending(10))
        #expect(second.last == .init(turn: 1, index: 0))
    }

    @Test func `a prompt with no matching ending uses order`() {
        let cursors = Self.threeTurns.cursors(for: Self.ending(10) + [5] + Self.ending(99))
        #expect(cursors.last == .init(turn: 1, index: 0))
    }
}
