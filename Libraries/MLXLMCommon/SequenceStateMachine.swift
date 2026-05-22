// Port of mlx_lm.generate.SequenceStateMachine.
// https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py

import Foundation

/// A trie node for matching a multi-token stop sequence.
public struct StateMachineTrieNode: Sendable {
    public var children: [Int: StateMachineTrieNode] = [:]

    /// Set on terminal nodes. Reaching this node completes the sequence
    /// and transitions to `transition.next`, or terminates if that's nil.
    public var transition: Transition?

    public struct Transition: Sendable {
        public let matchedSequence: [Int]
        public let next: String?
    }

    public init() {}
}

/// Snapshot of a single row's match progress.
public struct SequenceStateMachineState: Sendable {
    public let currentState: String?
    public let trieNode: StateMachineTrieNode?
    public let allStates: [String: StateMachineTrieNode]
    public let pendingMatch: [Int]

    public init(
        currentState: String?,
        trieNode: StateMachineTrieNode?,
        allStates: [String: StateMachineTrieNode],
        pendingMatch: [Int] = []
    ) {
        self.currentState = currentState
        self.trieNode = trieNode
        self.allStates = allStates
        self.pendingMatch = pendingMatch
    }
}

/// Per-row stop-sequence detector. Each named state holds a list of
/// `(sequence, nextState)` transitions; matching `sequence` from `state`
/// transitions to `nextState`, or terminates the row if `nextState` is nil.
///
/// A typical configuration is `["normal": [(eosTokens, nil)]]` -- one
/// terminal match on the model's EOS tokens.
public struct SequenceStateMachine: Sendable {

    public let states: [String: StateMachineTrieNode]
    public let initial: String

    public init(
        states: [String: [(sequence: [Int], next: String?)]],
        initial: String = "normal"
    ) {
        var compiled: [String: StateMachineTrieNode] = [:]
        for (name, transitions) in states {
            var root = StateMachineTrieNode()
            for (sequence, next) in transitions {
                let transition = StateMachineTrieNode.Transition(
                    matchedSequence: sequence,
                    next: next
                )
                Self.insert(
                    into: &root,
                    sequence: sequence,
                    index: 0,
                    transition: transition
                )
            }
            compiled[name] = root
        }
        self.states = compiled
        self.initial = initial
    }

    private static func insert(
        into node: inout StateMachineTrieNode,
        sequence: [Int],
        index: Int,
        transition: StateMachineTrieNode.Transition
    ) {
        if index == sequence.count {
            node.transition = transition
            return
        }
        let token = sequence[index]
        var child = node.children[token] ?? StateMachineTrieNode()
        insert(into: &child, sequence: sequence, index: index + 1, transition: transition)
        node.children[token] = child
    }

    /// An empty machine that never matches. Rows finish only on `max_tokens`.
    public init() {
        self.states = [:]
        self.initial = "normal"
    }

    public func makeState() -> SequenceStateMachineState {
        SequenceStateMachineState(
            currentState: states.isEmpty ? nil : initial,
            trieNode: states[initial],
            allStates: states,
            pendingMatch: []
        )
    }

    /// Advance the state by one token. Returns the new state, the matched
    /// sequence if a terminal node was reached on this token, and the state
    /// name after the transition (nil indicates the row terminated).
    public func match(
        _ state: SequenceStateMachineState, _ token: Int
    ) -> (
        next: SequenceStateMachineState,
        matchedSequence: [Int]?,
        currentState: String?
    ) {
        guard state.trieNode != nil, let currentState = state.currentState,
            let root = state.allStates[currentState]
        else {
            return (state, nil, state.currentState)
        }

        var candidate = state.pendingMatch + [token]
        while !candidate.isEmpty {
            if let child = Self.findPrefix(candidate, in: root) {
                if let transition = child.transition {
                    let nextState = transition.next
                    let nextNode = nextState.flatMap { state.allStates[$0] }
                    return (
                        SequenceStateMachineState(
                            currentState: nextState,
                            trieNode: nextNode,
                            allStates: state.allStates,
                            pendingMatch: []
                        ),
                        transition.matchedSequence,
                        nextState
                    )
                }
                return (
                    SequenceStateMachineState(
                        currentState: currentState,
                        trieNode: child,
                        allStates: state.allStates,
                        pendingMatch: candidate
                    ),
                    nil,
                    currentState
                )
            }

            candidate.removeFirst()
        }

        return (
            SequenceStateMachineState(
                currentState: currentState,
                trieNode: root,
                allStates: state.allStates,
                pendingMatch: []
            ),
            nil,
            currentState
        )
    }

    private static func findPrefix(
        _ tokens: [Int],
        in root: StateMachineTrieNode
    ) -> StateMachineTrieNode? {
        var node = root
        for token in tokens {
            guard let child = node.children[token] else {
                return nil
            }
            node = child
        }
        return node
    }
}
