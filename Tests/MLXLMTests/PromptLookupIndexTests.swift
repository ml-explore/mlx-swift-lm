// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXLMCommon

private func lookupReference(
    _ history: [Int], maximumTokens: Int, maximumOrder: Int,
    minimumOrder: Int = 1, minimumOccurrences: Int = 1, minimumConfidence: Float = 0
) -> [Int] {
    var result = [Int]()
    var suffix = history
    for _ in 0 ..< maximumTokens {
        var candidate: Int?
        for order in stride(from: min(maximumOrder, suffix.count), through: minimumOrder, by: -1) {
            guard history.count > order else { continue }
            let key = suffix.suffix(order)
            var counts = [Int: Int]()
            var total = 0
            for offset in 0 ..< history.count - order {
                if history[offset ..< offset + order].elementsEqual(key) {
                    counts[history[offset + order], default: 0] += 1
                    total += 1
                }
            }
            let best = counts.max {
                $0.value != $1.value ? $0.value < $1.value : $0.key > $1.key
            }
            if let best, best.value >= minimumOccurrences,
                Float(best.value) >= minimumConfidence * Float(total)
            {
                candidate = best.key
                break
            }
        }
        guard let candidate else { break }
        result.append(candidate)
        suffix.append(candidate)
    }
    return result
}

private struct LookupRandom {
    var state: UInt64 = 0x51ad_794e_d44d_e600
    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state ^ (state >> 29)
    }
}

struct PromptLookupIndexTests {
    @Test func longestContextAndStableFrequencyTies() {
        let index = PromptLookupIndex(capacity: 64)
        index.append(contentsOf: [1, 2, 9, 1, 2, 7, 1, 2])
        var draft = [Int]()
        index.draft(maximumTokens: 1, minimumOccurrences: 1, minimumConfidence: 0, into: &draft)
        #expect(draft == [7])
        index.draft(maximumTokens: 8, minimumOccurrences: 1, minimumConfidence: 0.6, into: &draft)
        #expect(draft.isEmpty)
    }

    @Test func draftDoesNotTeachItsOwnGuesses() {
        let history = [1, 2, 3, 1, 2]
        let index = PromptLookupIndex(capacity: 32)
        index.append(contentsOf: history)
        var draft = [Int]()
        for _ in 0 ..< 10 {
            index.draft(
                maximumTokens: 20, minimumOccurrences: 1, minimumConfidence: 0, into: &draft)
            #expect(draft == lookupReference(history, maximumTokens: 20, maximumOrder: 4))
            #expect(index.count == history.count)
        }
    }

    @Test func evictionRemovesExpiredWinnersAndCounts() {
        let index = PromptLookupIndex(capacity: 8, maximumOrder: 1)
        var history = [Int]()
        var draft = [Int]()
        for token in [1, 2, 1, 2, 1, 3, 1, 3, 1, 3, 1, 4, 1] {
            index.append(token)
            history.append(token)
            history = Array(history.suffix(8))
            index.draft(maximumTokens: 4, minimumOccurrences: 1, minimumConfidence: 0, into: &draft)
            #expect(draft == lookupReference(history, maximumTokens: 4, maximumOrder: 1))
        }
    }

    @Test func bulkAppendRetainsOnlyTheFinalWindow() {
        let index = PromptLookupIndex(capacity: 8)
        index.append(contentsOf: [8, 9, 8])
        let history = (0 ..< 1_000).map { $0 % 7 }
        index.append(contentsOf: history)
        var draft = [Int]()
        index.draft(maximumTokens: 12, minimumOccurrences: 1, minimumConfidence: 0, into: &draft)
        #expect(
            draft == lookupReference(Array(history.suffix(8)), maximumTokens: 12, maximumOrder: 4))
        #expect(index.count == 8)
    }

    @Test func fullWidthTokenIDsRemainDistinct() {
        let history = [Int.min, Int.max, 0, Int.min, Int.max, 0, 1 << 32, 0, Int.min, Int.max]
        let index = PromptLookupIndex(capacity: 64)
        index.append(contentsOf: history)
        var draft = [Int]()
        index.draft(maximumTokens: 12, minimumOccurrences: 1, minimumConfidence: 0, into: &draft)
        #expect(draft == lookupReference(history, maximumTokens: 12, maximumOrder: 4))
    }

    @Test func resetAndEmptyDraftReuse() {
        let index = PromptLookupIndex(capacity: 8)
        var draft = [1, 2, 3]
        index.append(contentsOf: [1, 2, 1, 2, 1])
        let bytes = index.allocatedBytes
        index.reset()
        index.draft(maximumTokens: 3, minimumOccurrences: 1, minimumConfidence: 0, into: &draft)
        #expect(draft.isEmpty)
        #expect(index.count == 0)
        index.append(contentsOf: [4, 5, 4])
        index.draft(maximumTokens: 2, minimumOccurrences: 1, minimumConfidence: 0, into: &draft)
        #expect(draft == [5, 4])
        index.draft(maximumTokens: 0, minimumOccurrences: 1, minimumConfidence: 0, into: &draft)
        #expect(draft.isEmpty)
        #expect(index.allocatedBytes == bytes)
    }

    @Test func randomDifferentialWithEvictionAndThresholds() {
        var random = LookupRandom()
        var draft = [Int]()
        for capacity in [2, 3, 4, 5, 8, 17, 64] {
            for order in 1 ... 4 {
                let index = PromptLookupIndex(capacity: capacity, maximumOrder: order)
                var history = [Int]()
                for step in 0 ..< 500 {
                    let token = Int(random.next() % (step % 3 == 0 ? 3 : 31)) - 3
                    index.append(token)
                    history.append(token)
                    history = Array(history.suffix(capacity))
                    let minimumOrder = Int(random.next() % UInt64(order)) + 1
                    let occurrences = Int(random.next() % 3) + 1
                    let confidence = Float(random.next() % 5) / 4
                    index.draft(
                        maximumTokens: 8, minimumOrder: minimumOrder,
                        minimumOccurrences: occurrences, minimumConfidence: confidence, into: &draft
                    )
                    #expect(
                        draft
                            == lookupReference(
                                history, maximumTokens: 8, maximumOrder: order,
                                minimumOrder: minimumOrder, minimumOccurrences: occurrences,
                                minimumConfidence: confidence))
                    #expect(index.count == history.count)
                }
            }
        }
    }
}
