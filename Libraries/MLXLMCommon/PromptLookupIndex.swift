// Copyright © 2026 Apple Inc.

/// A bounded frequency index over the observed token stream. The generation iterator owns it;
/// speculative drafts never enter the index until the target accepts them.
final class PromptLookupIndex {
    private struct Key: Equatable {
        var a = 0
        var b = 0
        var c = 0
        var d = 0
        var length = 0

        mutating func extend(_ token: Int) {
            switch length {
            case 0: a = token
            case 1: b = token
            case 2: c = token
            default: d = token
            }
            length += 1
        }

        mutating func append(_ token: Int) {
            d = c
            c = b
            b = a
            a = token
            length = min(length + 1, 4)
        }

        func prefix(_ count: Int) -> Key {
            Key(
                a: a, b: count > 1 ? b : 0, c: count > 2 ? c : 0,
                d: count > 3 ? d : 0, length: count)
        }

        var hash: UInt64 {
            var value = UInt64(length)
            value = PromptLookupIndex.mix(value ^ UInt64(truncatingIfNeeded: a))
            if length > 1 { value = PromptLookupIndex.mix(value ^ UInt64(truncatingIfNeeded: b)) }
            if length > 2 { value = PromptLookupIndex.mix(value ^ UInt64(truncatingIfNeeded: c)) }
            if length > 3 { value = PromptLookupIndex.mix(value ^ UInt64(truncatingIfNeeded: d)) }
            return value
        }
    }

    private struct Context {
        var key = Key()
        var total: UInt32 = 0
        var head: Int32 = -1
        var winner: Int32 = -1
        var upperBound: UInt32 = 0
    }

    private struct Continuation {
        var token = 0
        var count: UInt32 = 0
        var context: Int32 = -1
        var next: Int32 = -1
        var previous: Int32 = -1
    }

    let capacity: Int
    let maximumOrder: Int
    private(set) var count = 0
    private let tokens: UnsafeMutablePointer<Int>
    private var first = 0
    private let contexts: UnsafeMutablePointer<Context>
    private let continuations: UnsafeMutablePointer<Continuation>
    private let contextSlots: UnsafeMutablePointer<Int32>
    private let continuationSlots: UnsafeMutablePointer<Int32>
    private let entryCapacity: Int
    private let slotCount: Int
    private let mask: Int
    private var contextHighWater = 0
    private var continuationHighWater = 0
    private var freeContext: Int32 = -1
    private var freeContinuation: Int32 = -1

    /// Includes reserved entries, table slots and the token ring, excluding allocator overhead.
    var allocatedBytes: Int {
        capacity * MemoryLayout<Int>.stride
            + entryCapacity * (MemoryLayout<Context>.stride + MemoryLayout<Continuation>.stride)
            + slotCount * MemoryLayout<Int32>.stride * 2
    }

    init(capacity: Int, maximumOrder: Int = 4) {
        precondition(capacity >= 2 && capacity <= Int(Int32.max) / 8)
        precondition((1 ... 4).contains(maximumOrder))
        self.capacity = capacity
        self.maximumOrder = maximumOrder
        entryCapacity = capacity * maximumOrder
        var slots = 16
        while slots < entryCapacity * 2 { slots <<= 1 }
        slotCount = slots
        mask = slots - 1
        tokens = .allocate(capacity: capacity)
        tokens.initialize(repeating: 0, count: capacity)
        contexts = .allocate(capacity: entryCapacity)
        contexts.initialize(repeating: Context(), count: entryCapacity)
        continuations = .allocate(capacity: entryCapacity)
        continuations.initialize(repeating: Continuation(), count: entryCapacity)
        contextSlots = .allocate(capacity: slots)
        contextSlots.initialize(repeating: 0, count: slots)
        continuationSlots = .allocate(capacity: slots)
        continuationSlots.initialize(repeating: 0, count: slots)
    }

    deinit {
        tokens.deinitialize(count: capacity)
        tokens.deallocate()
        contexts.deinitialize(count: entryCapacity)
        contexts.deallocate()
        continuations.deinitialize(count: entryCapacity)
        continuations.deallocate()
        contextSlots.deinitialize(count: slotCount)
        contextSlots.deallocate()
        continuationSlots.deinitialize(count: slotCount)
        continuationSlots.deallocate()
    }

    func reset() {
        contextSlots.update(repeating: 0, count: slotCount)
        continuationSlots.update(repeating: 0, count: slotCount)
        count = 0
        first = 0
        contextHighWater = 0
        continuationHighWater = 0
        freeContext = -1
        freeContinuation = -1
    }

    func append(contentsOf tokens: [Int]) {
        if tokens.count >= capacity {
            reset()
            for token in tokens.suffix(capacity) { append(token) }
        } else {
            for token in tokens { append(token) }
        }
    }

    func append(_ token: Int) {
        if count == capacity {
            for order in 1 ... min(maximumOrder, count - 1) {
                var key = Key()
                for offset in (0 ..< order).reversed() { key.extend(self.token(at: offset)) }
                remove(key, token: self.token(at: order))
            }
            first = (first + 1) % capacity
            count -= 1
        }

        var key = Key()
        if count > 0 {
            for order in 1 ... min(maximumOrder, count) {
                key.extend(self.token(at: count - order))
                insert(key, token: token)
            }
        }
        tokens[(first + count) % capacity] = token
        count += 1
    }

    /// Uses the longest qualifying context, then backs off. Frequency ties choose the lower ID.
    /// Thresholds apply to every order; `result` retains its allocation between rounds.
    func draft(
        maximumTokens: Int,
        minimumOrder: Int = 1,
        minimumOccurrences: Int,
        minimumConfidence: Float,
        into result: inout [Int]
    ) {
        precondition((1 ... maximumOrder).contains(minimumOrder))
        precondition(minimumOccurrences >= 1)
        precondition(minimumConfidence.isFinite && (0 ... 1).contains(minimumConfidence))
        result.removeAll(keepingCapacity: true)
        guard maximumTokens > 0, count >= minimumOrder else { return }
        result.reserveCapacity(maximumTokens)
        var suffix = Key()
        for offset in 1 ... min(count, maximumOrder) {
            suffix.extend(token(at: count - offset))
        }
        for _ in 0 ..< maximumTokens {
            var candidate: Int?
            for order in stride(
                from: min(suffix.length, maximumOrder), through: minimumOrder, by: -1)
            {
                let slot = contextSlot(suffix.prefix(order))
                let reference = contextSlots[slot]
                guard reference != 0 else { continue }
                let index = Int(reference - 1)
                let context = contexts[index]
                guard Int(context.upperBound) >= minimumOccurrences,
                    Float(context.upperBound) >= minimumConfidence * Float(context.total)
                else { continue }
                let best = winner(index)
                guard best >= 0 else { continue }
                let continuation = continuations[Int(best)]
                guard Int(continuation.count) >= minimumOccurrences,
                    Float(continuation.count) >= minimumConfidence * Float(context.total)
                else { continue }
                candidate = continuation.token
                break
            }
            guard let candidate else { break }
            result.append(candidate)
            suffix.append(candidate)
        }
    }

    @inline(__always)
    private func token(at offset: Int) -> Int { tokens[(first + offset) % capacity] }

    @inline(__always)
    private static func mix(_ value: UInt64) -> UInt64 {
        var value = value
        value = (value ^ (value >> 30)) &* 0xbf58_476d_1ce4_e5b9
        value = (value ^ (value >> 27)) &* 0x94d0_49bb_1331_11eb
        return value ^ (value >> 31)
    }

    @inline(__always)
    private func contextSlot(_ key: Key) -> Int {
        var slot = Int(truncatingIfNeeded: key.hash) & mask
        while contextSlots[slot] != 0 {
            if contexts[Int(contextSlots[slot] - 1)].key == key { return slot }
            slot = (slot + 1) & mask
        }
        return slot
    }

    @inline(__always)
    private func continuationHash(_ context: Int32, _ token: Int) -> Int {
        Int(
            truncatingIfNeeded: Self.mix(
                UInt64(truncatingIfNeeded: token)
                    ^ (UInt64(UInt32(bitPattern: context)) &* 0x9e37_79b9_7f4a_7c15))) & mask
    }

    @inline(__always)
    private func continuationSlot(_ context: Int32, _ token: Int) -> Int {
        var slot = continuationHash(context, token)
        while continuationSlots[slot] != 0 {
            let entry = continuations[Int(continuationSlots[slot] - 1)]
            if entry.context == context && entry.token == token { return slot }
            slot = (slot + 1) & mask
        }
        return slot
    }

    private func insert(_ key: Key, token: Int) {
        let slot = contextSlot(key)
        let index: Int
        if contextSlots[slot] == 0 {
            if freeContext >= 0 {
                index = Int(freeContext)
                freeContext = contexts[index].head
            } else {
                index = contextHighWater
                precondition(index < entryCapacity)
                contextHighWater += 1
            }
            contexts[index] = Context(key: key)
            contextSlots[slot] = Int32(index + 1)
        } else {
            index = Int(contextSlots[slot] - 1)
        }
        let continuationSlot = continuationSlot(Int32(index), token)
        let entry: Int
        if continuationSlots[continuationSlot] == 0 {
            if freeContinuation >= 0 {
                entry = Int(freeContinuation)
                freeContinuation = continuations[entry].next
            } else {
                entry = continuationHighWater
                precondition(entry < entryCapacity)
                continuationHighWater += 1
            }
            let head = contexts[index].head
            continuations[entry] = Continuation(token: token, context: Int32(index), next: head)
            if head >= 0 { continuations[Int(head)].previous = Int32(entry) }
            contexts[index].head = Int32(entry)
            continuationSlots[continuationSlot] = Int32(entry + 1)
        } else {
            entry = Int(continuationSlots[continuationSlot] - 1)
        }
        continuations[entry].count += 1
        contexts[index].total += 1
        let best = contexts[index].winner
        if contexts[index].total == 1 || (best >= 0 && outranks(entry, Int(best))) {
            contexts[index].winner = Int32(entry)
        }
        contexts[index].upperBound = max(contexts[index].upperBound, continuations[entry].count)
    }

    private func remove(_ key: Key, token: Int) {
        let slot = contextSlot(key)
        precondition(contextSlots[slot] > 0)
        let index = Int(contextSlots[slot] - 1)
        let successorSlot = continuationSlot(Int32(index), token)
        precondition(continuationSlots[successorSlot] > 0)
        let entry = Int(continuationSlots[successorSlot] - 1)
        continuations[entry].count -= 1
        contexts[index].total -= 1
        if contexts[index].winner == Int32(entry) { contexts[index].winner = -1 }
        if continuations[entry].count == 0 {
            let next = continuations[entry].next
            let previous = continuations[entry].previous
            if next >= 0 { continuations[Int(next)].previous = previous }
            if previous >= 0 {
                continuations[Int(previous)].next = next
            } else {
                contexts[index].head = next
            }
            removeContinuationSlot(successorSlot)
            continuations[entry].next = freeContinuation
            freeContinuation = Int32(entry)
        }
        if contexts[index].total == 0 {
            removeContextSlot(slot)
            contexts[index].head = freeContext
            freeContext = Int32(index)
        }
    }

    @inline(__always)
    private func outranks(_ left: Int, _ right: Int) -> Bool {
        let lhs = continuations[left]
        let rhs = continuations[right]
        return lhs.count > rhs.count || (lhs.count == rhs.count && lhs.token < rhs.token)
    }

    private func winner(_ index: Int) -> Int32 {
        if contexts[index].winner >= 0 { return contexts[index].winner }
        var best = contexts[index].head
        guard best >= 0 else { return -1 }
        var current = continuations[Int(best)].next
        while current >= 0 {
            if outranks(Int(current), Int(best)) { best = current }
            current = continuations[Int(current)].next
        }
        contexts[index].winner = best
        contexts[index].upperBound = continuations[Int(best)].count
        return best
    }

    // Close probe clusters after deletion so an indefinitely moving window leaves no tombstones.
    private func removeContextSlot(_ removed: Int) {
        var hole = removed
        var slot = (hole + 1) & mask
        while contextSlots[slot] != 0 {
            let home =
                Int(truncatingIfNeeded: contexts[Int(contextSlots[slot] - 1)].key.hash) & mask
            if ((slot - home) & mask) >= ((hole - home) & mask) {
                contextSlots[hole] = contextSlots[slot]
                hole = slot
            }
            slot = (slot + 1) & mask
        }
        contextSlots[hole] = 0
    }

    private func removeContinuationSlot(_ removed: Int) {
        var hole = removed
        var slot = (hole + 1) & mask
        while continuationSlots[slot] != 0 {
            let entry = continuations[Int(continuationSlots[slot] - 1)]
            let home = continuationHash(entry.context, entry.token)
            if ((slot - home) & mask) >= ((hole - home) & mask) {
                continuationSlots[hole] = continuationSlots[slot]
                hole = slot
            }
            slot = (slot + 1) & mask
        }
        continuationSlots[hole] = 0
    }
}
