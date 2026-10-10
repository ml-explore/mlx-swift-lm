// Copyright © 2026 Apple Inc.

/// Allocation arithmetic only; capacity never changes a cache's live-token count.
enum KVCacheCapacity {
    static func next(current: Int, required: Int, step: Int, reservation: Int? = nil) -> Int {
        precondition(current >= 0 && required >= 0 && step > 0)
        guard max(required, reservation ?? 0) > current else { return current }

        let target: Int
        if let reservation, reservation >= required {
            target = reservation
        } else {
            let increment = min(max(current / 4, step), max(8192, step))
            let (grown, overflow) = current.addingReportingOverflow(increment)
            target = max(required, overflow ? Int.max : grown)
        }

        let remainder = target % step
        guard remainder != 0 else { return target }
        let (rounded, overflow) = target.addingReportingOverflow(step - remainder)
        return overflow ? Int.max : rounded
    }
}

/// Reserve known text positions without changing custom, rotating, or compressed caches.
package func withReservedPromptCache<Result>(
    _ cache: [KVCache], additionalTokens: Int, _ body: () throws -> Result
) rethrows -> Result {
    precondition(additionalTokens >= 0)
    let simpleCaches = KVCacheTree.leaves(in: cache).compactMap { leaf -> KVCacheSimple? in
        guard ObjectIdentifier(type(of: leaf.cache)) == ObjectIdentifier(KVCacheSimple.self)
        else { return nil }
        return leaf.cache as? KVCacheSimple
    }
    for cache in simpleCaches {
        let (capacity, overflow) = cache.offset.addingReportingOverflow(additionalTokens)
        if !overflow { cache.reserveCapacity(capacity) }
    }
    defer {
        for cache in simpleCaches { cache.clearCapacityReservation() }
    }
    return try body()
}
