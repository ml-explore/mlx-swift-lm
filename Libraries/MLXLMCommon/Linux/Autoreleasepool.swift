// Copyright © 2026 Apple Inc.

#if !canImport(ObjectiveC)

/// There is no autorelease machinery to drain without the Objective-C runtime,
/// so the pool is just the body. `public` because `MLXLLM` calls it too.
@inlinable
public func autoreleasepool<Result>(invoking body: () throws -> Result) rethrows -> Result {
    try body()
}

#endif
