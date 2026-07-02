// Copyright © 2026 Apple Inc.

#if !canImport(CoreMedia)

public struct CMTime {
    public var value: Int64
    public var timescale: Int32
}

#endif
