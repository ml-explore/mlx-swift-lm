// Copyright © 2026 Apple Inc.

#if !canImport(CoreGraphics)

public typealias CGFloat = Double

public struct CGSize: Sendable {
    public var width: CGFloat
    public var height: CGFloat

    public init(width: CGFloat, height: CGFloat) {
        self.width = width
        self.height = height
    }

    public init(width: Int, height: Int) {
        self.width = CGFloat(width)
        self.height = CGFloat(height)
    }
}

#endif
