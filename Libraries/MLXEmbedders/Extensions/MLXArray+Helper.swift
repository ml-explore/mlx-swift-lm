// Copyright © 2026 Apple Inc.

import MLX

extension MLXArray {

    public static func arange(_ size: Int) -> MLXArray {
        return MLXArray(Array(0 ..< size))
    }

}
