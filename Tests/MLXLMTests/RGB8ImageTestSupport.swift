// Copyright © 2026 Apple Inc.

import CryptoKit
import Foundation
import MLX
import MLXLMCommon

@testable import MLXVLM

// Image helpers shared by the image and vision tests.

extension RGB8Image {
    /// `(x*31 + y*17 + c*59 + ((x*y) % 7)*23) % 256`, the `pattern` of
    /// tools/lfm2_vl_test_vectors.py, which the expected values are computed on.
    static func testPattern(width: Int, height: Int) -> RGB8Image {
        var bytes = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                for c in 0 ..< 3 {
                    bytes[(y * width + x) * 3 + c] = UInt8(
                        (x * 31 + y * 17 + c * 59 + ((x * y) % 7) * 23) % 256)
                }
            }
        }
        return RGB8Image(width: width, height: height, bytes: bytes)
    }

    /// The sum of all bytes.
    var byteSum: Int {
        bytes.reduce(0) { $0 + Int($1) }
    }

    /// The pixels as a `[height, width, 3]` `.array` input. A height of 1 or 3 would read as
    /// channels first; see `RGB8Image(array:)`.
    var userImage: UserInput.Image {
        precondition(height != 1 && height != 3, "ambiguous array layout")
        return .array(MLXArray(bytes, [height, width, 3]))
    }
}

/// SHA-256 of the array's float32 values, in hex.
func sha256Hex(_ array: MLXArray) -> String {
    array.asType(.float32).asArray(Float.self).withUnsafeBytes {
        SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined()
    }
}
