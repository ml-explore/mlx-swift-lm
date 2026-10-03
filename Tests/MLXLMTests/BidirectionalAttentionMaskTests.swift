// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXEmbedders

// Every MLX op runs on the CPU via `Device.withDefaultDevice(.cpu)`, as in LFM2BidirectionalTests.

/// Reference mask built with loops, flattened as [B][L][L]:
/// `((halfWindow == nil || |i - j| <= halfWindow) && valid[b][j]) || i == j`.
private func referenceMask(length: Int, halfWindow: Int?, valid: [[Bool]]?) -> [Bool] {
    let rows = valid ?? [Array(repeating: true, count: length)]
    var out: [Bool] = []
    for v in rows {
        for i in 0 ..< length {
            for j in 0 ..< length {
                let inWindow = halfWindow.map { abs(i - j) <= $0 } ?? true
                out.append((inWindow && v[j]) || i == j)
            }
        }
    }
    return out
}

private func paddingMask(_ valid: [[Bool]]) -> MLXArray {
    MLXArray(valid.flatMap { $0.map { $0 ? Int32(1) : Int32(0) } }, [valid.count, valid[0].count])
}

/// Valid rows: `n` real tokens, then padding on the right (or on the left).
private func valid(length: Int, real: Int, left: Bool = false) -> [Bool] {
    (0 ..< length).map { left ? $0 >= length - real : $0 < real }
}

/// The boolean array of an `.array` mask; records an issue for any other mode or dtype.
private func boolArray(
    _ mode: MLXFast.ScaledDotProductAttentionMaskMode, shape: [Int],
    sourceLocation: SourceLocation = #_sourceLocation
) -> [Bool]? {
    guard case .array(let a) = mode else {
        Issue.record("expected an .array mask, got \(mode)", sourceLocation: sourceLocation)
        return nil
    }
    #expect(a.dtype == .bool, sourceLocation: sourceLocation)
    #expect(a.shape == shape, sourceLocation: sourceLocation)
    guard a.dtype == .bool, a.shape == shape else { return nil }
    return a.asArray(Bool.self)
}

struct BidirectionalAttentionMaskTests {

    @Test("H1: no window and no padding gives .none")
    func noRestriction() {
        Device.withDefaultDevice(.cpu) {
            let mode = createBidirectionalAttentionMask(
                length: 8, halfWindow: nil, paddingMask: nil)
            guard case .none = mode else {
                Issue.record("expected .none, got \(mode)")
                return
            }
        }
    }

    @Test("H2: window only is a symmetric band with an inclusive bound")
    func windowOnly() {
        Device.withDefaultDevice(.cpu) {
            let length = 12
            let mode = createBidirectionalAttentionMask(
                length: length, halfWindow: 3, paddingMask: nil)
            guard let m = boolArray(mode, shape: [1, 1, length, length]) else { return }
            #expect(m == referenceMask(length: length, halfWindow: 3, valid: nil))
            for i in 0 ..< length {
                for j in 0 ..< length {
                    #expect(m[i * length + j] == m[j * length + i])
                }
                #expect(m[i * length + i])
            }
            // |i - j| = 3 is inside the window, |i - j| = 4 is outside.
            #expect(m[0 * length + 3])
            #expect(!m[0 * length + 4])
            #expect(m[8 * length + 5])
            #expect(!m[8 * length + 4])
        }
    }

    @Test("H3: a zero half window leaves only the diagonal")
    func zeroWindow() {
        Device.withDefaultDevice(.cpu) {
            let length = 7
            let mode = createBidirectionalAttentionMask(
                length: length, halfWindow: 0, paddingMask: nil)
            guard let m = boolArray(mode, shape: [1, 1, length, length]) else { return }
            let identity = (0 ..< length * length).map { $0 / length == $0 % length }
            #expect(m == identity)
        }
    }

    @Test(
        "H4: a window that covers the sequence gives an all-true array",
        arguments: [9, 10, 1000, Int.max])
    func windowCoversSequence(halfWindow: Int) {
        Device.withDefaultDevice(.cpu) {
            let length = 10
            let mode = createBidirectionalAttentionMask(
                length: length, halfWindow: halfWindow, paddingMask: nil)
            guard let m = boolArray(mode, shape: [1, 1, length, length]) else { return }
            #expect(m.allSatisfy { $0 })
        }
    }

    @Test("H5: padding hides key columns, never query rows")
    func paddingOnly() {
        Device.withDefaultDevice(.cpu) {
            let length = 12
            let rows = [valid(length: length, real: 12), valid(length: length, real: 8)]
            let mode = createBidirectionalAttentionMask(
                length: length, halfWindow: nil, paddingMask: paddingMask(rows))
            guard let m = boolArray(mode, shape: [2, 1, length, length]) else { return }
            #expect(m == referenceMask(length: length, halfWindow: nil, valid: rows))
            let b = 1
            for i in 0 ..< length {
                for j in 0 ..< length {
                    let cell = m[(b * length + i) * length + j]
                    if i < 8 {
                        // A real token sees exactly the real keys.
                        #expect(cell == (j < 8), "real row \(i), key \(j)")
                    } else {
                        // A padded query row still sees every real key, plus itself.
                        #expect(cell == (j < 8 || j == i), "padded row \(i), key \(j)")
                    }
                }
            }
        }
    }

    @Test("H6: window and padding combine cell by cell", arguments: [false, true])
    func windowAndPadding(leftPadding: Bool) {
        Device.withDefaultDevice(.cpu) {
            let length = 12
            let rows = [
                valid(length: length, real: 12), valid(length: length, real: 8, left: leftPadding),
            ]
            let mode = createBidirectionalAttentionMask(
                length: length, halfWindow: 3, paddingMask: paddingMask(rows))
            guard let m = boolArray(mode, shape: [2, 1, length, length]) else { return }
            #expect(m == referenceMask(length: length, halfWindow: 3, valid: rows))
        }
    }

    @Test("H7: no query row is ever all false")
    func noEmptyRow() {
        Device.withDefaultDevice(.cpu) {
            let length = 20
            // One sequence fully padded, one with 3 real tokens and a window too small for them.
            let rows = [
                Array(repeating: false, count: length), valid(length: length, real: 3),
                valid(length: length, real: 20),
            ]
            for halfWindow in [nil, 0, 1, 4] as [Int?] {
                let mode = createBidirectionalAttentionMask(
                    length: length, halfWindow: halfWindow, paddingMask: paddingMask(rows))
                guard let m = boolArray(mode, shape: [3, 1, length, length]) else { return }
                for row in 0 ..< 3 * length {
                    let cells = m[(row * length) ..< ((row + 1) * length)]
                    #expect(
                        cells.contains(true),
                        "row \(row) with halfWindow \(String(describing: halfWindow))")
                }
            }
        }
    }

    @Test("H8: every padding dtype counts nonzero as a real token")
    func paddingDtypes() {
        Device.withDefaultDevice(.cpu) {
            let length = 6
            let expected = referenceMask(
                length: length, halfWindow: 2, valid: [[true, true, true, true, false, false]])
            let masks: [MLXArray] = [
                MLXArray([Int32(1), 1, 1, 1, 0, 0], [1, length]),
                MLXArray([Int64(1), 1, 1, 1, 0, 0], [1, length]),
                MLXArray([true, true, true, true, false, false], [1, length]),
                MLXArray([Float(1), 1, 1, 1, 0, 0], [1, length]),
                // Any nonzero value is a real token: 2, -1, 0.5 and NaN included.
                MLXArray([Float(2), -1, 0.5, .nan, 0, 0], [1, length]),
            ]
            for mask in masks {
                let mode = createBidirectionalAttentionMask(
                    length: length, halfWindow: 2, paddingMask: mask)
                guard let m = boolArray(mode, shape: [1, 1, length, length]) else { return }
                #expect(m == expected, "padding mask of dtype \(mask.dtype)")
            }
        }
    }

    @Test("H9: a single token always sees itself")
    func singleToken() {
        Device.withDefaultDevice(.cpu) {
            let windowed = createBidirectionalAttentionMask(
                length: 1, halfWindow: 0, paddingMask: nil)
            #expect(boolArray(windowed, shape: [1, 1, 1, 1]) == [true])
            let padded = createBidirectionalAttentionMask(
                length: 1, halfWindow: nil, paddingMask: MLXArray([Int32(0)], [1, 1]))
            #expect(boolArray(padded, shape: [1, 1, 1, 1]) == [true])
        }
    }

    @Test(
        "H10: real rows match the transformers rule written from sliding_window",
        arguments: [7, 8, 9], [false, true])
    func matchesTransformers(slidingWindow: Int, leftPadding: Bool) {
        Device.withDefaultDevice(.cpu) {
            let length = 16
            let rows = [valid(length: length, real: 11, left: leftPadding)]
            // transformers 5.17 stores w = sliding_window // 2 + 1 (configuration_gemma3.py:106),
            // ORs the causal sliding window with |i - j| < w (masking_utils.py:93-101, 135-140;
            // modeling_gemma3.py:471-481), then hides padded keys (masking_utils.py:169-178).
            let w = slidingWindow / 2 + 1
            let mode = createBidirectionalAttentionMask(
                length: length, halfWindow: slidingWindow / 2, paddingMask: paddingMask(rows))
            guard let m = boolArray(mode, shape: [1, 1, length, length]) else { return }
            for i in 0 ..< length where rows[0][i] {
                for j in 0 ..< length {
                    let hf = ((i - w < j && j <= i) || abs(i - j) < w) && rows[0][j]
                    #expect(
                        m[i * length + j] == hf,
                        "sliding_window \(slidingWindow), row \(i), key \(j)")
                }
            }
        }
    }
}
