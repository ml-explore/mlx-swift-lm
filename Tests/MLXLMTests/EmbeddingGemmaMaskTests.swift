// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXEmbedders

// A tiny EmbeddingGemma with random weights: 3 query heads, 1 key/value head, as in the real model.
// Every MLX op runs on the CPU via `Device.withDefaultDevice(.cpu)`.
// The shape precondition in Gemma3ModelBackbone (mask and inputs must match) is documented only:
// the repo has no way to test a precondition failure.

private func config(
    bidirectional: Bool?, layers: Int = 2, slidingWindow: Int = 8, pattern: Int = 2
) throws -> Gemma3Configuration {
    let flag = bidirectional.map { "\"use_bidirectional_attention\": \($0)," } ?? ""
    let json = """
        {
            "model_type": "gemma3_text", \(flag)
            "hidden_size": 32, "num_hidden_layers": \(layers), "intermediate_size": 64,
            "num_attention_heads": 3, "num_key_value_heads": 1, "head_dim": 8,
            "vocab_size": 64, "query_pre_attn_scalar": 8, "max_position_embeddings": 128,
            "sliding_window": \(slidingWindow), "sliding_window_pattern": \(pattern)
        }
        """
    return try JSONDecoder().decode(Gemma3Configuration.self, from: Data(json.utf8))
}

private func model(_ c: Gemma3Configuration, seed: UInt64 = 0) -> EmbeddingGemma {
    MLXRandom.seed(seed)
    let m = EmbeddingGemma(c)
    eval(m)
    return m
}

/// Token ids 1...63 (0 is the pad id), deterministic.
private func tokens(_ count: Int, offset: Int = 0) -> [Int32] {
    (0 ..< count).map { Int32(1 + (($0 + offset) * 7) % 63) }
}

private func run(_ m: EmbeddingGemma, _ ids: [[Int32]], mask: [[Int32]]?) -> (MLXArray, MLXArray) {
    let length = ids[0].count
    let input = MLXArray(ids.flatMap { $0 }, [ids.count, length])
    let attentionMask = mask.map { MLXArray($0.flatMap { $0 }, [$0.count, length]) }
    let out = m(input, positionIds: nil, tokenTypeIds: nil, attentionMask: attentionMask)
    let states = out.hiddenStates!
    let pooled = out.pooledOutput!
    eval(states, pooled)
    return (states, pooled)
}

/// Right-padded (or left-padded) ids and mask for texts of the given lengths.
private func padded(_ lengths: [Int], left: Bool = false) -> ([[Int32]], [[Int32]]) {
    let length = lengths.max()!
    var ids: [[Int32]] = []
    var mask: [[Int32]] = []
    for (row, n) in lengths.enumerated() {
        let real = tokens(n, offset: row * 11)
        let pad = [Int32](repeating: 0, count: length - n)
        ids.append(left ? pad + real : real + pad)
        mask.append(
            left ? pad + [Int32](repeating: 1, count: n) : [Int32](repeating: 1, count: n) + pad)
    }
    return (ids, mask)
}

private func bitEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
    a.shape == b.shape && arrayEqual(a.view(dtype: .uint32), b.view(dtype: .uint32)).item(Bool.self)
}

private func relNorm(_ a: MLXArray, _ b: MLXArray) -> Double {
    let x = a.asType(.float32).asArray(Float.self)
    let y = b.asType(.float32).asArray(Float.self)
    var num = 0.0
    var den = 0.0
    for (p, q) in zip(x, y) {
        num += (Double(p) - Double(q)) * (Double(p) - Double(q))
        den += Double(q) * Double(q)
    }
    return (num / den).squareRoot()
}

private func hasNaN(_ a: MLXArray) -> Bool {
    isNaN(a).any().item(Bool.self)
}

/// Batch rows against the same texts run one at a time: real positions and pooled output.
private func expectBatchMatchesSingles(
    _ m: EmbeddingGemma, lengths: [Int], left: Bool = false,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let (ids, mask) = padded(lengths, left: left)
    let (states, pooled) = run(m, ids, mask: mask)
    #expect(!hasNaN(states) && !hasNaN(pooled), sourceLocation: sourceLocation)
    let length = ids[0].count
    for (row, n) in lengths.enumerated() {
        let real = left ? (length - n) ..< length : 0 ..< n
        let (single, singlePooled) = run(
            m, [Array(ids[row][real])], mask: [[Int32](repeating: 1, count: n)])
        let rowStates = states[row, real.lowerBound ..< real.upperBound]
        #expect(relNorm(rowStates, single[0]) <= 1e-5, "row \(row)", sourceLocation: sourceLocation)
        #expect(
            relNorm(pooled[row], singlePooled[0]) <= 1e-5, "row \(row)",
            sourceLocation: sourceLocation)
    }
}

struct EmbeddingGemmaMaskTests {

    // MARK: - Flag false: today's causal path

    @Test("G1a: flag false ignores the padding mask in attention")
    func causalIgnoresMask() throws {
        try Device.withDefaultDevice(.cpu) {
            let m = model(try config(bidirectional: false))
            let (ids, mask) = padded([12, 7])
            let (withMask, _) = run(m, ids, mask: mask)
            let (withoutMask, _) = run(m, ids, mask: nil)
            #expect(bitEqual(withMask, withoutMask))
        }
    }

    @Test("G1b: flag false is causal, a later token never changes an earlier state")
    func causalIgnoresFuture() throws {
        try Device.withDefaultDevice(.cpu) {
            let m = model(try config(bidirectional: nil))
            var ids = tokens(10)
            let (before, _) = run(m, [ids], mask: nil)
            ids[9] = ids[9] % 63 + 1
            let (after, _) = run(m, [ids], mask: nil)
            #expect(bitEqual(before[0, ..<9], after[0, ..<9]))
            #expect(!bitEqual(before[0, 9], after[0, 9]))
        }
    }

    @Test("G1c: flag false keeps the causal window of slidingWindow keys")
    func causalWindow() throws {
        try Device.withDefaultDevice(.cpu) {
            // One sliding layer; the window is applied only when the input is longer than it.
            let m = model(try config(bidirectional: false, layers: 1, slidingWindow: 4))
            var ids = tokens(10)
            let (before, _) = run(m, [ids], mask: nil)
            ids[0] = ids[0] % 63 + 1
            let (after, _) = run(m, [ids], mask: nil)
            // Position 3 sees token 0 (distance 3 < 4); position 4 does not (distance 4).
            #expect(!bitEqual(before[0, 3], after[0, 3]))
            #expect(bitEqual(before[0, 4 ..< 10], after[0, 4 ..< 10]))
        }
    }

    // MARK: - Configuration

    @Test(
        "G2: use_bidirectional_attention decodes, absent and null mean false",
        arguments: [
            ("true", true), ("false", false), ("null", false), (nil, false),
        ] as [(String?, Bool)])
    func decodeFlag(value: String?, expected: Bool) throws {
        let flag = value.map { "\"use_bidirectional_attention\": \($0)," } ?? ""
        let json = "{ \(flag) \"model_type\": \"gemma3_text\" }"
        let c = try JSONDecoder().decode(Gemma3Configuration.self, from: Data(json.utf8))
        #expect(c.useBidirectionalAttention == expected)
    }

    // MARK: - Flag true

    @Test("G3: flag true is bidirectional in both directions")
    func bidirectional() throws {
        try Device.withDefaultDevice(.cpu) {
            // Length 5 fits inside the half window of 4, so every layer sees the whole sequence.
            let m = model(try config(bidirectional: true))
            let ids = tokens(5)
            let (base, _) = run(m, [ids], mask: nil)
            var lastChanged = ids
            lastChanged[4] = lastChanged[4] % 63 + 1
            let (afterLast, _) = run(m, [lastChanged], mask: nil)
            #expect(!bitEqual(base[0, 0], afterLast[0, 0]))
            var firstChanged = ids
            firstChanged[0] = firstChanged[0] % 63 + 1
            let (afterFirst, _) = run(m, [firstChanged], mask: nil)
            #expect(!bitEqual(base[0, 4], afterFirst[0, 4]))
        }
    }

    @Test(
        "G4: the sliding window reaches slidingWindow / 2 tokens on each side",
        arguments: [8, 9])
    func slidingWindow(slidingWindow: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            // One sliding layer; half window 4 for 8 and 9 (transformers: |i - j| < sw // 2 + 1).
            let m = model(try config(bidirectional: true, layers: 1, slidingWindow: slidingWindow))
            let ids = tokens(12)
            let (base, _) = run(m, [ids], mask: nil)
            var first = ids
            first[0] = first[0] % 63 + 1
            let (afterFirst, _) = run(m, [first], mask: nil)
            #expect(!bitEqual(base[0, 4], afterFirst[0, 4]))
            #expect(bitEqual(base[0, 5 ..< 12], afterFirst[0, 5 ..< 12]))
            var last = ids
            last[11] = last[11] % 63 + 1
            let (afterLast, _) = run(m, [last], mask: nil)
            #expect(!bitEqual(base[0, 7], afterLast[0, 7]))
            #expect(bitEqual(base[0, ..<7], afterLast[0, ..<7]))
        }
    }

    @Test("G5: a right-padded batch matches the texts run one at a time")
    func rightPadding() throws {
        try Device.withDefaultDevice(.cpu) {
            // The short row has padded query rows whose window holds no real key.
            expectBatchMatchesSingles(model(try config(bidirectional: true)), lengths: [20, 3])
        }
    }

    @Test(
        "G6: a mixed batch, one text longer than the window and one short, stays finite and exact")
    func mixedBatch() throws {
        try Device.withDefaultDevice(.cpu) {
            expectBatchMatchesSingles(model(try config(bidirectional: true)), lengths: [40, 6])
        }
    }

    @Test("G7: a left-padded batch matches the texts run one at a time")
    func leftPadding() throws {
        try Device.withDefaultDevice(.cpu) {
            expectBatchMatchesSingles(
                model(try config(bidirectional: true)), lengths: [20, 3], left: true)
        }
    }

    @Test("G8: a 1-D input with a 1-D mask equals the 2-D version")
    func oneDimensional() throws {
        try Device.withDefaultDevice(.cpu) {
            let m = model(try config(bidirectional: true))
            let ids = tokens(9)
            let mask: [Int32] = [1, 1, 1, 1, 1, 1, 0, 0, 0]
            let flat = m(
                MLXArray(ids), positionIds: nil, tokenTypeIds: nil, attentionMask: MLXArray(mask))
            let (states, pooled) = run(m, [ids], mask: [mask])
            #expect(bitEqual(flat.hiddenStates!, states))
            #expect(bitEqual(flat.pooledOutput!, pooled))
        }
    }

    @Test("G9: bfloat16 weights with padding run and stay finite")
    func bfloat16() throws {
        try Device.withDefaultDevice(.cpu) {
            let m = model(try config(bidirectional: true))
            m.update(parameters: m.parameters().mapValues { $0.asType(.bfloat16) })
            let (ids, mask) = padded([12, 5])
            let (states, pooled) = run(m, ids, mask: mask)
            #expect(states.dtype == .bfloat16)
            #expect(!hasNaN(states) && !hasNaN(pooled))
        }
    }
}
