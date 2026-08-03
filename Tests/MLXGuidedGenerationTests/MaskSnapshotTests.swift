// Copyright © 2025 Apple Inc.

import Testing

@testable import MLXGuidedGeneration

@Suite
struct MaskSnapshotTests {

    @Test
    func captureWithNilMaskProducesNilHash() {
        let snapshot = MaskSnapshot.capture(sampleMask: nil, vocabSize: 100, tokenIndex: 0)
        let summary = snapshot.summary()
        #expect(summary.contains("maskHash=nil"))
        #expect(summary.contains("token=0"))
        #expect(summary.contains("isStop=F"))
    }

    @Test
    func captureWithNonNilMaskProducesHexHash() {
        // A single UInt32 word with all bits set
        var maskWord: UInt32 = 0xFFFF_FFFF
        let snapshot = withUnsafePointer(to: &maskWord) { ptr in
            MaskSnapshot.capture(sampleMask: ptr, vocabSize: 32, tokenIndex: 5)
        }
        let summary = snapshot.summary()
        // Hash should be a hex string prefixed with 0x
        #expect(summary.contains("maskHash=0x"))
        #expect(summary.contains("token=5"))
        #expect(!summary.contains("maskHash=nil"))
    }

    @Test
    func stableHashForIdenticalMasks() {
        // Same mask data should produce identical hashes
        var mask1: [UInt32] = [0xDEAD_BEEF, 0xCAFE_BABE]
        var mask2: [UInt32] = [0xDEAD_BEEF, 0xCAFE_BABE]

        let snapshot1 = mask1.withUnsafeBufferPointer { buf in
            MaskSnapshot.capture(sampleMask: buf.baseAddress!, vocabSize: 64, tokenIndex: 0)
        }
        let snapshot2 = mask2.withUnsafeBufferPointer { buf in
            MaskSnapshot.capture(sampleMask: buf.baseAddress!, vocabSize: 64, tokenIndex: 0)
        }

        #expect(snapshot1.summary() == snapshot2.summary())
    }

    @Test
    func differentMasksProduceDifferentHashes() {
        var mask1: [UInt32] = [0xDEAD_BEEF, 0xCAFE_BABE]
        var mask2: [UInt32] = [0xDEAD_BEEF, 0x0000_0000]

        let snapshot1 = mask1.withUnsafeBufferPointer { buf in
            MaskSnapshot.capture(sampleMask: buf.baseAddress!, vocabSize: 64, tokenIndex: 0)
        }
        let snapshot2 = mask2.withUnsafeBufferPointer { buf in
            MaskSnapshot.capture(sampleMask: buf.baseAddress!, vocabSize: 64, tokenIndex: 0)
        }

        #expect(snapshot1.summary() != snapshot2.summary())
    }

    @Test
    func summaryFormatIsFixedWidthForDiffing() {
        // The hash should be zero-padded to 16 hex digits for consistent width
        var maskWord: UInt32 = 0x0000_0001
        let snapshot = withUnsafePointer(to: &maskWord) { ptr in
            MaskSnapshot.capture(sampleMask: ptr, vocabSize: 32, tokenIndex: 42)
        }
        let summary = snapshot.summary()
        // Format: [Diag] token=NNN isStop=F maskHash=0x0000000000000000
        #expect(summary.hasPrefix("[Diag] "))
        #expect(summary.contains("token=42"))
        #expect(summary.contains("isStop=F"))
        // Hash should be exactly 16 hex chars (64-bit FNV-1a)
        let hashRange = summary.range(of: "0x")!
        let hashStart = hashRange.upperBound
        let hashString = String(summary[hashStart...])
        #expect(hashString.count == 16)
    }

    @Test
    func isStopTrueShowsInSummary() {
        let snapshot = MaskSnapshot.capture(
            sampleMask: nil, vocabSize: 100, tokenIndex: 10, isStop: true
        )
        let summary = snapshot.summary()
        #expect(summary.contains("isStop=T"))
    }
}
