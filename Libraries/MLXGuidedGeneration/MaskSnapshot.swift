// Copyright © 2026 Apple Inc.

/// Captures the state of a grammar mask at a single generation step
/// for deterministic comparison between architectures.
struct MaskSnapshot {

    // MARK: - Private State

    private let tokenIndex: Int
    private let isStop: Bool
    private let maskHash: String

    // MARK: - Public API

    /// Captures a snapshot of the current mask state.
    ///
    /// - Parameters:
    ///   - sampleMask: Bitmask pointer from `MaskResult.mask` (rebound
    ///     to `UnsafePointer<UInt32>`), or nil when the mask needs no
    ///     application (unconditional splice).
    ///   - vocabSize: Number of valid bits in the mask. Determines how many
    ///     UInt32 words to read: `(vocabSize + 31) / 32`.
    ///   - tokenIndex: The current token generation index.
    ///   - isStop: Whether the grammar has reached a stop state.
    static func capture(
        sampleMask: UnsafePointer<UInt32>?,
        vocabSize: Int,
        tokenIndex: Int,
        isStop: Bool = false
    ) -> MaskSnapshot {
        let hash: String
        if let mask = sampleMask {
            hash = computeHash(mask: mask, vocabSize: vocabSize)
        } else {
            hash = "nil"
        }
        return MaskSnapshot(tokenIndex: tokenIndex, isStop: isStop, maskHash: hash)
    }

    /// Returns a fixed-width one-line summary for log diffing.
    ///
    /// Format: `[Diag] token=NNN isStop=F maskHash=0xABCD1234`
    func summary() -> String {
        let stopFlag = isStop ? "T" : "F"
        let hashField = maskHash == "nil" ? "nil" : "0x\(maskHash)"
        return "[Diag] token=\(tokenIndex) isStop=\(stopFlag) maskHash=\(hashField)"
    }

    // MARK: - Private

    /// FNV-1a hash over the UInt32 words of the bitmask.
    private static func computeHash(mask: UnsafePointer<UInt32>, vocabSize: Int) -> String {
        let wordCount = (vocabSize + 31) / 32
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325  // FNV-1a offset basis
        let prime: UInt64 = 0x100_0000_01b3  // FNV-1a prime

        for i in 0 ..< wordCount {
            let word = mask[i]
            // Hash each byte of the UInt32 word
            for shift in stride(from: 0, to: 32, by: 8) {
                let byte = UInt64((word >> shift) & 0xFF)
                hash ^= byte
                hash &*= prime
            }
        }

        let hex = String(hash, radix: 16, uppercase: true)
        // Zero-pad to 16 characters for fixed-width output
        return String(repeating: "0", count: max(0, 16 - hex.count)) + hex
    }
}
