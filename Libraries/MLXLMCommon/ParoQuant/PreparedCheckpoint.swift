import Foundation
import MLX
import os

private let logger = Logger(subsystem: "mlx-swift-lm", category: "paroquant")

/// Prepared Checkpoint — the once-converted MLX-native form of a PARO
/// checkpoint, stored beside the original so later loads skip the AutoAWQ
/// conversion (unpack/reorder/repack, Mamba split, MoE expert stacking,
/// rotation-key remap).
///
/// Rotation parameters (`theta` / `pairs` / `channel_scales`) are stored
/// verbatim and every rotation-derived value is recomputed at load by
/// `prepareDerivedRotationState()` — nothing semantic is baked into the
/// artifact, only layout conversion whose output is deterministic.
///
/// The artifact is self-describing: its safetensors metadata carries a
/// manifest of the source files it was converted from plus a format version.
/// Any mismatch (source re-download, conversion-semantics change) or read
/// failure makes the loader silently fall back to full conversion and
/// rewrite the artifact — a Prepared Checkpoint can never fail a load that
/// would otherwise succeed.
public enum ParoQuantPreparedCheckpoint {

    /// Artifact file name, written into the checkpoint directory.
    public static let fileName = "prepared_checkpoint.safetensors"

    /// In-progress write target; atomically renamed to `fileName` on
    /// completion. A crash can strand one — it is ignored by scans (below)
    /// and overwritten by the next write.
    static let temporaryFileName = "prepared_checkpoint.incomplete.safetensors"

    /// Never-shipped artifact name from the original cache design; kept
    /// excluded so a stray file can't be ingested as checkpoint weights.
    static let legacyFileName = "prerotated_cache.safetensors"

    /// File names that must never be treated as checkpoint sources — by the
    /// loader's raw-safetensors scan, by the manifest, and by any caller
    /// computing weight identity over the directory (e.g. the app's
    /// `ModelFingerprint`).
    public static let excludedFileNames: Set<String> = [
        fileName, temporaryFileName, legacyFileName,
    ]

    /// Bump whenever conversion semantics change — `convertAutoAWQ`, the
    /// Mamba projection split, MoE expert stacking, or the rotation-key
    /// remap. A bump invalidates every existing artifact on next load.
    static let formatVersion = 1

    /// Safetensors metadata key holding the JSON-encoded ``Manifest``.
    static let manifestKey = "paroquant.prepared_checkpoint.manifest"

    /// Refuse to write when the volume would drop below this fraction of the
    /// artifact's size in free space after the write (2× = artifact size in
    /// headroom left over).
    static let freeSpaceSafetyFactor: Int64 = 2

    // MARK: - Manifest

    /// Identity of the conversion inputs: every top-level source
    /// `*.safetensors` (artifact names excluded) plus `config.json`, as
    /// (name, size, mtime); byte-hashing is deliberately skipped for the
    /// same reason the app's `ModelFingerprint` skips it — it would cost
    /// seconds on exactly the path this artifact exists to speed up.
    struct Manifest: Codable, Equatable {
        struct Source: Codable, Equatable {
            let name: String
            let size: Int64
            let mtimeNs: Int64
        }

        let formatVersion: Int
        let sources: [Source]
    }

    /// Manifest describing the directory's current conversion inputs,
    /// sorted by file name for deterministic comparison.
    static func currentManifest(directory: URL) throws -> Manifest {
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )
        var sourceURLs = contents.filter {
            $0.pathExtension == "safetensors" && !excludedFileNames.contains($0.lastPathComponent)
        }
        sourceURLs.append(directory.appendingPathComponent("config.json", isDirectory: false))

        var sources: [Manifest.Source] = []
        sources.reserveCapacity(sourceURLs.count)
        for url in sourceURLs {
            let values = try url.resourceValues(forKeys: [
                .fileSizeKey, .contentModificationDateKey,
            ])
            // Same Double-derived derivation as the app's ModelFingerprint —
            // stored and recomputed values must round identically.
            let mtimeNs = values.contentModificationDate.map {
                Int64($0.timeIntervalSince1970 * 1_000_000_000)
            }
            sources.append(
                Manifest.Source(
                    name: url.lastPathComponent,
                    size: Int64(values.fileSize ?? 0),
                    mtimeNs: mtimeNs ?? 0
                ))
        }
        sources.sort { $0.name < $1.name }
        return Manifest(formatVersion: formatVersion, sources: sources)
    }

    // MARK: - Read

    static func artifactURL(for directory: URL) -> URL {
        directory.appendingPathComponent(fileName, isDirectory: false)
    }

    /// Load the prepared weights when a valid artifact matches `manifest`.
    /// Returns `nil` — after deleting the artifact — when it is absent,
    /// stale, or unreadable (self-heal: the caller falls back to full
    /// conversion and a background rewrite).
    static func load(directory: URL, manifest: Manifest) -> [String: MLXArray]? {
        let url = artifactURL(for: directory)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do {
            let (arrays, metadata) = try loadArraysAndMetadata(url: url)
            guard let manifestJSON = metadata[manifestKey],
                let stored = try? JSONDecoder().decode(
                    Manifest.self, from: Data(manifestJSON.utf8)),
                stored == manifest
            else {
                logger.notice("Prepared Checkpoint is stale — re-converting")
                try? FileManager.default.removeItem(at: url)
                return nil
            }
            return arrays
        } catch {
            logger.warning(
                "Prepared Checkpoint unreadable (\(String(describing: error), privacy: .public)) — re-converting"
            )
            try? FileManager.default.removeItem(at: url)
            return nil
        }
    }

    // MARK: - Write

    /// Serialize `weights` with the manifest embedded and atomically publish
    /// the artifact (temp file + rename). Skips — with a notice — when free
    /// disk space is below `freeSpaceSafetyFactor ×` the estimated artifact
    /// size. Failures only log: the loaded model is unaffected and the next
    /// load simply converts again.
    static func write(weights: [String: MLXArray], manifest: Manifest, directory: URL) {
        let clock = ContinuousClock()
        let start = clock.now
        let fm = FileManager.default
        let tmpURL = directory.appendingPathComponent(temporaryFileName, isDirectory: false)
        do {
            let estimatedBytes = weights.values.reduce(Int64(0)) { $0 + Int64($1.nbytes) }
            let free =
                try directory
                .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
                .volumeAvailableCapacityForImportantUsage ?? 0
            guard free >= estimatedBytes * freeSpaceSafetyFactor else {
                logger.notice(
                    "Prepared Checkpoint write skipped — free space \(free, privacy: .public)B < \(freeSpaceSafetyFactor, privacy: .public)× estimated \(estimatedBytes, privacy: .public)B"
                )
                return
            }

            let manifestJSON = String(
                decoding: try JSONEncoder().encode(manifest), as: UTF8.self)
            try? fm.removeItem(at: tmpURL)
            try save(
                arrays: weights,
                metadata: [manifestKey: manifestJSON],
                url: tmpURL
            )
            let url = artifactURL(for: directory)
            try? fm.removeItem(at: url)
            try fm.moveItem(at: tmpURL, to: url)
            let seconds = (clock.now - start) / .seconds(1)
            logger.notice(
                "Prepared Checkpoint written: \(estimatedBytes, privacy: .public)B in \(String(format: "%.1f", seconds), privacy: .public)s"
            )
        } catch {
            logger.warning(
                "Prepared Checkpoint write failed (\(String(describing: error), privacy: .public)) — next load converts again"
            )
            try? fm.removeItem(at: tmpURL)
        }
    }
}

/// Transport for the background persist `Task` — `MLXArray` is not
/// `Sendable`; the captured arrays are the immutable, already-evaluated
/// checkpoint tensors shared with the live model, read-only on both sides.
struct PreparedCheckpointPayload: @unchecked Sendable {
    let weights: [String: MLXArray]
    let manifest: ParoQuantPreparedCheckpoint.Manifest
    let directory: URL
}
