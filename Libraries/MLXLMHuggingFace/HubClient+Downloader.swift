import Foundation
import HuggingFace
import MLXLMCommon

public enum HuggingFaceDownloaderError: LocalizedError {
    case invalidRepositoryID(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let id):
            return "Invalid Hugging Face repository ID: '\(id)'. Expected format 'namespace/name'."
        }
    }
}

extension HubClient: Downloader {

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard let repoID = Repo.ID(rawValue: id) else {
            throw HuggingFaceDownloaderError.invalidRepositoryID(id)
        }
        let revision = revision ?? "main"

        // resolveCachedSnapshot resolves refs locally and verifies file presence
        // on disk, avoiding a network round-trip (100-200ms). The cached ref
        // may not reflect the latest remote version, so we only use it when freshness isn't
        // needed. If nothing is cached, fall through to download.
        if !useLatest {
            if let cached = resolveCachedSnapshot(
                repo: repoID, revision: revision, matching: patterns
            ) {
                return cached
            }
        }

        // downloadSnapshot always checks the network for branch names,
        // ensuring useLatest = true gets the latest commit.
        return try await downloadSnapshot(
            of: repoID,
            revision: revision,
            matching: patterns,
            progressHandler: { @MainActor progress in
                progressHandler(progress)
            }
        )
    }
}
