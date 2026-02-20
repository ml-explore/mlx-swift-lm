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

        // When useLatest is false, return cached files without a network call
        // if available. Uses cached repo info to verify all matching files are
        // present, avoiding the incomplete snapshot problem.
        // If nothing is cached, fall through to download.
        if !useLatest {
            if let cached = cachedSnapshotPath(
                repo: repoID, revision: revision, matching: patterns
            ) {
                return cached
            }
        }

        return try await downloadSnapshot(
            of: repoID,
            revision: revision,
            matching: patterns,
            progressHandler: progressHandler
        )
    }
}
