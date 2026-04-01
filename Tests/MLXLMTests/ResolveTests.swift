import Foundation
import MLXLMCommon
import Testing

/// A mock downloader that records every call for later assertion.
private struct MockDownloader: Downloader {

    struct Call: Equatable, Sendable {
        let id: String
        let revision: String?
        let patterns: [String]
    }

    let calls: LockIsolated<[Call]>
    let directory: URL

    init(directory: URL = URL(filePath: "/mock")) {
        self.calls = LockIsolated([])
        self.directory = directory
    }

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        calls.withLock { $0.append(Call(id: id, revision: revision, patterns: patterns)) }
        // Return a unique directory per id so tests can distinguish model vs tokenizer paths.
        return directory.appending(component: id.replacingOccurrences(of: "/", with: "_"))
    }
}

/// Minimal lock-based isolation for collecting values from async contexts.
private final class LockIsolated<Value: Sendable>: @unchecked Sendable {
    private var _value: Value
    private let lock = NSLock()

    init(_ value: Value) { _value = value }

    func withLock<R>(_ body: (inout Value) -> R) -> R {
        lock.lock()
        defer { lock.unlock() }
        return body(&_value)
    }

    var value: Value {
        lock.lock()
        defer { lock.unlock() }
        return _value
    }
}

@Suite struct ResolveTests {

    @Test func nilTokenizerSourceUsesModelDirectory() async throws {
        let downloader = MockDownloader()
        let config = ModelConfiguration(
            id: "org/model", revision: "abc123", tokenizerSource: nil)

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        // Only one download call — the model itself.
        #expect(downloader.calls.value.count == 1)
        #expect(downloader.calls.value[0].id == "org/model")
        #expect(downloader.calls.value[0].revision == "abc123")
        #expect(downloader.calls.value[0].patterns.contains("*.jinja"))

        // No separate tokenizer download, so both point to the model directory.
        #expect(resolved.modelDirectory == resolved.tokenizerDirectory)
    }

    @Test func tokenizerSourceIDWithoutRevisionPassesNil() async throws {
        let downloader = MockDownloader()
        let config = ModelConfiguration(
            id: "org/model", revision: "abc123",
            tokenizerSource: .id("org/tokenizer"))

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        #expect(downloader.calls.value.count == 2)

        // Model download uses model revision.
        #expect(downloader.calls.value[0].id == "org/model")
        #expect(downloader.calls.value[0].revision == "abc123")

        // Tokenizer download uses nil revision (provider default).
        #expect(downloader.calls.value[1].id == "org/tokenizer")
        #expect(downloader.calls.value[1].revision == nil)
        #expect(downloader.calls.value[1].patterns.contains("*.jinja"))

        // Model and tokenizer come from different repos, so directories differ.
        #expect(resolved.modelDirectory != resolved.tokenizerDirectory)
    }

    @Test func tokenizerSourceIDWithExplicitRevision() async throws {
        let downloader = MockDownloader()
        let config = ModelConfiguration(
            id: "org/model", revision: "v1.0",
            tokenizerSource: .id("org/tokenizer", revision: "tok-v2"))

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        #expect(downloader.calls.value.count == 2)

        #expect(downloader.calls.value[0].id == "org/model")
        #expect(downloader.calls.value[0].revision == "v1.0")

        #expect(downloader.calls.value[1].id == "org/tokenizer")
        #expect(downloader.calls.value[1].revision == "tok-v2")
        #expect(downloader.calls.value[1].patterns.contains("*.jinja"))

        // Model and tokenizer come from different repos, so directories differ.
        #expect(resolved.modelDirectory != resolved.tokenizerDirectory)
    }

    @Test func localDirectorySkipsDownloader() async throws {
        let downloader = MockDownloader()
        let localDir = URL(filePath: "/local/org/model")
        let config = ModelConfiguration(directory: localDir)

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        // No downloads should occur for a local directory.
        #expect(downloader.calls.value.isEmpty)

        // Both directories point to the local path.
        #expect(resolved.modelDirectory == localDir)
        #expect(resolved.tokenizerDirectory == localDir)
    }

    @Test func localDirectoryWithRemoteTokenizerSource() async throws {
        let downloader = MockDownloader()
        let localDir = URL(filePath: "/local/org/model")
        let config = ModelConfiguration(
            directory: localDir,
            tokenizerSource: .id("org/tokenizer", revision: "v3"))

        let resolved = try await resolve(
            configuration: config, from: downloader,
            useLatest: false, progressHandler: { _ in })

        // Only the tokenizer is downloaded; the model directory is local.
        #expect(downloader.calls.value.count == 1)
        #expect(downloader.calls.value[0].id == "org/tokenizer")
        #expect(downloader.calls.value[0].revision == "v3")
        #expect(downloader.calls.value[0].patterns.contains("*.jinja"))

        #expect(resolved.modelDirectory == localDir)
        #expect(resolved.tokenizerDirectory != localDir)
    }

    @Test func localConfigurationExposesResolvedDirectories() throws {
        let modelDir = URL(filePath: "/local/org/model")
        let tokenizerDir = URL(filePath: "/local/org/tokenizer")
        let config = ModelConfiguration(
            directory: modelDir,
            tokenizerSource: .directory(tokenizerDir))

        #expect(try config.modelDirectory == modelDir)
        #expect(try config.tokenizerDirectory == tokenizerDir)
    }

    @Test func tokenizerDirectoryFallsBackToModelDirectory() throws {
        let modelDir = URL(filePath: "/local/org/model")
        let config = ModelConfiguration(directory: modelDir)

        #expect(try config.modelDirectory == modelDir)
        #expect(try config.tokenizerDirectory == modelDir)
    }

    @Test func unresolvedRemoteConfigurationThrowsForDirectories() {
        let config = ModelConfiguration(
            id: "org/model",
            tokenizerSource: .id("org/tokenizer"))

        do {
            _ = try config.modelDirectory
            Issue.record("Expected modelDirectory to throw for unresolved remote config")
        } catch let error as ModelConfiguration.DirectoryError {
            #expect(error == .unresolvedModelDirectory("org/model"))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        do {
            _ = try config.tokenizerDirectory
            Issue.record("Expected tokenizerDirectory to throw for unresolved remote tokenizer")
        } catch let error as ModelConfiguration.DirectoryError {
            #expect(error == .unresolvedTokenizerDirectory("org/tokenizer"))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}
