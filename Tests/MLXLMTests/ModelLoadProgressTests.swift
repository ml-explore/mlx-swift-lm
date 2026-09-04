// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Model load progress")
struct ModelLoadProgressTests {

    @Test("the handlers group in one value")
    func handlersGroupBothPhases() {
        let weights: @Sendable (Progress) -> Void = { _ in }
        let download: @Sendable (Progress) -> Void = { _ in }

        let both = LoadProgressHandlers(download: download, weights: weights)
        #expect(both.download != nil)
        #expect(both.weights != nil)

        // the memberwise init has defaults, so the empty value exists
        let none = LoadProgressHandlers()
        #expect(none.download == nil)
        #expect(none.weights == nil)
    }

    @Test("the static factories select a single phase")
    func handlerFactories() {
        let handler: @Sendable (Progress) -> Void = { _ in }

        #expect(LoadProgressHandlers.weights(handler).weights != nil)
        #expect(LoadProgressHandlers.weights(handler).download == nil)
        #expect(LoadProgressHandlers.download(handler).download != nil)
        #expect(LoadProgressHandlers.download(handler).weights == nil)
    }

    /// Recorder for the `Progress` published by ``ModelLoadProgressReporter``.
    final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var values = [(completed: Int64, total: Int64)]()

        func record(_ progress: Progress) {
            lock.withLock {
                values.append((progress.completedUnitCount, progress.totalUnitCount))
            }
        }

        var completed: [Int64] {
            lock.withLock { values.map(\.completed) }
        }

        var fractions: [Double] {
            lock.withLock {
                values.map { $0.total > 0 ? Double($0.completed) / Double($0.total) : 0 }
            }
        }
    }

    private func makeDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func write(_ byteCount: Int, to url: URL) throws {
        try Data(repeating: 0, count: byteCount).write(to: url)
    }

    @Test("byte count sums the shards and ignores other files")
    func safetensorsByteCountSumsShards() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try write(1024, to: directory.appending(path: "model-00001-of-00002.safetensors"))
        try write(2048, to: directory.appending(path: "model-00002-of-00002.safetensors"))

        // not weights -- must not be counted
        try write(64, to: directory.appending(path: "config.json"))
        try write(64, to: directory.appending(path: "tokenizer.json"))

        #expect(safetensorsByteCount(in: directory) == 3072)
    }

    @Test("byte count follows the safetensors index")
    func safetensorsByteCountUsesIndex() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try write(1024, to: directory.appending(path: "model-00001-of-00002.safetensors"))
        try write(2048, to: directory.appending(path: "model-00002-of-00002.safetensors"))
        try write(4096, to: directory.appending(path: "stale.safetensors"))

        let index = """
            {
              "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors"
              }
            }
            """
        try Data(index.utf8).write(
            to: directory.appending(path: "model.safetensors.index.json"))

        #expect(safetensorsByteCount(in: directory) == 3072)
    }

    @Test("byte count of a directory without weights is zero")
    func safetensorsByteCountWithoutWeights() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try write(64, to: directory.appending(path: "config.json"))

        #expect(safetensorsByteCount(in: directory) == 0)
        #expect(safetensorsByteCount(in: directory.appending(path: "missing")) == 0)
    }

    @Test("progress of the individual files is aggregated")
    func reporterAggregatesAcrossFiles() {
        let first = URL(filePath: "/tmp/first.safetensors")
        let second = URL(filePath: "/tmp/second.safetensors")

        let recorder = Recorder()
        let reporter = ModelLoadProgressReporter(totalUnitCount: 300) { recorder.record($0) }

        reporter.update(.init(url: first, completedUnitCount: 0, totalUnitCount: 100))
        reporter.update(.init(url: second, completedUnitCount: 0, totalUnitCount: 200))
        reporter.update(.init(url: first, completedUnitCount: 100, totalUnitCount: 100))
        reporter.update(.init(url: second, completedUnitCount: 200, totalUnitCount: 200))

        // the last update of each file, summed
        #expect(recorder.completed.last == 300)
        #expect(recorder.fractions.last == 1)
        #expect(recorder.completed == recorder.completed.sorted())
    }

    @Test("updates are coalesced")
    func reporterCoalescesUpdates() {
        let url = URL(filePath: "/tmp/weights.safetensors")
        let total: Int64 = 1_000_000

        let recorder = Recorder()
        let reporter = ModelLoadProgressReporter(totalUnitCount: total) { recorder.record($0) }

        // one update per byte would be 1_000_000 callbacks
        for completed in stride(from: Int64(0), through: total, by: 1) {
            reporter.update(
                .init(url: url, completedUnitCount: completed, totalUnitCount: total))
        }

        #expect(recorder.completed.count <= 1001)
        #expect(recorder.completed.first == 0)
    }

    @Test("finish publishes completion")
    func reporterFinishCompletes() {
        let url = URL(filePath: "/tmp/weights.safetensors")

        let recorder = Recorder()
        let reporter = ModelLoadProgressReporter(totalUnitCount: 1000) { recorder.record($0) }

        // a model whose weights are partly dropped by sanitize() never reads every byte
        reporter.update(.init(url: url, completedUnitCount: 400, totalUnitCount: 1000))
        #expect(recorder.fractions.last == 0.4)

        reporter.finish()
        #expect(recorder.fractions.last == 1)
    }

    @Test("a model without weights still completes")
    func reporterWithoutWeights() {
        let recorder = Recorder()
        let reporter = ModelLoadProgressReporter(totalUnitCount: 0) { recorder.record($0) }

        reporter.finish()

        #expect(recorder.completed.last == 0)
        #expect(recorder.fractions.last == 0)
    }
}
