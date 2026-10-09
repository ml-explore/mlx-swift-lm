// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

private final class Connector: Module {
    @ModuleInfo var gate: Linear
    @ModuleInfo var projection: Linear?

    override init() {
        _gate.wrappedValue = Linear(64, 64, bias: false)
        _projection.wrappedValue = Linear(64, 64, bias: false)
    }
}

/// The vision modules of ``Captioner``, at the paths they have in the model.
private final class CaptionerVision: Module {
    final class Projection: Module {
        @ModuleInfo var projection = Linear(64, 64, bias: false)
    }

    @ModuleInfo(key: "vision_tower") var tower = Linear(64, 64, bias: false)
    @ModuleInfo var connector = Projection()
}

/// The dispatch queue that last built `CaptionerVision`.
private let loadQueueLabel = Mutexish<String>("")

private final class Mutexish<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value
    init(_ value: Value) { self.value = value }
    func withLock<R>(_ body: (inout Value) -> R) -> R { lock.withLock { body(&value) } }
}

/// A text model with a vision tower, and a vision projection nested in a module it shares.
private final class Captioner: Module, BaseLanguageModel, ModelComponentsProviding {
    @ModuleInfo(key: "language_model") var language: Linear
    @ModuleInfo(key: "vision_tower") var tower: Linear?
    @ModuleInfo var connector: Connector

    let vision = OnDemandComponent<CaptionerVision>(.vision) {
        loadQueueLabel.withLock { $0 = String(cString: __dispatch_queue_get_label(nil)) }
        return CaptionerVision()
    }
    var onDemandComponents: [AnyOnDemandComponent] { [vision] }

    override init() {
        _language.wrappedValue = Linear(64, 64, bias: false)
        _tower.wrappedValue = Linear(64, 64, bias: false)
        _connector.wrappedValue = Connector()
    }

    var modelComponents: [ModelComponent: [CheckpointComponent]] {
        [
            .vision: [
                CheckpointComponent(
                    name: "tower", namespaces: ["model.vision_tower"], destination: "vision_tower"),
                CheckpointComponent(
                    name: "projection", namespaces: ["model.connector.projection"],
                    destination: "connector.projection"),
            ]
        ]
    }

    /// Names of the tensors the loader passed to ``prepareCheckpoint(_:)``.
    var loadedNames = Set<String>()

    /// When set, `prepareCheckpoint` signals `entered` and then waits for `release`.
    var prepareGate: (entered: DispatchSemaphore, release: DispatchSemaphore)?

    func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        if let prepareGate {
            prepareGate.entered.signal()
            prepareGate.release.wait()
        }
        loadedNames = Set(checkpoint.weights.keys)
        var checkpoint = checkpoint
        checkpoint.weights = dropModelScope(checkpoint.weights)
        return checkpoint
    }
}

/// The same layout, without support for excluding components.
private final class FixedCaptioner: Module, BaseLanguageModel {
    @ModuleInfo(key: "language_model") var language = Linear(64, 64, bias: false)
    @ModuleInfo(key: "vision_tower") var tower: Linear? = Linear(64, 64, bias: false)
    @ModuleInfo var connector = Connector()

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        dropModelScope(weights)
    }
}

/// Raw checkpoints scope everything under `model.`, as `transformers` exports do.
private func dropModelScope(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    Dictionary(
        uniqueKeysWithValues: weights.map { (String($0.dropFirst("model.".count)), $1) })
}

/// Declares a component whose module is not optional.
private final class MisdeclaredCaptioner: Module, BaseLanguageModel, ModelComponentsProviding {
    @ModuleInfo(key: "language_model") var language = Linear(64, 64, bias: false)

    var modelComponents: [ModelComponent: [CheckpointComponent]] {
        [.vision: [.init(name: "language", namespaces: [], destination: "language_model")]]
    }
}

final class ModelComponentTests: XCTestCase {

    private let quantization = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
    private let visionNamespaces = ["model.vision_tower", "model.connector.projection"]

    func testExcludedComponentLoadsWithoutItsTensors() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        let model = Captioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: model, quantization: quantization,
            excludedComponents: [.vision])

        XCTAssertEqual(model.loadedNames, Set(raw(checkpoint.text).keys))
        XCTAssertNil(model.tower)
        XCTAssertNil(model.connector.projection)
        XCTAssertTrue(model.language is QuantizedLinear)
        try assertParameters(of: model, match: checkpoint.text)
    }

    func testOnDemandComponentLoadsOnFirstUseByDefaultWithoutChangingTheModel() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        let model = Captioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: model, quantization: quantization)

        XCTAssertEqual(model.loadedNames, Set(raw(checkpoint.text).keys))
        XCTAssertNil(model.tower)
        XCTAssertFalse(model.vision.isLoaded)
        let parameters = model.parameters().flattened().map(\.0)

        let vision = try model.vision.load()

        XCTAssertTrue(model.vision.isLoaded)
        XCTAssertTrue(vision.connector.projection is QuantizedLinear)
        try assertParameters(of: vision, match: checkpoint.visual)
        XCTAssertTrue(try model.vision.load() === vision)
        XCTAssertNil(model.tower)
        XCTAssertEqual(model.parameters().flattened().map(\.0), parameters)
    }

    func testConcurrentFirstUsesLoadOnce() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }
        let model = Captioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: model, quantization: quantization,
            componentLoading: .onFirstUse)

        final class Loads: @unchecked Sendable {
            let lock = NSLock()
            var modules = Set<ObjectIdentifier?>()
        }
        let loads = Loads()
        let vision = model.vision
        DispatchQueue.concurrentPerform(iterations: 8) { _ in
            let module = try? vision.load()
            loads.lock.withLock { _ = loads.modules.insert(module.map(ObjectIdentifier.init)) }
        }

        XCTAssertEqual(
            loads.modules, [ObjectIdentifier(try vision.load())])
    }

    /// Async callers load before prefill on a GCD queue, never on a Swift concurrency thread.
    func testMediaInputLoadsPendingComponentsOffTheConcurrencyPool() async throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }
        let model = Captioner()
        try await loadWeights(
            modelDirectory: checkpoint.directory, model: model, quantization: quantization)
        let text = LMInput(tokens: MLXArray([1, 2] as [Int32]))
        let image = LMInput(
            text: text.text, image: .init(pixels: MLXArray.zeros([1, 3, 2, 2])))

        XCTAssertEqual(ModelComponent.needed(by: text), [])
        XCTAssertEqual(ModelComponent.needed(by: image), [.vision])
        let pending = pendingOnDemandComponents(
            of: model, among: ModelComponent.needed(by: image))
        XCTAssertEqual(pending.map(\.component), [.vision])

        try await loadOnDemandComponents(pending)

        XCTAssertTrue(model.vision.isLoaded)
        let label = loadQueueLabel.withLock { $0 }
        XCTAssertTrue(label.hasPrefix("com.apple.root."), label)
        XCTAssertFalse(label.contains("cooperative"), label)
        XCTAssertTrue(
            pendingOnDemandComponents(of: model, among: ModelComponent.needed(by: image)).isEmpty)
    }

    /// A state query during a load returns at once. `ModelContainer.generate` asks for pending
    /// components while it holds the container, so a wait here would block text requests too.
    func testStateQueriesDoNotWaitForALoadInProgress() async throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }
        let model = Captioner()
        try await loadWeights(
            modelDirectory: checkpoint.directory, model: model, quantization: quantization)
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        model.prepareGate = (entered, release)

        let vision = model.vision
        let load = Task { try await vision.loadInBackground() }
        let entry = await wait(for: entered, seconds: 5)
        XCTAssertEqual(entry, .success)

        let container = SerialAccessContainer(model)
        let inspected = DispatchSemaphore(value: 0)
        let inspection = Task {
            let pending = await container.read { model in
                pendingOnDemandComponents(of: model, among: [.vision]).map(\.component)
            }
            inspected.signal()
            return (pending, vision.isLoaded)
        }
        let answered = await wait(for: inspected, seconds: 5)
        XCTAssertEqual(answered, .success, "state query waited for the load")

        release.signal()
        let (pending, loadedDuringInspection) = await inspection.value
        XCTAssertEqual(pending, [.vision])
        XCTAssertFalse(loadedDuringInspection)
        try await load.value
        XCTAssertTrue(vision.isLoaded)
        XCTAssertTrue(pendingOnDemandComponents(of: model, among: [.vision]).isEmpty)
    }

    private func wait(for semaphore: DispatchSemaphore, seconds: Double) async
        -> DispatchTimeoutResult
    {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                continuation.resume(returning: semaphore.wait(timeout: .now() + seconds))
            }
        }
    }

    func testExcludedOrImmediateComponentsDoNotLoadOnDemand() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        let excluded = Captioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: excluded, quantization: quantization,
            excludedComponents: [.vision], componentLoading: .onFirstUse)
        let immediate = Captioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: immediate, quantization: quantization,
            componentLoading: .immediate)

        XCTAssertNil(excluded.tower)
        XCTAssertNotNil(immediate.tower)
        for model in [excluded, immediate] {
            XCTAssertThrowsError(try model.vision.load()) { error in
                guard case OnDemandComponentError.unavailable(.vision) = error else {
                    return XCTFail("expected unavailable, got \(error)")
                }
            }
        }
    }

    /// Lazy and immediate loads of one model instance agree with fresh loads.
    func testReloadingSwitchesBetweenLazyAndImmediate() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }
        let model = Captioner()
        func load(_ loading: ComponentLoading) throws {
            try loadWeights(
                modelDirectory: checkpoint.directory, model: model, quantization: quantization,
                componentLoading: loading)
        }

        try load(.onFirstUse)
        try load(.onFirstUse)
        XCTAssertNil(model.tower)

        try load(.immediate)
        XCTAssertFalse(model.vision.isLoaded)
        try assertParameters(
            of: model, match: checkpoint.text.merging(checkpoint.visual) { a, _ in a })

        try load(.onFirstUse)
        XCTAssertNil(model.tower)
        try assertParameters(of: try model.vision.load(), match: checkpoint.visual)
    }

    func testImmediateLoadingLoadsEveryComponent() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        let model = Captioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: model, quantization: quantization,
            componentLoading: .immediate)

        XCTAssertNotNil(model.tower)
        XCTAssertTrue(model.connector.projection is QuantizedLinear)
        try assertParameters(
            of: model, match: checkpoint.text.merging(checkpoint.visual) { a, _ in a })
    }

    func testComponentsAModelCannotExcludeStillLoad() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        let fixed = FixedCaptioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: fixed, quantization: quantization,
            excludedComponents: [.vision])
        XCTAssertNotNil(fixed.tower)

        let model = Captioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: model, quantization: quantization,
            excludedComponents: [ModelComponent("audio")], componentLoading: .immediate)
        XCTAssertNotNil(model.tower)
    }

    func testComponentDestinationMustBeAnOptionalModule() {
        XCTAssertThrowsError(
            try prepareComponents(
                of: MisdeclaredCaptioner(), urls: [], excluded: [.vision], loading: .onFirstUse,
                perLayerQuantization: nil)
        ) { error in
            XCTAssertTrue(error is UpdateError, "\(error)")
        }
    }

    func testExcludedNamespacesKeepTheMetadataOfSkippedFiles() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        let loaded = try loadModelCheckpoint(
            urls: [checkpoint.textFile, checkpoint.vision], excludedNamespaces: visionNamespaces)

        XCTAssertEqual(Set(loaded.weights.keys), Set(raw(checkpoint.text).keys))
        XCTAssertEqual(loaded.metadata, ["source": "vision shard"])
    }

    func testExcludedNamespacesMatchWholePathComponents() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("model.safetensors")
        try save(
            arrays: [
                "model.vision_tower.weight": MLXArray.zeros([2]),
                "model.vision_tower_norm.weight": MLXArray.zeros([2]),
            ], url: url)

        let loaded = try loadModelCheckpoint(
            urls: [url], excludedNamespaces: ["model.vision_tower"])

        XCTAssertEqual(Set(loaded.weights.keys), ["model.vision_tower_norm.weight"])
    }

    // MARK: - Fixtures

    private struct Checkpoint {
        let directory: URL
        let textFile: URL
        let vision: URL
        let text: [String: MLXArray]
        let visual: [String: MLXArray]
    }

    /// Writes a raw checkpoint in two shards: text weights, then vision weights.
    ///
    /// The language model and the vision projection are quantized; the tower is not.
    private func writeCheckpoint() throws -> Checkpoint {
        let directory = try makeTemporaryDirectory()
        let source = Captioner()
        quantize(model: source, groupSize: 64, bits: 4) { path, _ in
            path == "language_model" || path == "connector.projection"
        }
        var text = [String: MLXArray]()
        var visual = [String: MLXArray]()
        for (key, value) in source.parameters().flattened() {
            if key.hasPrefix("vision_tower") || key.hasPrefix("connector.projection") {
                visual[key] = value
            } else {
                text[key] = value
            }
        }
        let textFile = directory.appendingPathComponent("model-00001-of-00002.safetensors")
        let vision = directory.appendingPathComponent("model-00002-of-00002.safetensors")
        try save(arrays: raw(text), url: textFile)
        try save(arrays: raw(visual), metadata: ["source": "vision shard"], url: vision)
        return Checkpoint(
            directory: directory, textFile: textFile, vision: vision, text: text, visual: visual)
    }

    private func raw(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: weights.map { ("model.\($0)", $1) })
    }

    private func assertParameters(of model: Module, match expected: [String: MLXArray]) throws {
        let actual = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(Set(actual.keys), Set(expected.keys))
        for (key, value) in expected {
            XCTAssertTrue(arrayEqual(try XCTUnwrap(actual[key]), value).item(Bool.self), key)
        }
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelComponentTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
