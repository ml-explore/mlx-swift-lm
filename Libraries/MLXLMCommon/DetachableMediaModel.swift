// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Where a model reads the media weights it loads on first use.
public struct MediaWeightSource: Sendable {
    public let modelDirectory: URL
    public let weightFileSelection: WeightFileSelection
    public let quantization: BaseConfiguration.Quantization?
    public let perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    public init(
        modelDirectory: URL,
        weightFileSelection: WeightFileSelection = .automatic,
        quantization: BaseConfiguration.Quantization? = nil,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) {
        self.modelDirectory = modelDirectory
        self.weightFileSelection = weightFileSelection
        self.quantization = quantization
        self.perLayerQuantization = perLayerQuantization
    }
}

/// A model whose media modules, a vision or audio tower and its projector, load on first use.
///
/// Factories detach the modules before loading, so a text-only session never reads their weights.
/// The first input with media calls ``loadMediaModules()``, which reads, verifies and evaluates
/// them apart from the model and only then attaches them: the model never holds unevaluated
/// arrays, and the language model is never touched. `sanitize(weights:metadata:)` must not depend
/// on whether they are attached, and ``LanguageModel/prepare()`` does not run again afterwards.
public protocol DetachableMediaModel: BaseLanguageModel {
    /// Top-level module keys of the media modules, such as `vision_tower`.
    ///
    /// Checkpoint names of media weights contain the key, before and after `sanitize`.
    var mediaModuleKeys: [String] { get }

    /// New, unloaded media modules keyed by ``mediaModuleKeys``.
    func makeMediaModules() -> [String: Module]

    /// Where ``loadMediaModules()`` reads the weights, or `nil` if it cannot load them.
    var mediaWeightSource: MediaWeightSource? { get set }
}

extension DetachableMediaModel {

    /// Whether the media modules are in the module tree.
    public var mediaModulesAreAttached: Bool {
        mediaModuleKeys.allSatisfy { key in
            if case .value(.module)? = items()[key] { return true }
            return false
        }
    }

    /// Remove the media modules from the module tree and release their weights.
    ///
    /// The next input with media loads them again from ``mediaWeightSource``.
    public func detachMediaModules() throws {
        let detached = mediaModuleKeys.map { ($0, NestedItem<String, Module>.none) }
        try update(
            modules: ModuleChildren(values: Dictionary(uniqueKeysWithValues: detached)),
            verify: .all)
    }

    /// Remove the media modules so the first input with media loads them from `source`.
    public func detachMediaModules(loadingFrom source: MediaWeightSource) throws {
        mediaWeightSource = source
        try detachMediaModules()
    }

    /// Load and attach the media modules if they are detached and ``mediaWeightSource`` is set.
    ///
    /// Call it with exclusive access to the model, as during prefill.
    public func loadMediaModules() throws {
        guard !mediaModulesAreAttached, let source = mediaWeightSource else { return }

        let prefixes = mediaWeightPrefixes
        let urls = try safetensorWeightURLs(
            in: source.modelDirectory,
            selection: source.weightFileSelection,
            additionalFiles: (self as? any AdditionalWeightFilesProviding)?.additionalWeightFiles
                ?? [])
        let (loaded, metadata) = try loadWeightArrays(urls: urls) {
            matchesWeightPrefixes($0, prefixes: prefixes)
        }
        let weights = sanitize(weights: loaded, metadata: metadata)

        let staged = MediaModuleStage(makeMediaModules())
        quantizeCheckpointLayers(
            of: staged, weights: weights, quantization: source.quantization,
            perLayerQuantization: source.perLayerQuantization)
        try staged.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(staged)

        try attach(staged.modules)
    }

    package func attach(_ modules: [String: Module]) throws {
        try update(modules: ModuleChildren(values: modules.mapValues { .value($0) }), verify: .all)
    }

    var mediaWeightPrefixes: [String] { mediaModuleKeys.map { $0 + "." } }
}

/// Detaches the media modules of `model`, if it has any.
///
/// A free function, not a cast at the call site: an `async` loader that casts and calls on the
/// model it goes on to return as `sending` fails region isolation.
package func detachMediaModulesIfSupported(
    of model: any BaseLanguageModel, loadingFrom source: MediaWeightSource
) throws {
    try (model as? any DetachableMediaModel)?.detachMediaModules(loadingFrom: source)
}

/// Whether `key` names a weight under one of `prefixes`, at its start or after a `.`.
///
/// The second form matches raw checkpoint names with an extra scope, such as
/// `model.vision_tower...`, before `sanitize` removes it.
func matchesWeightPrefixes(_ key: String, prefixes: [String]) -> Bool {
    prefixes.contains { key.hasPrefix($0) || key.contains("." + $0) }
}

/// Holds the media modules under their model keys while they load, so checkpoint paths apply.
private final class MediaModuleStage: Module {
    private(set) var modules: [String: Module]

    init(_ modules: [String: Module]) {
        self.modules = modules
        super.init()
    }

    override func items() -> ModuleItems {
        ModuleItems(values: modules.mapValues { .value(.module($0)) })
    }

    // Quantization replaces a top-level module, such as a projection `Linear`, through here.
    override func updateModule(key: String, _ value: Any) throws {
        guard modules[key] != nil, let module = value as? Module else {
            throw UpdateError.unableToSet("media module \(key)")
        }
        modules[key] = module
    }
}
