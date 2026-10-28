// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// A `LanguageModel` that forwards every requirement to a wrapped model.
///
/// Subclass it to change one behavior and keep the rest. It implements every
/// `LanguageModel` requirement itself, so a subclass override is always reached and no
/// requirement silently falls back to a protocol default.
///
/// As a `Module` it is transparent: parameters, children, updates, and freezing go to the
/// wrapped model with its own keys, not under `wrapped.`. So checkpoint loading,
/// `sanitize(weights:)`, quantization, and LoRA work through the wrapper.
///
/// Optional capabilities that the library detects with `as?`:
/// - `LoRAModel` and `SuppressedTokensProviding` are forwarded.
/// - `PreparedInputSplitting`, `SpeculativeCacheRewindModel`, and
///   `ModelComponentsProviding` are not: conformance cannot depend on the wrapped model,
///   and conforming with a no-op would change behavior.
open class ForwardingLanguageModel: Module, LanguageModel, LoRAModel, SuppressedTokensProviding {

    public let wrapped: any LanguageModel

    private var localSuppressedTokenIds = Set<Int>()

    public init(wrapping wrapped: any LanguageModel) {
        self.wrapped = wrapped
        super.init()
    }

    // MARK: - Module

    open override func items() -> ModuleItems {
        wrapped.items()
    }

    @discardableResult
    open override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        try wrapped.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        return self
    }

    @discardableResult
    open override func update(
        modules: ModuleChildren, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        try wrapped.update(modules: modules, verify: verify, path: path, modulePath: modulePath)
        return self
    }

    open override func updateModule(key: String, _ value: Any) throws {
        try wrapped.updateModule(key: key, value)
    }

    open override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false)
        throws
    {
        try wrapped.freeze(recursive: recursive, keys: keys, strict: strict)
    }

    open override func unfreeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        try wrapped.unfreeze(recursive: recursive, keys: keys, strict: strict)
    }

    open override func noGrad() -> Set<String> {
        wrapped.noGrad()
    }

    /// `train(_:)` visits the wrapped model's children but not the wrapped model itself.
    open override func didSetTrain(_ mode: Bool) {
        wrapped.train(mode)
    }

    // MARK: - BaseLanguageModel

    open var excludedCheckpointNamespaces: [String] { wrapped.excludedCheckpointNamespaces }

    open func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        try wrapped.prepareCheckpoint(checkpoint)
    }

    open func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try wrapped.sanitize(weights: weights)
    }

    open func sanitize(weights: [String: MLXArray], metadata: [String: String]) throws
        -> [String: MLXArray]
    {
        try wrapped.sanitize(weights: weights, metadata: metadata)
    }

    // MARK: - LanguageModel

    open func prepare() throws {
        try wrapped.prepare()
    }

    open func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        try wrapped.prepare(input, cache: cache, state: state, prefill: prefill)
    }

    open func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        wrapped.callAsFunction(input, cache: cache, state: state)
    }

    open func nextTokenLogits(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        wrapped.nextTokenLogits(input, cache: cache, state: state)
    }

    open func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        wrapped.callAsFunction(inputs, cache: cache)
    }

    open func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try wrapped.newCache(parameters: parameters)
    }

    open func cacheStatus(parameters: GenerateParameters?) throws -> KVCacheStatus {
        try wrapped.cacheStatus(parameters: parameters)
    }

    // MARK: - ChatConventionsProviding

    open var toolCallFormat: ToolCallFormat? { wrapped.toolCallFormat }

    open var reasoningConfig: ReasoningConfig? { wrapped.reasoningConfig }

    // MARK: - LoRAModel

    open var loraLayers: [Module] { (wrapped as? LoRAModel)?.loraLayers ?? [] }

    open var loraDefaultKeys: [String] {
        if let lora = wrapped as? LoRAModel {
            return lora.loraDefaultKeys
        }
        return []
    }

    // MARK: - SuppressedTokensProviding

    open var suppressedTokenIds: Set<Int> {
        get {
            (wrapped as? SuppressedTokensProviding)?.suppressedTokenIds
                ?? localSuppressedTokenIds
        }
        set {
            if let provider = wrapped as? SuppressedTokensProviding {
                provider.suppressedTokenIds = newValue
            } else {
                localSuppressedTokenIds = newValue
            }
        }
    }
}
