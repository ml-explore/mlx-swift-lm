// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Generator for block-diffusion language models.
///
/// The model first encodes the prompt into a KV cache, then repeatedly denoises
/// a full canvas. The iterator exposes the finalized canvas tokens one by one
/// so existing detokenization, EOS handling, tool parsing, and timing code can
/// remain shared with autoregressive iterators.
public struct BlockDiffusionTokenIterator: TokenIteratorProtocol {
    let model: any BlockDiffusionLanguageModel
    let cacheStorage: KVCacheStorage
    var cache: [KVCache] {
        get { cacheStorage.cache }
        set { cacheStorage.replace(with: newValue) }
    }
    var kvCachePlan: KVCachePlan { cacheStorage.plan }
    let prefillStepSize: Int?
    let minCanvasLength: Int
    let maxCanvasLength: Int
    let maxDenoisingSteps: Int
    let entropyBound: Float
    let temperatureMin: Float
    let temperatureMax: Float
    package let denoiserTemperature: Float
    let diffusionSampler: BlockDiffusionParameters.Sampler
    let prefersLogitsSelfConditioning: Bool
    let selfConditioningWeight: MLXArray?
    let stabilityThreshold: Int
    let confidenceThreshold: Float

    public var tokenCount = 0
    public let maxTokens: Int?
    public var promptPrefillTime: TimeInterval = 0.0

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    private var randomState: MLXRandom.RandomState
    private var committedPendingIndex = 0
    private var argmaxCanvasHistory: [MLXArray]?

    public init(
        input: LMInput,
        model: any BlockDiffusionLanguageModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        let plan = try parameters.kvCachePlan()
        try self.init(
            input: input, model: model,
            cacheStorage: KVCacheStorage(
                cache ?? (try model.newCache(parameters: parameters)), plan: plan),
            parameters: parameters)
    }

    package init(
        input: LMInput,
        model: any BlockDiffusionLanguageModel,
        cacheStorage: KVCacheStorage,
        parameters: GenerateParameters
    ) throws {
        let promptTokenCount = try Self.validatePrompt(input.text, model: model)
        let kvCachePlan = cacheStorage.plan
        let cacheStorage = try kvCachePlan.validated(cacheStorage)

        self.model = model
        self.cacheStorage = cacheStorage
        self.prefillStepSize = parameters.prefill.stepSize
        self.maxTokens = parameters.maxTokens ?? model.diffusionDefaultMaxTokens
        let resolvedCanvasBounds: (minimum: Int, maximum: Int) =
            switch parameters.diffusion.canvas {
            case .full:
                (model.diffusionCanvasLength, model.diffusionCanvasLength)
            case .adaptive(let minimumLength, let maximumLength):
                (
                    minimumLength ?? model.diffusionMinimumCanvasLength,
                    maximumLength ?? model.diffusionCanvasLength
                )
            }
        let resolvedMaxCanvas = Swift.min(
            model.diffusionCanvasLength,
            resolvedCanvasBounds.maximum)
        self.maxCanvasLength = Swift.max(1, resolvedMaxCanvas)
        self.minCanvasLength = Swift.min(
            self.maxCanvasLength,
            Swift.max(1, resolvedCanvasBounds.minimum))
        self.maxDenoisingSteps = model.diffusionMaxDenoisingSteps
        self.entropyBound = model.diffusionEntropyBound
        self.temperatureMin = model.diffusionTemperatureMin
        self.temperatureMax = model.diffusionTemperatureMax
        self.denoiserTemperature = parameters.diffusion.temperature
        self.diffusionSampler = parameters.diffusion.sampler
        self.prefersLogitsSelfConditioning = model.diffusionPrefersLogitsSelfConditioning
        self.selfConditioningWeight =
            self.prefersLogitsSelfConditioning ? nil : model.diffusionSelfConditioningWeight()
        self.stabilityThreshold = model.diffusionStabilityThreshold
        self.confidenceThreshold = model.diffusionConfidenceThreshold
        self.randomState =
            parameters.seed.map(MLXRandom.RandomState.init(seed:))
            ?? MLXRandom.RandomState()

        self.promptPrefillTime = try measureBlockDiffusionPrefill {
            try model.prepareDiffusion(
                input, cache: cacheStorage.cache, windowSize: parameters.prefill.stepSize)
            cacheStorage.commitProcessedTokens(promptTokenCount)
        }

        try kvCachePlan.applyAndValidate(to: cacheStorage)
    }

    private static func validatePrompt(
        _ text: LMInput.Text, model: any BlockDiffusionLanguageModel
    ) throws -> Int {
        guard text.tokens.ndim == 1 || text.tokens.ndim == 2 else {
            throw GenerateError.invalidAttentionMask(
                "expected token shape [length] or [batch, length], got \(text.tokens.shape).")
        }

        let batchSize = text.tokens.ndim == 1 ? 1 : text.tokens.dim(0)
        let sequenceLength = text.tokens.dim(-1)
        let validIndices = try blockDiffusionPromptIndices(
            mask: text.mask,
            sequenceLength: sequenceLength,
            batchSize: batchSize,
            modelName: String(describing: type(of: model)))
        return validIndices?.dim(0) ?? sequenceLength
    }

    private func nextCanvasLength() -> Int {
        guard let maxTokens else { return maxCanvasLength }
        let remaining = Swift.max(0, maxTokens - tokenCount)
        guard remaining > 0 else { return 0 }
        return Swift.min(maxCanvasLength, Swift.max(remaining, minCanvasLength))
    }

    private mutating func makeInitialCanvas(length: Int) -> MLXArray {
        withRandomState(randomState) {
            MLXRandom.randInt(
                low: Int32(0),
                high: Int32(model.diffusionVocabularySize),
                [1, length],
                type: Int32.self
            )
        }
    }

    private func temperature(curStep: Int) -> Float {
        temperatureMin
            + ((temperatureMax - temperatureMin) * Float(curStep) / Float(maxDenoisingSteps))
    }

    private func temperatureScaledLogits(_ logits: MLXArray, curStep: Int) -> MLXArray {
        logits / MLXArray(temperature(curStep: curStep))
    }

    private mutating func sampleDenoiserCanvas(logits: MLXArray) -> MLXArray {
        if denoiserTemperature <= 0 {
            return argMax(logits, axis: -1).asType(.int32)
        }

        return withRandomState(randomState) {
            categorical(logits / denoiserTemperature).asType(.int32)
        }
    }

    private func entropy(logits: MLXArray) -> MLXArray {
        let logprobs = logSoftmax(logits)
        let probs = exp(logprobs)
        return -(probs * logprobs).sum(axis: -1)
    }

    private func acceptedTokenMask(entropy: MLXArray) -> MLXArray {
        let sortedIndices = argSort(entropy, axis: -1)
        let sortedEntropy = takeAlong(entropy, sortedIndices, axis: -1)
        let cumulativeEntropy = cumsum(sortedEntropy, axis: -1)
        let cumulativeMaximumEntropy = cummax(sortedEntropy, axis: -1)
        let sortedSelection = (cumulativeEntropy - cumulativeMaximumEntropy) .<= entropyBound
        return putAlong(
            MLXArray.zeros(entropy.shape, type: Bool.self),
            sortedIndices,
            values: sortedSelection,
            axis: -1)
    }

    private func tokenProbability(logits: MLXArray, tokens: MLXArray) -> MLXArray {
        let tokenLogits = takeAlong(logits, expandedDimensions(tokens, axis: -1), axis: -1)
            .squeezed(axis: -1)
        return exp(tokenLogits - logSumExp(logits, axis: -1))
    }

    private func confidenceTransferMask(
        confidence: MLXArray,
        unrevealedMask: MLXArray,
        threshold: Float
    ) -> MLXArray {
        let transferMask = unrevealedMask & (confidence .>= threshold)
        let hasUnrevealed = unrevealedMask.any(axis: -1, keepDims: true)
        let hasTransfer = transferMask.any(axis: -1, keepDims: true)
        let needsForce = hasUnrevealed & logicalNot(hasTransfer)
        let maskedConfidence = MLX.where(unrevealedMask, confidence, MLXArray(-Float.infinity))
        let bestIndex = argMax(maskedConfidence, axis: -1)
        let positions = MLXArray(Int32(0) ..< Int32(confidence.dim(-1)))[.newAxis, 0...]
        let forced = (positions .== bestIndex[0..., .newAxis]) & needsForce
        return transferMask | forced
    }

    private mutating func diffusionShouldStop(argmaxCanvas: MLXArray, entropy: MLXArray) -> Bool {
        let stable: Bool

        if stabilityThreshold == 0 {
            stable = true
        } else {
            if argmaxCanvasHistory == nil {
                argmaxCanvasHistory = Array(
                    repeating: MLXArray.full(
                        argmaxCanvas.shape, values: MLXArray(Int32(-1)), type: Int32.self),
                    count: stabilityThreshold)
            }

            stable =
                argmaxCanvasHistory?.allSatisfy {
                    ($0 .== argmaxCanvas).all().item(Bool.self)
                } ?? false
            argmaxCanvasHistory?.removeFirst()
            argmaxCanvasHistory?.append(argmaxCanvas)
        }

        let meanEntropy = entropy.mean().item(Float.self)
        return stable && meanEntropy < confidenceThreshold
    }

    private mutating func refillPendingTokens() {
        let length = nextCanvasLength()
        guard length > 0 else { return }

        var currentCanvas = makeInitialCanvas(length: length)
        var argmaxCanvas = currentCanvas
        var selfConditioningLogits: MLXArray?
        var selfConditioningEmbeddings: MLXArray?
        var draftRevealMask = MLXArray.zeros(currentCanvas.shape, type: Bool.self)
        var draftCanvas = currentCanvas
        argmaxCanvasHistory = nil

        denoisingLoop: for curStep in stride(from: maxDenoisingSteps, through: 1, by: -1) {
            let rawLogits =
                if prefersLogitsSelfConditioning {
                    model.diffusionLogits(
                        canvasTokens: currentCanvas,
                        cache: cache,
                        selfConditioningLogits: selfConditioningLogits)
                } else {
                    model.diffusionLogits(
                        canvasTokens: currentCanvas,
                        cache: cache,
                        selfConditioningEmbeddings: selfConditioningEmbeddings)
                }

            let processedLogits = temperatureScaledLogits(rawLogits, curStep: curStep)
            argmaxCanvas = argMax(processedLogits, axis: -1).asType(.int32)

            if curStep == 1 {
                break
            }

            let denoiserCanvas = sampleDenoiserCanvas(logits: processedLogits).asType(.int32)
            let tokenEntropy = entropy(logits: processedLogits)

            switch diffusionSampler {
            case .entropyBound:
                let acceptedMask = acceptedTokenMask(entropy: tokenEntropy)
                currentCanvas = MLX.where(
                    acceptedMask, denoiserCanvas, makeInitialCanvas(length: length))
                draftRevealMask = acceptedMask
                draftCanvas = argmaxCanvas

            case .confidenceThreshold(let threshold):
                let unrevealedMask = logicalNot(draftRevealMask)
                let confidence = tokenProbability(logits: processedLogits, tokens: denoiserCanvas)
                let acceptedMask = confidenceTransferMask(
                    confidence: confidence,
                    unrevealedMask: unrevealedMask,
                    threshold: threshold)
                let acceptedCanvas = MLX.where(acceptedMask, denoiserCanvas, draftCanvas)
                currentCanvas = MLX.where(
                    draftRevealMask | acceptedMask,
                    acceptedCanvas,
                    makeInitialCanvas(length: length))
                draftRevealMask = draftRevealMask | acceptedMask
                draftCanvas = acceptedCanvas

                if draftRevealMask.all().item(Bool.self) {
                    argmaxCanvas = draftCanvas
                    break denoisingLoop
                }
            }

            if prefersLogitsSelfConditioning {
                selfConditioningLogits = processedLogits
                asyncEval(currentCanvas, argmaxCanvas, processedLogits)
            } else {
                selfConditioningEmbeddings = model.diffusionSelfConditioningEmbeddings(
                    logits: processedLogits,
                    weight: selfConditioningWeight)
                asyncEval(currentCanvas, argmaxCanvas, selfConditioningEmbeddings!)
            }

            if diffusionShouldStop(argmaxCanvas: argmaxCanvas, entropy: tokenEntropy) {
                break
            }
        }

        eval(argmaxCanvas)
        pendingTokens = argmaxCanvas.flattened().asArray(Int.self)
        pendingIndex = 0
        committedPendingIndex = 0
    }

    private mutating func commitPendingTokens(upTo targetIndex: Int) {
        let targetIndex = Swift.min(targetIndex, pendingIndex)
        guard targetIndex > committedPendingIndex else { return }

        let tokens = pendingTokens[committedPendingIndex ..< targetIndex].map(Int32.init)
        model.acceptDiffusionTokens(
            MLXArray(tokens).reshaped([1, tokens.count]),
            cache: cache,
            windowSize: prefillStepSize)
        committedPendingIndex = targetIndex
        cacheStorage.commitProcessedTokens(tokens.count)
        kvCachePlan.apply(to: cacheStorage)
        asyncEval(cache)
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            commitPendingTokens(upTo: pendingIndex)
            return nil
        }

        if pendingIndex >= pendingTokens.count {
            commitPendingTokens(upTo: pendingIndex)
            pendingTokens.removeAll(keepingCapacity: true)
            pendingIndex = 0
            committedPendingIndex = 0
            refillPendingTokens()
        }

        guard pendingIndex < pendingTokens.count else {
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }

    /// The generation loop reports every pulled token it did not emit (a stop
    /// token withheld from the consumer) through this hook. Roll the pull back
    /// so neither the token counter nor the pending-token ledger — which
    /// `finalizeGeneration()` commits into the KV cache — includes it.
    public mutating func discardGeneratedToken() {
        guard pendingIndex > 0 else { return }
        pendingIndex -= 1
        tokenCount = Swift.max(0, tokenCount - 1)
    }
}

extension BlockDiffusionTokenIterator: GenerationFinalizingTokenIterator {
    /// Commit the canvas tokens the generation loop actually emitted into the
    /// KV cache; the un-emitted remainder of the last canvas never enters the
    /// cache, keeping the shared processed-token timeline exact.
    mutating func finalizeGeneration() {
        commitPendingTokens(upTo: pendingIndex)
    }
}

private func measureBlockDiffusionPrefill(_ closure: () throws -> Void) rethrows -> TimeInterval {
    let start = Date.timeIntervalSinceReferenceDate
    try closure()
    return Date.timeIntervalSinceReferenceDate - start
}
