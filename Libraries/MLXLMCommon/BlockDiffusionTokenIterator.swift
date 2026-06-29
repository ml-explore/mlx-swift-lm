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
    var cache: [KVCache]
    let prefillStepSize: Int
    let minCanvasLength: Int
    let maxCanvasLength: Int
    let maxDenoisingSteps: Int
    let entropyBound: Float
    let temperatureMin: Float
    let temperatureMax: Float
    let denoiserTemperature: Float
    let diffusionSampler: DiffusionSampler
    let diffusionThreshold: Float
    let selfConditioningWeight: MLXArray?
    let stabilityThreshold: Int
    let confidenceThreshold: Float

    public var tokenCount = 0
    public let maxTokens: Int?
    public var promptPrefillTime: TimeInterval = 0.0

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    private var randomState = MLXRandom.RandomState()
    private var committedPendingIndex = 0
    private var argmaxCanvasHistory: [MLXArray]?

    public init(
        input: LMInput,
        model: any BlockDiffusionLanguageModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.cache = cache ?? model.newCache(parameters: parameters)
        self.prefillStepSize = parameters.prefillStepSize
        self.maxTokens = parameters.maxTokens ?? model.diffusionDefaultMaxTokens
        let resolvedMaxCanvas =
            if parameters.diffusionFullCanvas {
                model.diffusionCanvasLength
            } else {
                Swift.min(
                    model.diffusionCanvasLength,
                    parameters.diffusionMaxCanvasLength ?? model.diffusionCanvasLength)
            }
        self.maxCanvasLength = Swift.max(1, resolvedMaxCanvas)
        self.minCanvasLength = Swift.min(
            self.maxCanvasLength,
            Swift.max(1, parameters.diffusionMinCanvasLength ?? model.diffusionMinimumCanvasLength))
        self.maxDenoisingSteps = model.diffusionMaxDenoisingSteps
        self.entropyBound = model.diffusionEntropyBound
        self.temperatureMin = model.diffusionTemperatureMin
        self.temperatureMax = model.diffusionTemperatureMax
        self.denoiserTemperature = parameters.temperature
        self.diffusionSampler = parameters.diffusionSampler
        self.diffusionThreshold = parameters.diffusionThreshold
        self.selfConditioningWeight = model.diffusionSelfConditioningWeight()
        self.stabilityThreshold = model.diffusionStabilityThreshold
        self.confidenceThreshold = model.diffusionConfidenceThreshold

        self.promptPrefillTime = try measureBlockDiffusionPrefill {
            try model.prepareDiffusion(
                input, cache: self.cache, windowSize: parameters.prefillStepSize)
        }
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
        unrevealedMask: MLXArray
    ) -> MLXArray {
        let transferMask = unrevealedMask & (confidence .>= diffusionThreshold)
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
        var selfConditioningEmbeddings: MLXArray?
        var draftRevealMask = MLXArray.zeros(currentCanvas.shape, type: Bool.self)
        var draftCanvas = currentCanvas
        argmaxCanvasHistory = nil

        for curStep in stride(from: maxDenoisingSteps, through: 1, by: -1) {
            let rawLogits = model.diffusionLogits(
                canvasTokens: currentCanvas,
                cache: cache,
                selfConditioningEmbeddings: selfConditioningEmbeddings
            )

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

            case .confidenceThreshold:
                let unrevealedMask = logicalNot(draftRevealMask)
                let confidence = tokenProbability(logits: processedLogits, tokens: denoiserCanvas)
                let acceptedMask = confidenceTransferMask(
                    confidence: confidence,
                    unrevealedMask: unrevealedMask)
                let acceptedCanvas = MLX.where(acceptedMask, denoiserCanvas, draftCanvas)
                currentCanvas = MLX.where(
                    draftRevealMask | acceptedMask,
                    acceptedCanvas,
                    makeInitialCanvas(length: length))
                draftRevealMask = draftRevealMask | acceptedMask
                draftCanvas = acceptedCanvas

                if draftRevealMask.all().item(Bool.self) {
                    argmaxCanvas = draftCanvas
                    break
                }
            }

            selfConditioningEmbeddings = model.diffusionSelfConditioningEmbeddings(
                logits: processedLogits,
                weight: selfConditioningWeight)
            asyncEval(currentCanvas, argmaxCanvas, selfConditioningEmbeddings!)

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

    mutating public func finalize(tokenCount emittedTokenCount: Int) {
        let blockStartTokenCount = tokenCount - pendingIndex
        let emittedInCurrentBlock = Swift.max(
            0, Swift.min(pendingIndex, emittedTokenCount - blockStartTokenCount))
        commitPendingTokens(upTo: emittedInCurrentBlock)
    }
}

private func measureBlockDiffusionPrefill(_ closure: () throws -> Void) rethrows -> TimeInterval {
    let start = Date.timeIntervalSinceReferenceDate
    try closure()
    return Date.timeIntervalSinceReferenceDate - start
}
