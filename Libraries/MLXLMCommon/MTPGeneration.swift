// Copyright © 2025 Apple Inc.
//
// Multi-Token Prediction (MTP) speculative decoding support.
//
// MTP differs from classic two-model speculative decoding (SpeculativeTokenIterator)
// in that the drafter is a small model embedded within the backbone checkpoint that
// shares KV cache and embeddings with the backbone. This eliminates the overhead of
// maintaining a separate draft model's cache.
//
// The generation cycle:
// 1. Backbone forward → logits + last hidden state
// 2. MTP drafter uses hidden state to propose K tokens (tiny, fast)
// 3. Backbone verifies all K tokens in one forward pass
// 4. Leviathan-Chen rejection sampling accepts/rejects each draft
// 5. Accepted tokens + bonus token emitted; cache trimmed on rejection

import Foundation
import MLX
import MLXNN

// MARK: - MTP Model Protocols

/// Output from a backbone model that supports MTP speculative decoding.
///
/// Includes both the standard logits and the pre-logit hidden states needed
/// by the MTP drafter model.
public struct MTPBackboneOutput {
    public let logits: MLXArray
    public let hiddenStates: MLXArray
    public let state: LMOutput.State?

    public init(logits: MLXArray, hiddenStates: MLXArray, state: LMOutput.State? = nil) {
        self.logits = logits
        self.hiddenStates = hiddenStates
        self.state = state
    }
}

/// Protocol for backbone language models that support MTP speculative decoding.
///
/// Backbone models conforming to this protocol expose their pre-logit hidden states,
/// which the MTP drafter consumes to generate draft tokens. The token embeddings are
/// also exposed so the drafter can concatenate them with hidden states.
public protocol MTPBackboneModel: LanguageModel {
    /// Forward pass returning both logits and last-layer hidden states.
    func forwardMTP(_ inputs: MLXArray, cache: [KVCache]?) -> MTPBackboneOutput

    /// The token embedding layer, shared with the drafter.
    var sharedEmbeddings: Embedding { get }

    /// Embedding scale factor (typically sqrt(hidden_size)).
    var embeddingScale: Float { get }

    /// Layer types for the backbone (e.g. "sliding_attention", "full_attention").
    /// Used to extract the correct shared KV states for the drafter.
    var backboneLayerTypes: [String] { get }
}

/// Output from an MTP draft model forward pass.
public struct MTPDraftOutput {
    /// Up-projected hidden state back in backbone dimension space.
    public let projectedState: MLXArray
    /// Logits over the vocabulary.
    public let logits: MLXArray

    public init(projectedState: MLXArray, logits: MLXArray) {
        self.projectedState = projectedState
        self.logits = logits
    }
}

/// Protocol for MTP draft models (e.g. Gemma4AssistantModel).
///
/// Unlike a standard ``LanguageModel``, an MTP draft model:
/// - Takes pre-projected embeddings as input (not raw tokens)
/// - Uses shared KV from the backbone (no separate cache)
/// - Returns a projected state for chained drafting
public protocol MTPDraftModel: Module {
    /// Forward pass for one draft step.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: Concatenated `[backbone_hidden, token_embedding]`
    ///     of shape `[B, 1, 2 * backbone_hidden_size]`
    ///   - sharedKVStates: KV states from backbone cache, keyed by layer type
    /// - Returns: Projected state and logits
    func forward(
        inputsEmbeds: MLXArray,
        sharedKVStates: [String: (MLXArray, MLXArray)]?
    ) -> MTPDraftOutput
}

// MARK: - MTP Verification

/// Result of verifying a batch of draft tokens against backbone logits.
public struct MTPVerificationResult {
    public let acceptedTokens: [Int]
    public let nextToken: Int
    public let acceptanceProbabilities: [Float]

    public var acceptanceRate: Float {
        guard !acceptanceProbabilities.isEmpty else { return 0 }
        return Float(acceptedTokens.count) / Float(acceptanceProbabilities.count)
    }

    public var allTokens: [Int] { acceptedTokens + [nextToken] }
}

/// Verifies draft tokens against target model logits using Leviathan-Chen
/// rejection sampling with residual correction.
///
/// Critical: probability ratios are computed in fp32 to avoid BF16 underflow.
///
/// Reference: "Fast Inference from Transformers via Speculative Decoding"
/// (Leviathan et al., 2023)
public struct MTPVerifier {
    private let epsilon: Float = 1e-10

    public init() {}

    /// Verify draft tokens against target logits.
    ///
    /// - Parameters:
    ///   - draftTokens: Token IDs proposed by the drafter
    ///   - draftLogits: Logits from drafter for each position
    ///   - targetLogits: Logits from backbone verification pass `[K+1, vocab]`
    ///   - temperature: Sampling temperature (0 = greedy argmax matching)
    public func verify(
        draftTokens: [Int],
        draftLogits: [MLXArray],
        targetLogits: MLXArray,
        temperature: Float
    ) -> MTPVerificationResult {
        let n = draftTokens.count

        if temperature == 0 {
            return verifyGreedy(draftTokens: draftTokens, targetLogits: targetLogits)
        }

        var acceptedTokens: [Int] = []
        var nextToken: Int?
        var probabilities: [Float] = []

        for i in 0 ..< n {
            var draftProbs = softmax(draftLogits[i].asType(.float32) / temperature)
            var targetProbs = softmax(targetLogits[i].asType(.float32) / temperature)
            draftProbs = draftProbs.squeezed()
            targetProbs = targetProbs.squeezed()

            let draftToken = draftTokens[i]
            let pTarget = targetProbs[draftToken].item(Float.self)
            let pDraft = max(draftProbs[draftToken].item(Float.self), epsilon)

            let acceptProb = min(1.0, pTarget / pDraft)
            probabilities.append(acceptProb)

            let u = Float.random(in: 0 ..< 1)
            if u < acceptProb {
                acceptedTokens.append(draftToken)
            } else {
                let residual = maximum(targetProbs - draftProbs, MLXArray(0.0))
                let residualSum = residual.sum().item(Float.self)
                if residualSum > epsilon {
                    let normalized = residual / residualSum
                    nextToken = categorical(log(normalized + epsilon)).item(Int.self)
                } else {
                    nextToken = categorical(log(targetProbs + epsilon)).item(Int.self)
                }
                break
            }
        }

        if nextToken == nil {
            let bonusLogits = targetLogits[n].asType(.float32).squeezed()
            nextToken = categorical(bonusLogits / temperature).item(Int.self)
        }

        return MTPVerificationResult(
            acceptedTokens: acceptedTokens,
            nextToken: nextToken!,
            acceptanceProbabilities: probabilities
        )
    }

    private func verifyGreedy(
        draftTokens: [Int],
        targetLogits: MLXArray
    ) -> MTPVerificationResult {
        var acceptedTokens: [Int] = []
        var nextToken: Int?

        for i in 0 ..< draftTokens.count {
            let targetToken = argMax(targetLogits[i].squeezed(), axis: -1).item(Int.self)
            if targetToken == draftTokens[i] {
                acceptedTokens.append(draftTokens[i])
            } else {
                nextToken = targetToken
                break
            }
        }

        if nextToken == nil {
            nextToken = argMax(
                targetLogits[draftTokens.count].squeezed(), axis: -1
            ).item(Int.self)
        }

        return MTPVerificationResult(
            acceptedTokens: acceptedTokens,
            nextToken: nextToken!,
            acceptanceProbabilities: acceptedTokens.map { _ in Float(1.0) }
        )
    }
}

// MARK: - MTP Speculative Token Iterator

/// Generator of tokens using MTP (Multi-Token Prediction) speculative decoding.
///
/// Unlike ``SpeculativeTokenIterator`` which uses a separate draft model with its own
/// KV cache, this iterator uses the backbone's built-in MTP assistant heads that share
/// the backbone's KV cache and embeddings.
///
/// To use it directly:
///
/// ```swift
/// let iterator = try MTPSpeculativeTokenIterator(
///     input: input, backbone: backboneModel, drafter: assistantModel,
///     parameters: generateParameters, numDraftTokens: 3)
///
/// for token in iterator {
///     // process token
/// }
/// ```
///
/// Port of MTP speculative generation from Gemma 4 assistant models.
public struct MTPSpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text

    let backbone: any MTPBackboneModel
    let drafter: any MTPDraftModel

    var backboneState: LMOutput.State?
    var cache: [KVCache]

    var processor: LogitProcessor?
    let sampler: LogitSampler
    let verifier: MTPVerifier

    var tokenCount = 0
    let maxTokens: Int?
    let numDraftTokens: Int

    private var pendingTokens = [Int]()
    private var pendingIndex = 0

    private var lastHiddenState: MLXArray?

    var promptPrefillTime: TimeInterval = 0.0

    /// Initialize an `MTPSpeculativeTokenIterator`.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - backbone: the backbone ``MTPBackboneModel`` (verifier)
    ///   - drafter: the MTP ``MTPDraftModel`` (assistant)
    ///   - cache: optional ``KVCache`` for the backbone
    ///   - parameters: the generation parameters
    ///   - numDraftTokens: number of tokens the drafter proposes per round
    public init(
        input: LMInput,
        backbone: any MTPBackboneModel,
        drafter: any MTPDraftModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int = 3
    ) throws {
        self.y = input.text
        self.backbone = backbone
        self.drafter = drafter

        self.cache = cache ?? backbone.newCache(parameters: parameters)
        guard canTrimPromptCache(self.cache) else {
            throw KVCacheError(message: "MTP speculative decoding requires trimmable KV caches.")
        }

        self.sampler = parameters.sampler()
        self.processor = parameters.processor()
        self.maxTokens = parameters.maxTokens
        self.numDraftTokens = numDraftTokens
        self.verifier = MTPVerifier()

        self.promptPrefillTime = try mtpMeasure {
            try prepare(input: input, windowSize: parameters.prefillStepSize)
        }
    }

    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        switch try backbone.prepare(input, cache: cache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens

            // Forward the remaining tokens through MTP-capable path to get hidden states
            let mtpOutput = backbone.forwardMTP(y[text: .newAxis].tokens, cache: cache)
            // Keep only the last position — the drafter needs [1, 1, hidden_size]
            let lastPos = mtpOutput.hiddenStates.dim(1) - 1
            lastHiddenState = mtpOutput.hiddenStates[0..., lastPos..<(lastPos + 1), 0...]

            var logits = mtpOutput.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            backboneState = mtpOutput.state
            asyncEval(y.tokens)

        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            backboneState = result.state
            asyncEval(y.tokens)
        }
    }

    /// Extract shared KV states from the backbone cache for the drafter.
    func extractSharedKV() -> [String: (MLXArray, MLXArray)] {
        let layerTypes = backbone.backboneLayerTypes
        var lastByType = [String: Int]()
        for (i, lt) in layerTypes.enumerated() {
            if i < cache.count {
                lastByType[lt] = i
            }
        }

        var result = [String: (MLXArray, MLXArray)]()
        for (lt, cacheIdx) in lastByType {
            let state = cache[cacheIdx].innerState()
            if state.count >= 2 {
                result[lt] = (state[0], state[1])
            }
        }
        return result
    }

    /// Run one round of MTP speculative decoding.
    mutating func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? numDraftTokens
        let numDraft = Swift.min(remaining, numDraftTokens)
        guard numDraft > 0 else { return }

        // Draft phase: use MTP assistant to propose tokens
        let sharedKV = extractSharedKV()
        let currentToken = y.tokens.item(Int.self)

        var draftTokens = [Int]()
        var draftLogits = [MLXArray]()
        var currentHidden = lastHiddenState ?? MLXArray.zeros([1, 1, backbone.sharedEmbeddings.weight.dim(1)])
        var draftToken = currentToken

        for _ in 0 ..< numDraft {
            let tokenEmbedding = backbone.sharedEmbeddings(
                MLXArray([Int32(draftToken)])
            ).expandedDimensions(axis: 0) * backbone.embeddingScale

            let inputEmbeds = concatenated([currentHidden, tokenEmbedding], axis: -1)
            let output = drafter.forward(inputsEmbeds: inputEmbeds, sharedKVStates: sharedKV)
            let logits = output.logits.squeezed(axis: 0).squeezed(axis: 0)

            let token: Int
            if sampler is ArgMaxSampler {
                token = argMax(logits, axis: -1).item(Int.self)
            } else {
                let logits32 = logits.asType(.float32)
                token = categorical(softmax(logits32)).item(Int.self)
            }

            draftTokens.append(token)
            draftLogits.append(logits)
            currentHidden = output.projectedState
            draftToken = token
        }

        // Verify phase: backbone processes all draft tokens in one pass
        let verifyTokens = [y.tokens] + draftTokens.map { MLXArray([$0]) }
        let verifyInput = concatenated(verifyTokens)
        let verifyMTP = backbone.forwardMTP(
            verifyInput.expandedDimensions(axis: 0), cache: cache)

        let verifyStart = verifyInput.dim(0) - (numDraft + 1)
        let verifyLogits = verifyMTP.logits[0..., verifyStart..., 0...]
            .squeezed(axis: 0)

        // Accept/reject with Leviathan-Chen
        let temperature: Float = (sampler is ArgMaxSampler) ? 0 : 0.6
        let result = verifier.verify(
            draftTokens: draftTokens,
            draftLogits: draftLogits,
            targetLogits: verifyLogits,
            temperature: temperature
        )

        // Emit accepted tokens
        for token in result.acceptedTokens {
            processor?.didSample(token: MLXArray([token]))
            pendingTokens.append(token)
        }

        let finalToken = MLXArray([result.nextToken])
        processor?.didSample(token: finalToken)
        pendingTokens.append(result.nextToken)

        // Trim cache: rewind positions beyond accepted prefix
        let accepted = result.acceptedTokens.count
        let trimAmount = numDraft - accepted
        if trimAmount > 0 {
            trimPromptCache(cache, numTokens: trimAmount)
        }

        // Update state for next round
        let hiddenIdx = verifyStart + accepted
        lastHiddenState = verifyMTP.hiddenStates[0..., hiddenIdx ... hiddenIdx, 0...]
        backboneState = verifyMTP.state
        y = .init(tokens: finalToken)
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        if pendingTokens.isEmpty {
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}

// MARK: - Generate Functions

/// Generates text asynchronously using MTP speculative decoding.
///
/// This function uses the backbone model's built-in MTP assistant heads as a drafter,
/// potentially achieving 2-3x speedup over standard autoregressive generation without
/// any quality degradation.
///
/// Both models share the same tokenizer and KV cache. The drafter's parameters are
/// typically < 1% of the backbone's, adding negligible memory overhead.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache`` for the backbone.
///   - parameters: The configuration options for token generation.
///   - context: The model context for the backbone model.
///   - mtpDrafter: The MTP draft model (e.g. `Gemma4AssistantModel`).
///   - numDraftTokens: Number of tokens the drafter proposes per round (default: 3).
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `Generation` values.
/// - Throws: An error if the backbone does not conform to ``MTPBackboneModel``.
public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    mtpDrafter: any MTPDraftModel,
    numDraftTokens: Int = 3,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<Generation> {
    guard let backbone = context.model as? any MTPBackboneModel else {
        throw MTPGenerationError.backboneNotMTPCapable
    }

    let iterator = try MTPSpeculativeTokenIterator(
        input: input,
        backbone: backbone,
        drafter: mtpDrafter,
        cache: cache,
        parameters: parameters,
        numDraftTokens: numDraftTokens
    )
    let (stream, _) = generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: TextToolTokenLoopHandler(
            tokenizer: context.tokenizer,
            format: context.configuration.toolCallFormat ?? .json
        )
    )
    return stream
}

/// Generates raw token IDs asynchronously using MTP speculative decoding.
///
/// Same as the `Generation` variant but yields raw token IDs instead of decoded text.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache`` for the backbone.
///   - parameters: The configuration options for token generation.
///   - context: The model context for the backbone model.
///   - mtpDrafter: The MTP draft model.
///   - numDraftTokens: Number of tokens the drafter proposes per round (default: 3).
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    mtpDrafter: any MTPDraftModel,
    numDraftTokens: Int = 3,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<TokenGeneration> {
    guard let backbone = context.model as? any MTPBackboneModel else {
        throw MTPGenerationError.backboneNotMTPCapable
    }

    let iterator = try MTPSpeculativeTokenIterator(
        input: input,
        backbone: backbone,
        drafter: mtpDrafter,
        cache: cache,
        parameters: parameters,
        numDraftTokens: numDraftTokens
    )
    let (stream, _) = generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: RawTokenLoopHandler()
    )
    return stream
}

/// Errors specific to MTP generation.
public enum MTPGenerationError: LocalizedError {
    case backboneNotMTPCapable

    public var errorDescription: String? {
        switch self {
        case .backboneNotMTPCapable:
            "Backbone model does not conform to MTPBackboneModel. Only Gemma 4 models with MTP assistant heads are supported."
        }
    }
}

private func mtpMeasure(_ closure: () throws -> Void) rethrows -> TimeInterval {
    let start = Date.timeIntervalSinceReferenceDate
    try closure()
    return Date.timeIntervalSinceReferenceDate - start
}
