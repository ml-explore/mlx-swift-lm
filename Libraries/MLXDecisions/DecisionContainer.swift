// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// A thread-safe decision model, returned by ``DecisionModelFactory``.
///
/// A decision reads the state, its media and each question, and returns option probabilities
/// without generating tokens. Requests from several tasks run one at a time.
public final class DecisionContainer: Sendable {
    /// Returns option probabilities for a validated request: one list per question, in request
    /// and option order. A JSON `null` state arrives as `nil`, a JSON string state as text.
    typealias PredictOperation =
        @Sendable (sending DecisionRequest, DecisionExecutionOptions) async throws ->
        DecisionProbabilities

    /// Identifier of the loaded model.
    public let modelID: String

    /// The questions and media the model accepts.
    public let capabilities: DecisionCapabilities

    private let predictOperation: PredictOperation

    init(
        modelID: String,
        capabilities: DecisionCapabilities,
        predictOperation: @escaping PredictOperation
    ) {
        self.modelID = modelID
        self.capabilities = capabilities
        self.predictOperation = predictOperation
    }

    /// Validate `request`, run the model and build typed answers.
    public func predict(
        _ request: consuming sending DecisionRequest,
        options: DecisionExecutionOptions = .init()
    ) async throws -> DecisionResponse {
        // Python reads `None` as no state and a `str` as text.
        switch request.state {
        case .json(.null): request.state = nil
        case .json(.string(let text)): request.state = .text(text)
        default: break
        }
        try validateDecisionRequest(request, options: options, capabilities: capabilities)
        let questions = request.questions
        try Task.checkCancellation()
        let output = try await predictOperation(request, options)
        let answers = try decisionAnswers(questions: questions, probabilities: output.probabilities)
        return DecisionResponse(modelID: modelID, answers: answers, inputTokens: output.inputTokens)
    }

    /// Answer named questions about a state and its media. Answers keep the order of
    /// `questions`.
    ///
    /// ```swift
    /// let response = try await decider.predict(
    ///     state: "I was charged twice, please refund one.",
    ///     questions: [
    ///         "refund": .bool("Is the customer asking for a refund?"),
    ///         "team": .choice("Which team should handle this?", options: ["billing", "technical"]),
    ///     ])
    /// ```
    public func predict(
        state: DecisionState? = nil,
        questions: KeyValuePairs<String, DecisionQuestion>,
        images: consuming sending [UserInput.Image] = [],
        audio: consuming sending UserInput.Audio? = nil,
        options: DecisionExecutionOptions = .init()
    ) async throws -> DecisionResponse {
        let request = DecisionRequest(
            state: state, questions: questions, images: images, audio: audio)
        return try await predict(request, options: options)
    }
}

/// Check a request against a model's capabilities, in the order `mlx_vlm.decision.predict`
/// and then the model's own question parser check it. Remote media are rejected last.
func validateDecisionRequest(
    _ request: borrowing DecisionRequest,
    options: DecisionExecutionOptions,
    capabilities: DecisionCapabilities
) throws {
    guard !capabilities.questionKinds.isEmpty else {
        throw DecisionError.unsupportedModel("This model does not support decision prediction")
    }
    guard options.tokenBudget > 0 else {
        throw DecisionError.invalidExecutionOption(
            name: "tokenBudget", value: options.tokenBudget)
    }
    if !request.images.isEmpty, !capabilities.media.contains(.images) {
        throw DecisionError.unsupportedMedia(.images)
    }
    if request.audio != nil, !capabilities.media.contains(.audio) {
        throw DecisionError.unsupportedMedia(.audio)
    }
    guard !request.questions.isEmpty else {
        throw DecisionError.noQuestions
    }
    var names = Set<String>()
    for named in request.questions where !names.insert(named.name).inserted {
        throw DecisionError.duplicateQuestionName(named.name)
    }
    if request.state == nil, request.images.isEmpty, request.audio == nil {
        throw DecisionError.missingState
    }
    for named in request.questions {
        let name = named.name
        let kind = named.question.kind
        guard capabilities.questionKinds.contains(kind) else {
            throw DecisionError.unsupportedQuestionKind(name: name, kind: kind)
        }
        switch named.question.criteria {
        case .choice(let options):
            guard options.count >= 2 else {
                throw DecisionError.tooFewCriteria(name: name, kind: kind, minimum: 2)
            }
            if options.contains(where: { $0.label.isEmpty }) {
                throw DecisionError.emptyLabel(name: name)
            }
            var labels = Set<String>()
            for option in options where !labels.insert(option.label).inserted {
                throw DecisionError.duplicateLabel(name: name, label: option.label)
            }
        case .score(let levels):
            guard levels.count >= 2 else {
                throw DecisionError.tooFewCriteria(name: name, kind: kind, minimum: 2)
            }
        case .bool:
            break
        }
    }
    // The model's own parser runs after the shared checks pass for every question.
    if let maximum = capabilities.maximumScoreLevels {
        for named in request.questions {
            if case .score(let levels) = named.question.criteria, levels.count > maximum {
                throw DecisionError.tooManyScoreLevels(
                    name: named.name, count: levels.count, maximum: maximum)
            }
        }
    }
    for image in request.images {
        if case .url(let url) = image.source, !url.isFileURL, url.scheme?.lowercased() != "data" {
            throw DecisionError.invalidImage(
                "\(url.absoluteString): only file and data: URLs are read; download remote files first"
            )
        }
    }
    if case .url(let url)? = request.audio?.source, !url.isFileURL {
        throw DecisionError.invalidAudio(
            "\(url.absoluteString): only file URLs are read; download remote files first")
    }
}

/// Build typed answers from per-question option probabilities.
func decisionAnswers(
    questions: [DecisionRequest.NamedQuestion], probabilities: [[Double]]
) throws -> [DecisionAnswer] {
    guard probabilities.count == questions.count else {
        throw DecisionError.invalidProbabilities(
            "Expected \(questions.count) probability lists, got \(probabilities.count)")
    }
    return try zip(questions, probabilities).map { named, p in
        let name = named.name
        let expected: Int
        switch named.question.criteria {
        case .choice(let options): expected = options.count
        case .score(let levels): expected = levels.count
        case .bool: expected = 2
        }
        guard p.count == expected else {
            throw DecisionError.invalidProbabilities(
                "Expected \(expected) probabilities for question \(name), got \(p.count)")
        }
        guard p.allSatisfy({ $0.isFinite && $0 >= 0 }), abs(p.reduce(0, +) - 1) <= 1e-3 else {
            throw DecisionError.invalidProbabilities(
                "Probabilities for question \(name) do not form a distribution: \(p)")
        }
        let value: DecisionAnswer.Value
        switch named.question.criteria {
        case .choice(let options):
            var best = 0
            for i in p.indices where p[i] > p[best] {
                best = i
            }
            value = .choice(options[best].label)
        case .score:
            value = .score(p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element })
        case .bool:
            value = .bool(p[1] >= 0.5)
        }
        return DecisionAnswer(name: name, value: value, probabilities: p)
    }
}

// MARK: - Model plumbing

/// Models that answer decision questions.
protocol DecisionModel: BaseLanguageModel {
    /// The questions and media the model accepts.
    var decisionCapabilities: DecisionCapabilities { get }

    /// Option probabilities for a validated request, in request and option order.
    /// Bool questions give `[P(false), P(true)]`.
    func decisionProbabilities(
        _ request: consuming DecisionRequest,
        tokenizer: any Tokenizer,
        options: DecisionExecutionOptions
    ) async throws -> DecisionProbabilities
}

/// Serial access to a decision model and its tokenizer.
final class DecisionModelContainer: Sendable {
    private let context: SerialAccessContainer<(model: any DecisionModel, tokenizer: any Tokenizer)>

    /// The questions and media the model accepts.
    let capabilities: DecisionCapabilities

    /// Wrap `model`, which is put in evaluation mode, and its tokenizer.
    init(model: any DecisionModel, tokenizer: any Tokenizer) {
        model.train(false)
        self.capabilities = model.decisionCapabilities
        self.context = SerialAccessContainer((model, tokenizer))
    }

    /// Option probabilities for a validated request.
    func probabilities(
        _ request: consuming sending DecisionRequest, options: DecisionExecutionOptions
    ) async throws -> DecisionProbabilities {
        let request = SendableBox(request)
        return try await context.read { context in
            try Task.checkCancellation()
            return try await context.model.decisionProbabilities(
                request.consume(), tokenizer: context.tokenizer, options: options)
        }
    }

    /// Wrap this container in a ``DecisionContainer``.
    func decisionContainer(modelID: String) -> DecisionContainer {
        DecisionContainer(modelID: modelID, capabilities: capabilities) { request, options in
            try await self.probabilities(request, options: options)
        }
    }
}
