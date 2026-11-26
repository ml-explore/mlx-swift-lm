// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXDecisions
@testable import MLXVLM

// MARK: - Fixtures

private let d1Capabilities = DecisionCapabilities(
    questionKinds: [.choice, .score, .bool], media: [.images], maximumScoreLevels: 10)

private let team = DecisionQuestion.choice("Which team?", options: ["billing", "technical"])
private let urgency = DecisionQuestion.score("How urgent?", levels: ["low", "mid", "high"])
private let refund = DecisionQuestion.bool("Refund?")

private func testImage() -> UserInput.Image {
    .url(URL(fileURLWithPath: "/nonexistent/cat.png"))
}

private func testAudio() -> UserInput.Audio {
    .url(URL(fileURLWithPath: "/nonexistent/clip.wav"))
}

private func validate(
    _ request: DecisionRequest,
    tokenBudget: Int = DecisionExecutionOptions().tokenBudget,
    capabilities: DecisionCapabilities = d1Capabilities
) throws {
    try validateDecisionRequest(
        request, options: DecisionExecutionOptions(tokenBudget: tokenBudget),
        capabilities: capabilities)
}

private func named(_ pairs: [(String, DecisionQuestion)]) -> [DecisionRequest.NamedQuestion] {
    pairs.map { DecisionRequest.NamedQuestion(name: $0.0, question: $0.1) }
}

// MARK: - Requests and validation

struct DecisionValidationTests {
    @Test func requestKeepsQuestionOrderAndLiterals() {
        let request = DecisionRequest(
            state: "plain text", questions: ["urgency": urgency, "team": team, "refund": refund])
        #expect(request.questions.map(\.name) == ["urgency", "team", "refund"])
        #expect(request.questions.map(\.question.kind) == [.score, .choice, .bool])
        #expect(request.state == .text("plain text"))

        let option: DecisionOption = "billing"
        #expect(option == DecisionOption("billing", description: nil))
        #expect(
            DecisionQuestion.bool("q", whenTrue: "yes").criteria
                == .bool(whenTrue: "yes", whenFalse: nil))
        #expect(DecisionExecutionOptions().tokenBudget == 16_384)
    }

    @Test func validRequestsPass() throws {
        try validate(DecisionRequest(state: "s", questions: ["team": team, "refund": refund]))
        // Media can carry the whole state.
        try validate(DecisionRequest(questions: ["refund": refund], images: [testImage()]))
        try validate(
            DecisionRequest(questions: ["refund": refund], audio: testAudio()),
            capabilities: DecisionCapabilities(questionKinds: [.bool], media: [.audio]))
        // Labels are not trimmed, so a space is a valid label.
        try validate(
            DecisionRequest(state: "s", questions: ["q": .choice("q", options: [" ", "  "])]))
        try validate(
            DecisionRequest(state: "s", questions: ["s": .score("q", levels: ["a", "a"])]))
        // Ten score levels is the limit, not past it.
        try validate(
            DecisionRequest(
                state: "s", questions: ["q": .score("q", levels: (0 ..< 10).map(String.init))]))
        try validate(DecisionRequest(state: .json([:]), questions: ["q": refund]))
        try validate(DecisionRequest(state: "", questions: ["q": refund]))
        try validate(DecisionRequest(state: "s", questions: ["q": refund]), tokenBudget: 1)
    }

    @Test func anEmptyImageListIsNoMedia() {
        #expect(throws: DecisionError.noQuestions) {
            try validate(
                DecisionRequest(state: "s", questions: [], images: []),
                capabilities: DecisionCapabilities(questionKinds: [.bool], media: []))
        }
    }

    @Test func questionsMustBePresentAndUniquelyNamed() {
        #expect(throws: DecisionError.noQuestions) {
            try validate(DecisionRequest(questions: []))
        }
        // Duplicate names are a structural error, found before any question is checked.
        #expect(throws: DecisionError.duplicateQuestionName("a")) {
            try validate(
                DecisionRequest(
                    questions: named([
                        ("a", .choice("q", options: ["x"])), ("b", refund), ("a", refund),
                    ])))
        }
    }

    @Test func questionKindMustBeSupported() {
        let choiceOnly = DecisionCapabilities(questionKinds: [.choice], media: [])
        #expect(throws: DecisionError.unsupportedQuestionKind(name: "u", kind: .score)) {
            try validate(
                DecisionRequest(
                    state: "s",
                    questions: ["t": team, "u": urgency, "r": .choice("q", options: [])]
                ),
                capabilities: choiceOnly)
        }
        #expect(throws: DecisionError.unsupportedQuestionKind(name: "r", kind: .bool)) {
            try validate(
                DecisionRequest(state: "s", questions: ["r": refund]), capabilities: choiceOnly)
        }
    }

    @Test func choiceAndScoreNeedTwoCriteria() {
        for options: [DecisionOption] in [[], [""], ["one"]] {
            #expect(throws: DecisionError.tooFewCriteria(name: "c", kind: .choice, minimum: 2)) {
                try validate(
                    DecisionRequest(state: "s", questions: ["c": .choice("q", options: options)]))
            }
        }
        for levels: [String] in [[], ["one"]] {
            #expect(throws: DecisionError.tooFewCriteria(name: "s", kind: .score, minimum: 2)) {
                try validate(
                    DecisionRequest(state: "s", questions: ["s": .score("q", levels: levels)]))
            }
        }
    }

    @Test func choiceLabelsMustBeNonemptyThenUnique() {
        #expect(throws: DecisionError.emptyLabel(name: "c")) {
            try validate(
                DecisionRequest(
                    state: "s", questions: ["c": .choice("q", options: ["a", "a", ""])]))
        }
        #expect(throws: DecisionError.duplicateLabel(name: "c", label: "a")) {
            try validate(
                DecisionRequest(
                    state: "s",
                    questions: [
                        "c": .choice(
                            "q",
                            options: [
                                "a", DecisionOption("b"), DecisionOption("a", description: "x"),
                            ])
                    ]))
        }
    }

    @Test func scoreLevelLimitComesAfterTheSharedChecks() throws {
        let eleven = DecisionQuestion.score("q", levels: (0 ..< 11).map(String.init))
        #expect(throws: DecisionError.tooManyScoreLevels(name: "s", count: 11, maximum: 10)) {
            try validate(DecisionRequest(state: "s", questions: ["t": team, "s": eleven]))
        }
        // mlx_vlm.decision.predict checks every question before the model parses any.
        #expect(throws: DecisionError.tooFewCriteria(name: "c", kind: .choice, minimum: 2)) {
            try validate(
                DecisionRequest(
                    state: "s", questions: ["s": eleven, "c": .choice("q", options: ["x"])]))
        }
        // A model without a limit accepts any number of levels.
        try validate(
            DecisionRequest(state: "s", questions: ["s": eleven]),
            capabilities: DecisionCapabilities(questionKinds: [.score], media: []))
    }

    @Test func onlyLocalMediaURLsAreRead() throws {
        let both = DecisionCapabilities(
            questionKinds: [.bool], media: [.images, .audio], maximumScoreLevels: 10)
        let questions = named([("r", refund)])
        let pixel = try #require(URL(string: "data:image/png;base64,iVBORw0KGgo="))
        try validate(DecisionRequest(questions: questions, images: [testImage(), .url(pixel)]))
        try validate(DecisionRequest(questions: questions, audio: testAudio()), capabilities: both)

        for text in ["https://example.com/cat.png", "http://example.com/cat", "ftp://h/cat.png"] {
            let image = UserInput.Image.url(try #require(URL(string: text)))
            #expect(
                throws: DecisionError.invalidImage(
                    "\(text): only file and data: URLs are read; download remote files first")
            ) {
                try validate(DecisionRequest(questions: questions, images: [testImage(), image]))
            }
        }
        for text in ["https://example.com/clip.wav", "data:audio/wav;base64,UklGRg=="] {
            let clip = UserInput.Audio.url(try #require(URL(string: text)))
            #expect(
                throws: DecisionError.invalidAudio(
                    "\(text): only file URLs are read; download remote files first")
            ) {
                try validate(DecisionRequest(questions: questions, audio: clip), capabilities: both)
            }
        }

        // The checks that follow mlx_vlm.decision.predict come first.
        let remote = UserInput.Image.url(try #require(URL(string: "https://example.com/cat.png")))
        #expect(throws: DecisionError.unsupportedMedia(.images)) {
            try validate(
                DecisionRequest(questions: questions, images: [remote]),
                capabilities: DecisionCapabilities(questionKinds: [.bool], media: []))
        }
        #expect(throws: DecisionError.noQuestions) {
            try validate(DecisionRequest(questions: [], images: [remote]))
        }
    }

    @Test func validationOrder() {
        let none = DecisionCapabilities(questionKinds: [.choice], media: [])
        let order: [(DecisionError, () throws -> Void)] = [
            (
                .unsupportedModel("This model does not support decision prediction"),
                {
                    try validate(
                        DecisionRequest(questions: []), tokenBudget: 0,
                        capabilities: DecisionCapabilities(questionKinds: [], media: []))
                }
            ),
            (
                .invalidExecutionOption(name: "tokenBudget", value: 0),
                {
                    try validate(
                        DecisionRequest(questions: [], images: [testImage()], audio: testAudio()),
                        tokenBudget: 0, capabilities: none)
                }
            ),
            (
                .invalidExecutionOption(name: "tokenBudget", value: -1),
                {
                    try validate(DecisionRequest(questions: []), tokenBudget: -1)
                }
            ),
            (
                .unsupportedMedia(.images),
                {
                    try validate(
                        DecisionRequest(questions: [], images: [testImage()], audio: testAudio()),
                        capabilities: none)
                }
            ),
            (
                .unsupportedMedia(.audio),
                { try validate(DecisionRequest(questions: [], audio: testAudio())) }
            ),
            (.noQuestions, { try validate(DecisionRequest(questions: []), capabilities: none) }),
            (
                .duplicateQuestionName("a"),
                {
                    try validate(
                        DecisionRequest(questions: named([("a", refund), ("a", refund)])),
                        capabilities: none)
                }
            ),
            (
                .missingState,
                {
                    try validate(DecisionRequest(questions: ["a": refund]), capabilities: none)
                }
            ),
            (
                .unsupportedQuestionKind(name: "a", kind: .bool),
                {
                    try validate(
                        DecisionRequest(state: "s", questions: ["a": refund]), capabilities: none)
                }
            ),
        ]
        for (expected, run) in order {
            #expect(throws: expected) { try run() }
        }
    }

    @Test func errorMessagesMatchPython() {
        #expect(
            DecisionError.unsupportedMedia(.images).errorDescription
                == "This model does not support image input")
        #expect(
            DecisionError.unsupportedMedia(.audio).errorDescription
                == "This model does not support audio input")
        #expect(
            DecisionError.noQuestions.errorDescription == "At least one named question is required")
        #expect(
            DecisionError.unsupportedQuestionKind(name: "q", kind: .score).errorDescription?
                .hasPrefix("This model does not support 'score' decisions") == true)
        #expect(
            DecisionError.tooFewCriteria(name: "q", kind: .choice, minimum: 2).errorDescription?
                .hasPrefix("choice needs at least 2 criteria") == true)
        #expect(
            DecisionError.emptyLabel(name: "q").errorDescription?
                .hasPrefix("Decision labels must be nonempty strings") == true)
        #expect(
            DecisionError.imagesWithAudio.errorDescription
                == "A request carries images or audio, not both")
        #expect(
            DecisionError.invalidExecutionOption(name: "tokenBudget", value: 0).errorDescription
                == "tokenBudget must be positive, got 0")
        #expect(
            DecisionError.imagePlaceholderInText("<image>").errorDescription
                == "The state or a question contains <image>, which the model reads as an image")
    }
}

// MARK: - Answers

struct DecisionAnswerTests {
    @Test func choiceTakesTheFirstMostProbableOption() throws {
        let question = DecisionQuestion.choice("q", options: ["a", "b", "c"])
        let answers = try decisionAnswers(
            questions: named([("q", question)]), probabilities: [[0.4, 0.4, 0.2]])
        let answer = try #require(answers.first)
        #expect(answer.name == "q")
        #expect(answer.value == .choice("a"))
        #expect(answer.confidence == 0.4)
        #expect(answer.probabilities == [0.4, 0.4, 0.2])

        let later = try decisionAnswers(
            questions: named([("q", question)]), probabilities: [[0.1, 0.2, 0.7]])
        #expect(later.first?.value == .choice("c"))
        #expect(later.first?.confidence == 0.7)
    }

    @Test func scoreIsTheExpectedLevel() throws {
        let p = [0.3991, 0.5125, 0.0884]
        let answers = try decisionAnswers(
            questions: named([("u", urgency)]), probabilities: [p])
        let answer = try #require(answers.first)
        // Python: sum(i * v for i, v in enumerate(p)), unrounded.
        var expected = 0.0
        for (i, v) in p.enumerated() { expected += Double(i) * v }
        #expect(answer.value == .score(expected))
        #expect(answer.confidence == 0.5125)
        #expect(answer.probabilities == p)
    }

    @Test func boolReadsFalseThenTrue() throws {
        let cases: [([Double], Bool)] = [
            ([0.2, 0.8], true), ([0.5, 0.5], true), ([0.6, 0.4], false),
            // Unrounded: Python would round 0.49996 up to 0.5 and answer true.
            ([0.50004, 0.49996], false),
        ]
        for (p, value) in cases {
            let answer = try #require(
                try decisionAnswers(questions: named([("r", refund)]), probabilities: [p]).first)
            #expect(answer.value == .bool(value))
            #expect(answer.probabilities == p)
            #expect(answer.confidence == p.max())
        }
    }

    @Test func answersKeepRequestOrder() throws {
        let answers = try decisionAnswers(
            questions: named([("r", refund), ("t", team), ("u", urgency)]),
            probabilities: [[0.1, 0.9], [0.3, 0.7], [0.2, 0.3, 0.5]])
        #expect(answers.map(\.name) == ["r", "t", "u"])
        #expect(answers.map(\.value) == [.bool(true), .choice("technical"), .score(1.3)])
    }

    @Test func malformedProbabilitiesAreRejected() throws {
        let questions = named([("t", team)])
        let bad: [[[Double]]] = [
            [],
            [[0.5, 0.5], [0.5, 0.5]],
            [[1.0]],
            [[0.3, 0.3, 0.4]],
            [[.nan, 1.0]],
            [[.infinity, 0]],
            [[-0.1, 1.1]],
            [[0.5, 0.49]],
            [[0.5, 0.502]],
        ]
        for probabilities in bad {
            #expect(throws: DecisionError.self) {
                try decisionAnswers(questions: questions, probabilities: probabilities)
            }
        }
        // Sums within 1e-3 of one pass.
        _ = try decisionAnswers(questions: questions, probabilities: [[0.5, 0.5009]])
        _ = try decisionAnswers(questions: questions, probabilities: [[0.0, 1.0]])
    }
}

// MARK: - Containers

private struct OperationCall: Sendable, Equatable {
    var names: [String]
    var state: DecisionState?
    var imageCount: Int
    var hasAudio: Bool
    var tokenBudget: Int
}

private actor OperationLog {
    private(set) var calls: [OperationCall] = []

    func record(_ call: OperationCall) {
        calls.append(call)
    }
}

private func recordingContainer(
    _ log: OperationLog, capabilities: DecisionCapabilities = d1Capabilities,
    probabilities: [[Double]] = [[0.25, 0.75]], inputTokens: Int = 42
) -> DecisionContainer {
    DecisionContainer(modelID: "test/decider", capabilities: capabilities) { request, options in
        await log.record(
            OperationCall(
                names: request.questions.map(\.name), state: request.state,
                imageCount: request.images.count, hasAudio: request.audio != nil,
                tokenBudget: options.tokenBudget))
        return DecisionProbabilities(probabilities: probabilities, inputTokens: inputTokens)
    }
}

private enum TestOperationError: Error, Equatable {
    case failed
}

struct DecisionContainerTests {
    @Test func predictPassesTheRequestAndBuildsTheResponse() async throws {
        let log = OperationLog()
        let container = recordingContainer(
            log, probabilities: [[0.25, 0.75], [0.9, 0.1]], inputTokens: 100)
        #expect(container.modelID == "test/decider")
        #expect(container.capabilities == d1Capabilities)

        let response = try await container.predict(
            state: "I was charged twice", questions: ["refund": refund, "team": team],
            images: [testImage()], options: DecisionExecutionOptions(tokenBudget: 64))

        #expect(
            await log.calls == [
                OperationCall(
                    names: ["refund", "team"], state: .text("I was charged twice"), imageCount: 1,
                    hasAudio: false, tokenBudget: 64)
            ])
        #expect(response.modelID == "test/decider")
        #expect(response.inputTokens == 100)
        #expect(response.answers.map(\.name) == ["refund", "team"])
        #expect(response["refund"]?.value == .bool(true))
        #expect(response["team"]?.value == .choice("billing"))
        #expect(response["missing"] == nil)
    }

    @Test func validationRunsBeforeTheOperation() async throws {
        let log = OperationLog()
        let container = recordingContainer(log)
        await #expect(throws: DecisionError.missingState) {
            try await container.predict(state: nil, questions: ["refund": refund])
        }
        await #expect(throws: DecisionError.unsupportedMedia(.audio)) {
            try await container.predict(
                state: "s", questions: ["refund": refund], audio: testAudio())
        }
        let url = try #require(URL(string: "https://example.com/cat.png"))
        await #expect(
            throws: DecisionError.invalidImage(
                "https://example.com/cat.png: only file and data: URLs are read; download remote files first"
            )
        ) {
            try await container.predict(
                state: "s", questions: ["refund": refund], images: [.url(url)])
        }
        await #expect(throws: DecisionError.invalidExecutionOption(name: "tokenBudget", value: 0)) {
            try await container.predict(
                state: "s", questions: ["refund": refund],
                options: DecisionExecutionOptions(tokenBudget: 0))
        }
        #expect(await log.calls.isEmpty)
    }

    @Test func jsonNullAndStringStatesReachTheOperationAsPythonReadsThem() async throws {
        let log = OperationLog()
        let container = recordingContainer(log)
        // Python reads a JSON null as None, which is rejected without media.
        await #expect(throws: DecisionError.missingState) {
            try await container.predict(state: .json(.null), questions: ["refund": refund])
        }
        #expect(await log.calls.isEmpty)

        _ = try await container.predict(
            state: .json(.null), questions: ["refund": refund], images: [testImage()])
        _ = try await container.predict(
            state: .json([:]), questions: ["refund": refund], images: [testImage()])
        // A JSON string is the text itself, as a Python `str` state is.
        _ = try await container.predict(
            state: .json(.string("say \"hi\"")), questions: ["refund": refund])
        #expect(await log.calls.map(\.state) == [nil, .json([:]), .text("say \"hi\"")])
    }

    @Test func requestsUseTheDefaultExecutionOptions() async throws {
        let log = OperationLog()
        let container = recordingContainer(log)
        _ = try await container.predict(DecisionRequest(state: "s", questions: ["r": refund]))
        _ = try await container.predict(questions: ["r": refund], images: [testImage()])
        #expect(await log.calls.map(\.tokenBudget) == [16_384, 16_384])
        #expect(await log.calls.map(\.state) == [.text("s"), nil])
    }

    @Test func cancellationStopsBeforeTheOperation() async {
        let log = OperationLog()
        let container = recordingContainer(log)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await container.predict(state: "s", questions: ["refund": refund])
        }
        await #expect(throws: CancellationError.self) {
            try await task.value
        }
        #expect(await log.calls.isEmpty)
    }

    @Test func operationErrorsAndBadOutputSurface() async {
        let failing = DecisionContainer(modelID: "m", capabilities: d1Capabilities) { _, _ in
            throw TestOperationError.failed
        }
        await #expect(throws: TestOperationError.failed) {
            try await failing.predict(state: "s", questions: ["refund": refund])
        }

        let log = OperationLog()
        let wrongCount = recordingContainer(log, probabilities: [[0.5, 0.5]])
        await #expect(throws: DecisionError.self) {
            try await wrongCount.predict(state: "s", questions: ["refund": refund, "team": team])
        }
    }
}

// MARK: - Decision models

private struct ModelCall: Sendable, Equatable {
    var training: [Bool]
    var weightSum: Float
    var tokenBudget: Int
}

private actor ModelLog {
    private(set) var calls: [ModelCall] = []

    func record(_ call: ModelCall) {
        calls.append(call)
    }
}

private final class TinyDecisionModel: Module, DecisionModel {
    static let capabilities = DecisionCapabilities(
        questionKinds: [.choice, .score, .bool], media: [], maximumScoreLevels: 10)

    @ModuleInfo var linear: Linear
    @ModuleInfo var dropout: Dropout

    let log: ModelLog
    let decisionCapabilities = TinyDecisionModel.capabilities

    init(log: ModelLog) {
        self.log = log
        self._linear.wrappedValue = Linear(2, 2)
        self._dropout.wrappedValue = Dropout(p: 0.5)
        super.init()
    }

    func decisionProbabilities(
        _ request: consuming DecisionRequest, tokenizer: any Tokenizer,
        options: DecisionExecutionOptions
    ) async throws -> DecisionProbabilities {
        let weightSum = (linear.weight.sum() + (linear.bias?.sum() ?? MLXArray(0))).item(Float.self)
        await log.record(
            ModelCall(
                training: [training, linear.training, dropout.training], weightSum: weightSum,
                tokenBudget: options.tokenBudget))
        let probabilities = request.questions.map { question -> [Double] in
            let count =
                switch question.question.criteria {
                case .choice(let options): options.count
                case .score(let levels): levels.count
                case .bool: 2
                }
            return Array(repeating: 1 / Double(count), count: count)
        }
        let text = request.state?.serialized(indent: nil) ?? ""
        return DecisionProbabilities(
            probabilities: probabilities,
            inputTokens: tokenizer.encode(text: text, addSpecialTokens: false).count)
    }
}

/// One token per code point.
private let scalarTokenizer = VocabTokenizer(vocabulary: [], characterBase: 0)

private actor RecordingTokenizerLoader: TokenizerLoader {
    private(set) var directories: [URL] = []

    func load(from directory: URL) async throws -> any Tokenizer {
        directories.append(directory)
        return scalarTokenizer
    }
}

private func temporaryModelDirectory(
    config: String, weights: [String: MLXArray] = [:]
) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appending(component: "decision-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try Data(config.utf8).write(to: directory.appending(component: "config.json"))
    if !weights.isEmpty {
        try save(arrays: weights, url: directory.appending(component: "model.safetensors"))
    }
    return directory
}

/// The tiny d1 checkpoint of mlx-vlm's `test_lfm2_vl_checkpoint_with_d1_auto_map_loads_as_d1`:
/// an `lfm2_vl` config with d1's `auto_map`, random weights in the Hugging Face layout. Also
/// returns the in-memory model whose weights were saved.
private func tinyD1Checkpoint() async throws -> (directory: URL, model: D1) {
    let config = tinyConfigJSON(#""auto_map": {"AutoModel": "modeling_d1.D1Model"},"#)
    let created = try await VLMTypeRegistry.shared.createModel(
        configuration: Data(config.utf8), modelType: "lfm2_vl")
    let model = D1(try #require(created as? LFM2VL))
    let languagePrefix = "language_model.model."
    var weights = [String: MLXArray]()
    for (key, value) in model.base.parameters().flattened() {
        guard key.hasPrefix(languagePrefix) else {
            weights[key] = value
            continue
        }
        let hubKey = "model.language_model." + key.dropFirst(languagePrefix.count)
        weights[hubKey] = key.hasSuffix(".conv.conv.weight") ? value.transposed(0, 2, 1) : value
    }
    let directory = try temporaryModelDirectory(config: config, weights: weights)
    try Data(#"{"processor_class": "Lfm2VlProcessor"}"#.utf8)
        .write(to: directory.appending(component: "processor_config.json"))
    return (directory, model)
}

struct DecisionModelContainerTests {
    @Test func containerPutsTheModelInEvaluationMode() async throws {
        let log = ModelLog()
        let model = TinyDecisionModel(log: log)
        #expect(model.training && model.linear.training && model.dropout.training)

        let container = DecisionModelContainer(model: model, tokenizer: scalarTokenizer)
        #expect(!model.training && !model.linear.training && !model.dropout.training)
        #expect(container.capabilities == TinyDecisionModel.capabilities)

        let predictor = container.decisionContainer(modelID: "test/tiny")
        #expect(predictor.modelID == "test/tiny")
        #expect(predictor.capabilities == TinyDecisionModel.capabilities)

        let response = try await predictor.predict(
            state: "abcd", questions: ["team": team, "urgency": urgency, "refund": refund],
            options: DecisionExecutionOptions(tokenBudget: 128))
        #expect(response.inputTokens == 4)
        #expect(response["team"]?.value == .choice("billing"))
        guard case .score(let level) = response["urgency"]?.value else {
            Issue.record("Expected a score answer")
            return
        }
        #expect(abs(level - 1) < 1e-12)
        #expect(response["refund"]?.value == .bool(true))

        let calls = await log.calls
        #expect(calls.count == 1)
        #expect(calls.first?.training == [false, false, false])
        #expect(calls.first?.tokenBudget == 128)
    }

    @Test func containerChecksCancellationBeforeTheModel() async {
        let log = ModelLog()
        let container = DecisionModelContainer(
            model: TinyDecisionModel(log: log), tokenizer: scalarTokenizer)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await container.probabilities(
                DecisionRequest(state: "s", questions: ["refund": refund]),
                options: DecisionExecutionOptions())
        }
        await #expect(throws: CancellationError.self) {
            try await task.value
        }
        #expect(await log.calls.isEmpty)
    }

    /// The default budget splits this request in two batches, where 65,536 reads it in one.
    /// In float32 the answers agree to rounding.
    @Test func theDefaultBudgetOnlyChangesTheBatches() async throws {
        let (directory, d1) = try await tinyD1Checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokenizer = scalarTokenizer
        let state = DecisionState.text(String(repeating: "The card was charged twice. ", count: 22))
        let questions = (0 ..< 30).map { (index: Int) -> DecisionQuestion in
            .choice("Route \(index)", options: ["A", "B"])
        }
        let budget = DecisionExecutionOptions().tokenBudget
        try Device.withDefaultDevice(.cpu) {
            let inputs = try d1.decisionInputs(
                state: state, questions: questions, tokenizer: tokenizer)
            let batches = { (budget: Int) -> Int in
                D1Prompt.chunks(
                    lengths: inputs.branches.map(\.count), trunk: inputs.trunk.count,
                    budget: budget
                ).count
            }
            #expect(batches(65_536) == 1)
            #expect(batches(budget) == 2)

            let split = try d1.probabilities(
                state: state, questions: questions, tokenizer: tokenizer, tokenBudget: budget)
            let whole = try d1.probabilities(
                state: state, questions: questions, tokenizer: tokenizer, tokenBudget: 65_536)
            #expect(split.inputTokens == whole.inputTokens)
            let differences = zip(split.probabilities.joined(), whole.probabilities.joined())
                .map { (pair: (Double, Double)) -> Double in abs(pair.0 - pair.1) }
            #expect(differences.count == 60)
            #expect((differences.max() ?? 1) < 1e-5)
        }
    }
}

// MARK: - Factory

struct DecisionModelFactoryTests {
    private func descriptor(_ json: String) throws -> DecisionDescriptor {
        try JSONDecoder().decode(DecisionDescriptor.self, from: Data(json.utf8))
    }

    @Test func descriptorRoutesD1Checkpoints() throws {
        let cases: [(String, DecisionArchitecture?)] = [
            (#"{"model_type": "lfm2_vl", "auto_map": {"AutoModel": "modeling_d1.D1Model"}}"#, .d1),
            (
                #"{"model_type": "lfm2-vl", "auto_map": {"AutoModel": "LiquidAI/d1-3B--modeling_d1.D1Model"}}"#,
                .d1
            ),
            (#"{"model_type": "lfm2_vl"}"#, nil),
            (#"{"model_type": "lfm2_vl", "auto_map": null}"#, nil),
            (#"{"model_type": "lfm2_vl", "auto_map": {"AutoConfig": "x.D1Config"}}"#, nil),
            (#"{"model_type": "lfm2_vl", "auto_map": {"AutoModel": "modeling.Lfm2VlModel"}}"#, nil),
            (#"{"model_type": "lfm2_vl", "auto_map": {"AutoModel": ["m.D1Model", null]}}"#, nil),
            (#"{"model_type": "lfm2", "auto_map": {"AutoModel": "modeling_d1.D1Model"}}"#, nil),
            (
                #"{"model_type": "d1_omni", "auto_map": {"AutoConfig": "configuration_d1.D1OmniConfig", "AutoModel": "modeling_d1.D1OmniModel"}}"#,
                .d1Omni
            ),
            (
                #"{"model_type": "lfm2_vl", "auto_map": {"AutoModel": "modeling_d1.D1OmniModel"}}"#,
                nil
            ),
            (#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"#, nil),
        ]
        for (json, expected) in cases {
            #expect(try descriptor(json).architecture == expected, "\(json)")
        }
        #expect(throws: DecodingError.self) {
            try descriptor(#"{"auto_map": {"AutoModel": "modeling_d1.D1Model"}}"#)
        }
    }

    @Test func factoryRejectsCheckpointsItCannotLoad() async throws {
        let loader = RecordingTokenizerLoader()
        let cases: [(config: String, message: String)] = [
            (#"{"model_type": "lfm2_vl"}"#, "Unsupported decision model type 'lfm2_vl'."),
            (
                #"{"model_type": "lfm2_vl", "auto_map": {"AutoModel": "modeling.Lfm2VlModel"}}"#,
                "Unsupported decision model type 'lfm2_vl' with AutoModel 'modeling.Lfm2VlModel'."
            ),
            (#"{"model_type": "qwen3"}"#, "Unsupported decision model type 'qwen3'."),
        ]
        for (config, message) in cases {
            let directory = try temporaryModelDirectory(config: config)
            defer { try? FileManager.default.removeItem(at: directory) }
            do {
                _ = try await DecisionModelFactory.shared.loadContainer(
                    from: directory, using: loader)
                Issue.record("Expected an unsupported model error for \(config)")
            } catch DecisionError.unsupportedModel(let text) {
                #expect(text.contains(message), "\(text)")
            }
        }
        #expect(await loader.directories.isEmpty)
    }

    @Test func d1RouteLoadsTheCheckpointWithTheVLMFactory() async throws {
        let (directory, model) = try await tinyD1Checkpoint()
        defer { try? FileManager.default.removeItem(at: directory) }
        let loader = RecordingTokenizerLoader()
        let modelID = ResolvedModelConfiguration(directory: directory).name
        let loaded = try await DecisionModelFactory.shared.loadContainer(
            from: directory, using: loader)
        #expect(await loader.directories == [directory])
        #expect(loaded.modelID == modelID)
        #expect(loaded.capabilities == d1Capabilities)

        // The scalar tokenizer has no single-token "yes", so ask choice and score questions.
        let questions: KeyValuePairs<String, DecisionQuestion> = [
            "route": .choice("Route", options: ["A", "B"]), "urgency": urgency,
        ]
        let inMemory = DecisionModelContainer(model: model, tokenizer: scalarTokenizer)
            .decisionContainer(modelID: "test/d1")
        let expected = try await inMemory.predict(state: "A B", questions: questions)
        let actual = try await loaded.predict(state: "A B", questions: questions)
        #expect(actual.answers == expected.answers)
        #expect(actual.inputTokens == expected.inputTokens)
        #expect(actual.answers.map(\.name) == ["route", "urgency"])
    }

    @Test func factoryReportsAMissingConfiguration() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "decision-tests-missing-\(UUID().uuidString)")
        do {
            _ = try await DecisionModelFactory.shared.loadContainer(
                from: directory, using: RecordingTokenizerLoader())
            Issue.record("Expected a configuration file error")
        } catch ModelFactoryError.configurationFileError(let file, _, _) {
            #expect(file == "config.json")
        }
    }

    @Test func directLoaderBuildsTheModelAndLoadsWeightsAndTokenizer() async throws {
        let directory = try temporaryModelDirectory(
            config: #"{"model_type": "tiny_decision"}"#,
            weights: [
                "linear.weight": MLXArray([0.5, 0.5, 0.5, 0.5] as [Float], [2, 2]),
                "linear.bias": MLXArray([1, 2] as [Float]),
            ])
        defer { try? FileManager.default.removeItem(at: directory) }
        let log = ModelLog()
        let loader = RecordingTokenizerLoader()
        var configurations: [String] = []

        let container = try await DecisionModelFactory.shared.loadDecisionModel(
            resolved: ResolvedModelConfiguration(directory: directory), modelID: "test/tiny",
            using: loader
        ) { data in
            configurations.append(String(decoding: data, as: UTF8.self))
            return TinyDecisionModel(log: log)
        }

        #expect(configurations == [#"{"model_type": "tiny_decision"}"#])
        #expect(await loader.directories == [directory])
        #expect(container.modelID == "test/tiny")
        #expect(container.capabilities == TinyDecisionModel.capabilities)

        let response = try await container.predict(state: "abc", questions: ["refund": refund])
        #expect(response.inputTokens == 3)
        let calls = await log.calls
        #expect(calls.first?.training == [false, false, false])
        #expect(calls.first?.weightSum == 5)
    }

    @Test func directLoaderReportsConfigurationDecodingErrors() async throws {
        struct Strict: Decodable { var width: Int }
        let directory = try temporaryModelDirectory(config: #"{"model_type": "tiny_decision"}"#)
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            _ = try await DecisionModelFactory.shared.loadDecisionModel(
                resolved: ResolvedModelConfiguration(directory: directory), modelID: "test/tiny",
                using: RecordingTokenizerLoader()
            ) { data in
                _ = try JSONDecoder().decode(Strict.self, from: data)
                return TinyDecisionModel(log: ModelLog())
            }
            Issue.record("Expected a configuration decoding error")
        } catch ModelFactoryError.configurationDecodingError(let file, _, _) {
            #expect(file == "config.json")
        }
    }
}

// MARK: - Python-compatible JSON

private struct PythonJSONCase: Sendable, CustomTestStringConvertible {
    let value: JSONValue
    let indented: String
    let compact: String

    var testDescription: String { compact }
}

struct PythonJSONTests {
    @Test(arguments: pythonJSONCases)
    fileprivate func dumpsMatchesPython(_ testCase: PythonJSONCase) {
        // JSONValue objects have no order, so they convert with sorted keys, as these
        // `sort_keys=True` references expect.
        let value = DecisionJSON(testCase.value)
        #expect(PythonJSON.dumps(value, indent: 2) == testCase.indented)
        #expect(PythonJSON.dumps(value, indent: nil) == testCase.compact)
    }

    @Test func floatReprMatchesPython() {
        // Python repr(float) for each value; Swift alone writes 9.1e+15 for the second.
        let cases: [(Double, String)] = [
            (1e15, "1000000000000000.0"), (9.1e15, "9100000000000000.0"),
            (9_007_199_254_740_992, "9007199254740992.0"), (1e16, "1e+16"), (1e17, "1e+17"),
            (1.2345678901234567e19, "1.2345678901234567e+19"), (1e-4, "0.0001"),
            (1e-5, "1e-05"), (1.5e-5, "1.5e-05"), (0.00012, "0.00012"), (-2.5e-7, "-2.5e-07"),
            (-0.0, "-0.0"), (0, "0.0"), (1, "1.0"), (100, "100.0"), (1_234_567, "1234567.0"),
            (0.1, "0.1"), (1.0 / 3.0, "0.3333333333333333"), (5e-324, "5e-324"),
            (2.2250738585072014e-308, "2.2250738585072014e-308"),
            (1.7976931348623157e308, "1.7976931348623157e+308"), (1e300, "1e+300"),
            (-123_456_789.123, "-123456789.123"), (.nan, "NaN"), (.infinity, "Infinity"),
            (-.infinity, "-Infinity"),
        ]
        for (value, expected) in cases {
            #expect(PythonJSON.floatRepr(value) == expected)
        }
    }

    @Test func stateSerialization() throws {
        let state: DecisionState = "I was charged twice, please refund one."
        #expect(state == .text("I was charged twice, please refund one."))
        #expect(state.serialized(indent: 2) == "I was charged twice, please refund one.")
        #expect(state.serialized(indent: nil) == "I was charged twice, please refund one.")

        // Members keep the literal's order, as a Python dict keeps insertion order.
        let json = DecisionState.json(["name": "Zoë", "items": [1, 2.5], "ok": true, "note": nil])
        #expect(
            json.serialized(indent: 2)
                == "{\n  \"name\": \"Zoë\",\n  \"items\": [\n    1,\n    2.5\n  ],\n  \"ok\": true,\n  \"note\": null\n}"
        )
        #expect(
            json.serialized(indent: nil)
                == "{\"name\": \"Zoë\", \"items\": [1, 2.5], \"ok\": true, \"note\": null}"
        )
        #expect(DecisionState.json(.array([])).serialized(indent: 2) == "[]")

        // The case decides how a number is written; Python writes 12.0 for a float.
        let amounts = DecisionState.json(["amount": .double(12.0), "count": .int(12)])
        #expect(amounts.serialized(indent: nil) == "{\"amount\": 12.0, \"count\": 12}")

        // JSONValue keeps no order and decodes 1.0 as an int.
        let decoded = try JSONDecoder().decode(
            JSONValue.self, from: Data(#"{"b": 1.0, "a": 2.5}"#.utf8))
        #expect(PythonJSON.dumps(DecisionJSON(decoded)) == #"{"a": 2.5, "b": 1}"#)
    }
}

// Generated with Python json.dumps(value, ensure_ascii=False, sort_keys=True, indent=...).
private let pythonJSONCases: [PythonJSONCase] = [
    PythonJSONCase(
        value: .object([
            "customer": .object([
                "name": .string("Zo\u{eb}"), "tier": .string("gold"), "id": .int(1042),
            ]), "messages": .array([.string("I was charged twice"), .string("please refund one")]),
            "amount": .double(12.5), "refunded": .bool(false), "note": .null,
        ]),
        indented:
            "{\n  \"amount\": 12.5,\n  \"customer\": {\n    \"id\": 1042,\n    \"name\": \"Zo\u{eb}\",\n    \"tier\": \"gold\"\n  },\n  \"messages\": [\n    \"I was charged twice\",\n    \"please refund one\"\n  ],\n  \"note\": null,\n  \"refunded\": false\n}",
        compact:
            "{\"amount\": 12.5, \"customer\": {\"id\": 1042, \"name\": \"Zo\u{eb}\", \"tier\": \"gold\"}, \"messages\": [\"I was charged twice\", \"please refund one\"], \"note\": null, \"refunded\": false}"
    ),
    PythonJSONCase(
        value: .object([
            "b": .array([.int(1), .double(2.5), .object(["c": .null, "a": .bool(true)])]),
            "a": .object(["x": .string("y"), "list": .array([]), "obj": .object([:])]),
            "z": .array([.array([])]),
        ]),
        indented:
            "{\n  \"a\": {\n    \"list\": [],\n    \"obj\": {},\n    \"x\": \"y\"\n  },\n  \"b\": [\n    1,\n    2.5,\n    {\n      \"a\": true,\n      \"c\": null\n    }\n  ],\n  \"z\": [\n    []\n  ]\n}",
        compact:
            "{\"a\": {\"list\": [], \"obj\": {}, \"x\": \"y\"}, \"b\": [1, 2.5, {\"a\": true, \"c\": null}], \"z\": [[]]}"
    ),
    PythonJSONCase(
        value: .array([]),
        indented: "[]",
        compact: "[]"),
    PythonJSONCase(
        value: .object([:]),
        indented: "{}",
        compact: "{}"),
    PythonJSONCase(
        value: .array([.array([]), .object([:]), .array([.array([])]), .array([.object([:])])]),
        indented: "[\n  [],\n  {},\n  [\n    []\n  ],\n  [\n    {}\n  ]\n]",
        compact: "[[], {}, [[]], [{}]]"),
    PythonJSONCase(
        value: .object(["a": .array([]), "b": .object([:])]),
        indented: "{\n  \"a\": [],\n  \"b\": {}\n}",
        compact: "{\"a\": [], \"b\": {}}"),
    PythonJSONCase(
        value: .array([
            .int(1),
            .array([
                .int(2),
                .array([.int(3), .array([.int(4), .object(["deep": .array([.int(5)])])])]),
            ]),
        ]),
        indented:
            "[\n  1,\n  [\n    2,\n    [\n      3,\n      [\n        4,\n        {\n          \"deep\": [\n            5\n          ]\n        }\n      ]\n    ]\n  ]\n]",
        compact: "[1, [2, [3, [4, {\"deep\": [5]}]]]]"),
    PythonJSONCase(
        value: .string("caf\u{e9} \u{65e5}\u{672c}\u{8a9e} \u{1f600} e\u{301}"),
        indented: "\"caf\u{e9} \u{65e5}\u{672c}\u{8a9e} \u{1f600} e\u{301}\"",
        compact: "\"caf\u{e9} \u{65e5}\u{672c}\u{8a9e} \u{1f600} e\u{301}\""),
    PythonJSONCase(
        value: .string("line\u{2028}para\u{2029}end\u{85}"),
        indented: "\"line\u{2028}para\u{2029}end\u{85}\"",
        compact: "\"line\u{2028}para\u{2029}end\u{85}\""),
    PythonJSONCase(
        value: .string("\u{0}\u{1}\u{8}\t\n\u{b}\u{c}\r\u{1b}\u{1f}\u{7f}"),
        indented: "\"\\u0000\\u0001\\b\\t\\n\\u000b\\f\\r\\u001b\\u001f\u{7f}\"",
        compact: "\"\\u0000\\u0001\\b\\t\\n\\u000b\\f\\r\\u001b\\u001f\u{7f}\""),
    PythonJSONCase(
        value: .string("He said \"hi\" \\ C:\\path/to\\file 'single'"),
        indented: "\"He said \\\"hi\\\" \\\\ C:\\\\path/to\\\\file 'single'\"",
        compact: "\"He said \\\"hi\\\" \\\\ C:\\\\path/to\\\\file 'single'\""),
    PythonJSONCase(
        value: .string("<|reserved_7|> {not: json} [x]"),
        indented: "\"<|reserved_7|> {not: json} [x]\"",
        compact: "\"<|reserved_7|> {not: json} [x]\""),
    PythonJSONCase(
        value: .string(""),
        indented: "\"\"",
        compact: "\"\""),
    PythonJSONCase(
        value: .array([
            .int(0), .int(-1), .int(42), .int(9_007_199_254_740_993),
            .int(9_223_372_036_854_775_807), .int(-9_223_372_036_854_775_808),
        ]),
        indented:
            "[\n  0,\n  -1,\n  42,\n  9007199254740993,\n  9223372036854775807,\n  -9223372036854775808\n]",
        compact: "[0, -1, 42, 9007199254740993, 9223372036854775807, -9223372036854775808]"),
    PythonJSONCase(
        value: .array([.bool(true), .bool(false), .null]),
        indented: "[\n  true,\n  false,\n  null\n]",
        compact: "[true, false, null]"),
    PythonJSONCase(
        value: .bool(true),
        indented: "true",
        compact: "true"),
    PythonJSONCase(
        value: .null,
        indented: "null",
        compact: "null"),
    PythonJSONCase(
        value: .int(7),
        indented: "7",
        compact: "7"),
    PythonJSONCase(
        value: .double(-0.0),
        indented: "-0.0",
        compact: "-0.0"),
    PythonJSONCase(
        value: .object([
            "zeta": .int(1), "\u{e9}tat": .int(2), "Zulu": .int(3), "alpha": .int(4), "": .int(5),
            "a b": .int(6), "10": .int(7), "9": .int(8),
        ]),
        indented:
            "{\n  \"\": 5,\n  \"10\": 7,\n  \"9\": 8,\n  \"Zulu\": 3,\n  \"a b\": 6,\n  \"alpha\": 4,\n  \"zeta\": 1,\n  \"\u{e9}tat\": 2\n}",
        compact:
            "{\"\": 5, \"10\": 7, \"9\": 8, \"Zulu\": 3, \"a b\": 6, \"alpha\": 4, \"zeta\": 1, \"\u{e9}tat\": 2}"
    ),
    PythonJSONCase(
        value: .object([
            "quote\"key": .string("v"), "back\\slash": .string("w"), "new\nline": .string("x"),
            "tab\tkey": .string("y"),
        ]),
        indented:
            "{\n  \"back\\\\slash\": \"w\",\n  \"new\\nline\": \"x\",\n  \"quote\\\"key\": \"v\",\n  \"tab\\tkey\": \"y\"\n}",
        compact:
            "{\"back\\\\slash\": \"w\", \"new\\nline\": \"x\", \"quote\\\"key\": \"v\", \"tab\\tkey\": \"y\"}"
    ),
]
