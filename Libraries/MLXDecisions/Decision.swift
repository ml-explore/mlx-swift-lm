// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

// MARK: - Questions

/// The kind of answer a decision question asks for.
public enum DecisionQuestionKind: String, Sendable, Hashable {
    /// Pick one label from a set of options.
    case choice
    /// Rate on an ordered scale of levels, lowest first.
    case score
    /// Answer yes or no.
    case bool
}

/// Media a decision model can read in addition to, or instead of, a text state.
public enum DecisionMediaKind: String, Sendable, Hashable {
    /// One or more still images.
    case images
}

/// One option of a choice question.
public struct DecisionOption: Sendable, Hashable, ExpressibleByStringLiteral {
    /// The label returned as the answer value.
    public var label: String

    /// Text the model reads for this option. When `nil` or empty, the model reads the label.
    public var description: String?

    /// Create an option with a label and an optional description.
    public init(_ label: String, description: String? = nil) {
        self.label = label
        self.description = description
    }

    /// Create an option whose label is its only text.
    public init(stringLiteral value: String) {
        self.init(value)
    }
}

/// A typed question asked about a decision state.
public struct DecisionQuestion: Sendable, Hashable {
    /// What the answer is chosen from.
    public enum Criteria: Sendable, Hashable {
        /// Options in the order the model reads them.
        case choice([DecisionOption])
        /// Level descriptions, lowest first.
        case score([String])
        /// Optional descriptions of what `true` and `false` mean.
        case bool(whenTrue: String?, whenFalse: String?)
    }

    /// The question the model answers.
    public var instructions: String

    /// What the answer is chosen from.
    public var criteria: Criteria

    /// The kind of answer this question asks for.
    public var kind: DecisionQuestionKind {
        switch criteria {
        case .choice: .choice
        case .score: .score
        case .bool: .bool
        }
    }

    /// Create a question from instructions and criteria.
    public init(instructions: String, criteria: Criteria) {
        self.instructions = instructions
        self.criteria = criteria
    }

    /// A question answered by one of `options`.
    public static func choice(_ instructions: String, options: [DecisionOption]) -> Self {
        Self(instructions: instructions, criteria: .choice(options))
    }

    /// A question answered on a scale of `levels`, lowest first.
    public static func score(_ instructions: String, levels: [String]) -> Self {
        Self(instructions: instructions, criteria: .score(levels))
    }

    /// A yes-or-no question, optionally describing what each answer means.
    public static func bool(
        _ instructions: String, whenTrue: String? = nil, whenFalse: String? = nil
    ) -> Self {
        Self(instructions: instructions, criteria: .bool(whenTrue: whenTrue, whenFalse: whenFalse))
    }
}

/// A JSON value whose object members keep their order.
///
/// Models read a JSON state in the layout they were trained on, and key order changes their
/// answers, so objects keep the order you write them in. Dictionary literals keep source order:
///
/// ```swift
/// let state: DecisionJSON = ["premise": "A man is sleeping.", "hypothesis": "A man rests."]
/// ```
public indirect enum DecisionJSON: Sendable, Hashable {
    /// JSON `null`.
    case null
    /// `true` or `false`.
    case bool(Bool)
    /// Written without a decimal point, as `1`.
    case int(Int)
    /// Written as Python's `repr(float)`, as `1.0`.
    case double(Double)
    /// A string.
    case string(String)
    /// Items in order.
    case array([DecisionJSON])
    /// Members in the order the model reads them.
    case object([Member])

    /// One key and value of a JSON object.
    public struct Member: Sendable, Hashable {
        /// The member's key.
        public var key: String
        /// The member's value.
        public var value: DecisionJSON

        /// Create an object member.
        public init(key: String, value: DecisionJSON) {
            self.key = key
            self.value = value
        }
    }

    /// Convert a `JSONValue`, which keeps no member order, so object keys come out sorted.
    init(_ value: JSONValue) {
        switch value {
        case .null: self = .null
        case .bool(let flag): self = .bool(flag)
        case .int(let number): self = .int(number)
        case .double(let number): self = .double(number)
        case .string(let text): self = .string(text)
        case .array(let items): self = .array(items.map(DecisionJSON.init))
        case .object(let members):
            self = .object(
                members.keys.sorted().map { Member(key: $0, value: DecisionJSON(members[$0]!)) })
        }
    }
}

extension DecisionJSON: ExpressibleByNilLiteral, ExpressibleByBooleanLiteral,
    ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral, ExpressibleByStringLiteral,
    ExpressibleByArrayLiteral, ExpressibleByDictionaryLiteral
{
    /// `null`.
    public init(nilLiteral: ()) { self = .null }
    /// A boolean.
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    /// An integer.
    public init(integerLiteral value: Int) { self = .int(value) }
    /// A floating-point number.
    public init(floatLiteral value: Double) { self = .double(value) }
    /// A string.
    public init(stringLiteral value: String) { self = .string(value) }
    /// An array whose items keep the literal's order.
    public init(arrayLiteral elements: DecisionJSON...) { self = .array(elements) }

    /// An object whose members keep the literal's order.
    public init(dictionaryLiteral elements: (String, DecisionJSON)...) {
        self = .object(elements.map { Member(key: $0.0, value: $0.1) })
    }
}

/// The state a decision model reads before its questions.
public enum DecisionState: Sendable, Hashable, ExpressibleByStringLiteral {
    /// Text read as written.
    case text(String)

    /// A JSON value. Each model writes it in the layout it was trained on, with object members
    /// in the order given. A JSON `null` counts as no state, and a JSON string as text.
    case json(DecisionJSON)

    /// Create a text state.
    public init(stringLiteral value: String) {
        self = .text(value)
    }

    /// The state as prompt text: text as written, JSON as `json.dumps` writes it.
    ///
    /// Models never see a JSON `null` or string: ``DecisionContainer`` passes them on as no
    /// state and as text.
    func serialized(indent: Int?) -> String {
        switch self {
        case .text(let text): text
        case .json(let value): PythonJSON.dumps(value, indent: indent)
        }
    }
}

// MARK: - Requests and responses

/// Named questions about one state and its media.
///
/// Media are not `Sendable`, so a request is consumed by the call that predicts it.
public struct DecisionRequest {
    /// A question and the name its answer is reported under.
    public struct NamedQuestion: Sendable, Hashable {
        /// The name the answer is reported under. Names must be unique within a request.
        public var name: String

        /// The question.
        public var question: DecisionQuestion

        /// Create a named question.
        public init(name: String, question: DecisionQuestion) {
            self.name = name
            self.question = question
        }
    }

    /// The state to decide about. Optional when images carry the whole state.
    ///
    /// A JSON `null` counts as no state, as Python's `None` does.
    public var state: DecisionState?

    /// Questions in the order their answers are reported.
    public var questions: [NamedQuestion]

    /// Images, read in order: file or `data:` URLs, Core Image images, or arrays. Other URLs
    /// throw ``DecisionError/invalidImage(_:)``, because the library does not download media.
    public var images: [UserInput.Image]

    /// Create a request from ordered named questions.
    public init(
        state: DecisionState? = nil,
        questions: [NamedQuestion],
        images: [UserInput.Image] = []
    ) {
        self.state = state
        self.questions = questions
        self.images = images
    }

    /// Create a request from a dictionary literal of named questions, keeping its order.
    init(
        state: DecisionState? = nil,
        questions: KeyValuePairs<String, DecisionQuestion>,
        images: [UserInput.Image] = []
    ) {
        self.init(
            state: state,
            questions: questions.map { NamedQuestion(name: $0.key, question: $0.value) },
            images: images)
    }
}

/// Execution limits for a decision request.
public struct DecisionExecutionOptions: Sendable, Equatable {
    /// Maximum padded token positions in one model batch, media prefixes included.
    ///
    /// Peak memory grows with the batch, so use a smaller budget on devices with less memory.
    /// A question longer than the budget still runs, alone in its batch.
    public var tokenBudget: Int

    /// Create execution options.
    public init(tokenBudget: Int = 16_384) {
        self.tokenBudget = tokenBudget
    }
}

/// The answer to one named question.
public struct DecisionAnswer: Sendable, Hashable {
    /// The answer value.
    public enum Value: Sendable, Hashable {
        /// The most probable option label.
        case choice(String)
        /// The expected level, `Σ level × probability`.
        case score(Double)
        /// Whether the probability of `true` is at least 0.5.
        case bool(Bool)
    }

    /// The question name.
    public let name: String

    /// The answer value.
    public let value: Value

    /// The probability of each option in option order: choice options, score levels lowest
    /// first, or `[P(false), P(true)]` for a bool question.
    public let probabilities: [Double]

    /// The probability of the most probable option.
    public var confidence: Double { probabilities.max() ?? 0 }
}

/// The answers to a decision request.
public struct DecisionResponse: Sendable {
    /// Identifier of the model that answered.
    public let modelID: String

    /// Answers in request order.
    public let answers: [DecisionAnswer]

    /// Input positions, counted as the model's reference counts them.
    public let inputTokens: Int

    /// The answer to the question named `name`.
    public subscript(name: String) -> DecisionAnswer? {
        answers.first { $0.name == name }
    }
}

/// Raw model output for a decision request.
struct DecisionProbabilities: Sendable, Equatable {
    /// Per question, in request order, the probability of each option in option order.
    /// Bool questions give `[P(false), P(true)]`.
    var probabilities: [[Double]]

    /// Input positions, counted as the model's reference counts them.
    var inputTokens: Int
}

/// The questions and media a decision model accepts.
public struct DecisionCapabilities: Sendable, Hashable {
    /// Question kinds the model answers.
    public let questionKinds: Set<DecisionQuestionKind>

    /// Media the model reads.
    public let media: Set<DecisionMediaKind>

    /// The most levels a score question may have, if the model limits them.
    public let maximumScoreLevels: Int?

    init(
        questionKinds: Set<DecisionQuestionKind>,
        media: Set<DecisionMediaKind>,
        maximumScoreLevels: Int? = nil
    ) {
        self.questionKinds = questionKinds
        self.media = media
        self.maximumScoreLevels = maximumScoreLevels
    }
}

// MARK: - Errors

/// Errors raised while validating or answering a decision request.
public enum DecisionError: LocalizedError, Sendable, Equatable {
    /// The model cannot answer decision questions.
    case unsupportedModel(String)
    /// The model cannot read this kind of media.
    case unsupportedMedia(DecisionMediaKind)
    /// The request has no questions.
    case noQuestions
    /// Two questions share a name.
    case duplicateQuestionName(String)
    /// The model cannot answer this kind of question.
    case unsupportedQuestionKind(name: String, kind: DecisionQuestionKind)
    /// A choice or score question has too few options or levels.
    case tooFewCriteria(name: String, kind: DecisionQuestionKind, minimum: Int)
    /// A score question has more levels than the model reads.
    case tooManyScoreLevels(name: String, count: Int, maximum: Int)
    /// A choice option has an empty label.
    case emptyLabel(name: String)
    /// Two options of a choice question share a label.
    case duplicateLabel(name: String, label: String)
    /// The request has no state or images.
    case missingState
    /// The tokenizer has no distinct single-token code left for every option.
    case noSingleTokenAlias(optionCount: Int)
    /// The tokenizer has no token the model needs, such as an answer or an image marker.
    case missingToken(String)
    /// With images, the state or a lone question contains the text the model reserves for images.
    case imagePlaceholderInText(String)
    /// An image could not be read, or its URL is neither a file nor a `data:` URL.
    case invalidImage(String)
    /// An execution option is out of range.
    case invalidExecutionOption(name: String, value: Int)
    /// The model returned malformed probabilities.
    case invalidProbabilities(String)

    /// A message that describes the error.
    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let message):
            message
        case .unsupportedMedia(.images):
            "This model does not support image input"
        case .noQuestions:
            "At least one named question is required"
        case .duplicateQuestionName(let name):
            "Question names must be unique: \(name)"
        case .unsupportedQuestionKind(let name, let kind):
            "This model does not support '\(kind.rawValue)' decisions (question \(name))"
        case .tooFewCriteria(let name, let kind, let minimum):
            "\(kind.rawValue) needs at least \(minimum) criteria (question \(name))"
        case .tooManyScoreLevels(let name, let count, let maximum):
            "A score has at most \(maximum) levels, got \(count) (question \(name))"
        case .emptyLabel(let name):
            "Decision labels must be nonempty strings (question \(name))"
        case .duplicateLabel(let name, let label):
            "Decision labels must be unique: \(label) (question \(name))"
        case .missingState:
            "A state is required unless images are given"
        case .noSingleTokenAlias(let count):
            "No single-token alias left for \(count) options"
        case .missingToken(let message):
            message
        case .imagePlaceholderInText(let placeholder):
            "The state or a question contains \(placeholder), which the model reads as an image"
        case .invalidImage(let message):
            "Failed to load image: \(message)"
        case .invalidExecutionOption(let name, let value):
            "\(name) must be positive, got \(value)"
        case .invalidProbabilities(let message):
            message
        }
    }
}
