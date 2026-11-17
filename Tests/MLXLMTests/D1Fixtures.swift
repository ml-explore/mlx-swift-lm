// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXDecisions

/// Checkpoints and fixtures for the gated real-weight d1 suites, named by environment
/// variables. `tools/d1_fixtures.py` writes the fixture directory; README.md explains how
/// to run the suites.
enum D1Fixtures {
    private static let environment = ProcessInfo.processInfo.environment

    /// The `LiquidAI/d1-3B` snapshot, `MLX_D1_3B_MODEL`.
    static let d1Model = environment["MLX_D1_3B_MODEL"].map { URL(fileURLWithPath: $0) }

    /// The `LiquidAI/d1-omni-600M` snapshot, `MLX_D1_OMNI_MODEL`.
    static let omniModel = environment["MLX_D1_OMNI_MODEL"].map { URL(fileURLWithPath: $0) }

    /// The directory `tools/d1_fixtures.py` writes, `MLX_D1_FIXTURES`.
    static let directory = environment["MLX_D1_FIXTURES"].map { URL(fileURLWithPath: $0) }

    /// A fixture or input file in ``directory``. Fails when it is missing.
    static func url(_ name: String) throws -> URL {
        let url = try #require(directory, "Set MLX_D1_FIXTURES").appending(component: name)
        try #require(
            FileManager.default.fileExists(atPath: url.path),
            "\(url.path) is missing; write it with tools/d1_fixtures.py")
        return url
    }

    /// One question of a fixture case, in the order Python asked it.
    struct Question: Decodable {
        var name: String
        var type: String
        var instructions: String
        /// `[label, description or null]`
        var options: [[String?]]?
        var levels: [String]?
        var whenTrue: String?
        var whenFalse: String?

        enum CodingKeys: String, CodingKey {
            case name, type, instructions, options, levels
            case whenTrue = "when_true"
            case whenFalse = "when_false"
        }

        var named: DecisionRequest.NamedQuestion {
            let question: DecisionQuestion =
                switch type {
                case "choice":
                    .choice(
                        instructions,
                        options: (options ?? []).map {
                            DecisionOption($0[0] ?? "", description: $0.count > 1 ? $0[1] : nil)
                        })
                case "score": .score(instructions, levels: levels ?? [])
                default: .bool(instructions, whenTrue: whenTrue, whenFalse: whenFalse)
                }
            return .init(name: name, question: question)
        }
    }

    /// A fixture state: a JSON string is text, as Python reads a `str`.
    static func state(_ value: JSONValue?) -> DecisionState? {
        switch value {
        case nil, .null?: nil
        case .string(let text)?: .text(text)
        case let value?: .json(DecisionJSON(value))
        }
    }
}

func maxAbsDiff(_ a: [[Double]], _ b: [[Double]]) -> Double {
    precondition(a.map(\.count) == b.map(\.count))
    return zip(a.joined(), b.joined()).map { abs($0 - $1) }.max() ?? 0
}

/// The tiny `d1` config from mlx-vlm `tests/extraction_cases.json`, as an `lfm2_vl` checkpoint.
/// `extra` goes first in the object, e.g. an `auto_map` entry.
func tinyConfigJSON(_ extra: String = "") -> String {
    """
    {\(extra)
      "model_type": "lfm2_vl",
      "text_config": {
        "model_type": "lfm2", "vocab_size": 320, "hidden_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 32,
        "block_dim": 16, "block_multiple_of": 8, "block_auto_adjust_ff_dim": false,
        "layer_types": ["conv", "full_attention"], "max_position_embeddings": 4096
      },
      "vision_config": {
        "model_type": "siglip2_vision_model", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_patches": 16
      },
      "projector_hidden_size": 16, "image_token_id": 303
    }
    """
}

/// One token per vocabulary entry or special token, one per code point otherwise
/// (`characterBase + code point`), as Python's `_VocabTokenizer`.
struct VocabTokenizer: Tokenizer {
    var vocabulary: [String]
    var specialTokens: [String: Int] = [:]
    var characterBase = 1000
    var bos: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if let index = vocabulary.firstIndex(of: text) {
            return [index]
        }
        let scalars = Array(text.unicodeScalars)
        var ids = [Int]()
        var i = 0
        scan: while i < scalars.count {
            for (token, id) in specialTokens {
                let pattern = Array(token.unicodeScalars)
                if scalars[i...].starts(with: pattern) {
                    ids.append(id)
                    i += pattern.count
                    continue scan
                }
            }
            ids.append(characterBase + Int(scalars[i].value))
            i += 1
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { specialTokens[token] }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { bos }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
