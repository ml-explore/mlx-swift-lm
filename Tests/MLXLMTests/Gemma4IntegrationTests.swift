// Integration test for Gemma 4 model loading and inference.
// Run: xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests/Gemma4IntegrationTests

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

struct TestTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any Tokenizer {
        try SimpleTokenizer(directory: directory)
    }
}

struct SimpleTokenizer: Tokenizer {
    let vocab: [String: Int]
    let reverseVocab: [Int: String]
    let bosTokenStr: String?
    let eosTokenStr: String?

    var bosToken: String? { bosTokenStr }
    var eosToken: String? { eosTokenStr }
    var unknownToken: String? { nil }

    init(directory: URL) throws {
        let configURL = directory.appending(component: "tokenizer_config.json")
        let configData = try Data(contentsOf: configURL)
        let configJSON = try JSONSerialization.jsonObject(with: configData) as? [String: Any] ?? [:]
        bosTokenStr = configJSON["bos_token"] as? String
        eosTokenStr = configJSON["eos_token"] as? String

        let tokenizerURL = directory.appending(component: "tokenizer.json")
        let tokenizerData = try Data(contentsOf: tokenizerURL)
        let tokenizerJSON =
            try JSONSerialization.jsonObject(with: tokenizerData) as? [String: Any] ?? [:]
        let model = tokenizerJSON["model"] as? [String: Any] ?? [:]
        let vocabList = model["vocab"] as? [String: Int] ?? [:]
        self.vocab = vocabList
        self.reverseVocab = Dictionary(uniqueKeysWithValues: vocabList.map { ($1, $0) })
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens = [Int]()
        if addSpecialTokens, let bos = bosToken, let id = vocab[bos] { tokens.append(id) }
        for char in text.utf8 {
            if let id = vocab[String(UnicodeScalar(char))] { tokens.append(id) }
        }
        return tokens
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { reverseVocab[$0] }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { vocab[token] }
    func convertIdToToken(_ id: Int) -> String? { reverseVocab[id] }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var prompt = ""
        if let bos = bosToken { prompt += bos }
        for msg in messages {
            if let role = msg["role"] as? String, let content = msg["content"] as? String {
                prompt += "<start_of_turn>\(role)\n\(content)<end_of_turn>\n"
            }
        }
        prompt += "<start_of_turn>model\n"
        return encode(text: prompt, addSpecialTokens: false)
    }
}

public class Gemma4IntegrationTests: XCTestCase {

    /// Test loading the real Gemma 4 E4B model and running inference
    func testGemma4RealModel() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let modelDir =
            "\(home)/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-8bit/snapshots/dc1f72fa71acb997e1581a8ec8f69edd6e8f5707"

        guard FileManager.default.fileExists(atPath: modelDir) else {
            throw XCTSkip("Gemma 4 model not found in HF cache — download first")
        }

        let modelURL = URL(fileURLWithPath: modelDir)

        print("Loading Gemma 4 E4B 8-bit...")
        let start = Date()

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL, using: TestTokenizerLoader())

        let loadTime = Date().timeIntervalSince(start)
        print("Loaded in \(String(format: "%.1f", loadTime))s")

        // Generate
        let info: GenerateCompletionInfo = try await container.perform {
            (context: ModelContext) async throws -> GenerateCompletionInfo in
            let tokens = context.tokenizer.encode(text: "Hello")
            let input = LMInput(tokens: MLXArray(tokens))
            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: .init(maxTokens: 32, temperature: 0.0, topP: 1.0),
                context: context)

            var output = ""
            var completionInfo: GenerateCompletionInfo?
            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    output += text
                case .info(let i):
                    completionInfo = i
                default:
                    break
                }
            }
            print("Output: \(output)")
            return completionInfo ?? GenerateCompletionInfo(
                promptTokenCount: 0, generationTokenCount: 0,
                promptTime: 0, generationTime: 0)
        }

        print("Gen: \(info.generationTokenCount) tokens, \(String(format: "%.1f", info.tokensPerSecond)) tok/s")

        XCTAssertGreaterThan(info.generationTokenCount, 0, "Should generate at least 1 token")
    }
}
