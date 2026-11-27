// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXNN
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct ProbeWeatherArgs {
    @Guide(description: "City and country.")
    var location: String
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct ProbePlaceArgs {
    @Guide(description: "What to look for.")
    var query: String
}

/// Folders listed in `MLX_STOP_PROBE_DIRS`, or nil when it is unset.
private let stopProbeDirectories: [URL]? = {
    guard let value = ProcessInfo.processInfo.environment["MLX_STOP_PROBE_DIRS"], !value.isEmpty
    else { return nil }
    return value.split(separator: ":").map { URL(fileURLWithPath: String($0)) }
}()

/// Walks valid outputs on real vocabularies and compares the stop bits of
/// the mask with `canStop()`. Loads only tokenizer and config files, from the
/// colon-separated folders in `MLX_STOP_PROBE_DIRS`; skipped when it is unset.
@Suite(.serialized, .enabled(if: stopProbeDirectories != nil))
struct StopBitProbeIntegrationTests {

    private static let compilations =
        ProcessInfo.processInfo.environment["MLX_STOP_PROBE_COMPILATIONS"].flatMap { Int($0) } ?? 4
    private static let grammarFilter = ProcessInfo.processInfo.environment["MLX_STOP_PROBE_GRAMMAR"]

    private static let constrainedStrings = #"""
        {"type":"object","properties":{"code":{"type":"string","pattern":"^[A-Z]{3}-[0-9]{2}$"},"title":{"type":"string","maxLength":24},"subtitle":{"type":"string","maxLength":24},"color":{"type":"string","enum":["red","green","blue"]},"note":{"type":"string"}},"required":["code","title","subtitle","color","note"],"additionalProperties":false}
        """#

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func grammars() throws -> [(name: String, kind: Kind, outputs: [String])] {
        let tools = [
            Transcript.ToolDefinition(
                name: "get_weather", description: "Weather for a city",
                parameters: ProbeWeatherArgs.generationSchema),
            Transcript.ToolDefinition(
                name: "find_place", description: "Find a place",
                parameters: ProbePlaceArgs.generationSchema),
        ]
        return [
            (
                "itinerary-production", .json(TestFixtures.itinerarySchemaProduction),
                [itinerary(production: true)]
            ),
            (
                "itinerary-constrained", .json(TestFixtures.itinerarySchemaConstrained),
                [itinerary(production: false)]
            ),
            (
                "constrained-strings", .json(constrainedStrings),
                [
                    #"{"code":"ABC-12","title":"Rome in spring","subtitle":"Three days","color":"green","note":"bring a hat"}"#
                ]
            ),
            (
                "tool-calling", .tag(try SchemaConverter.encodeToolCallingGrammar(tools: tools)),
                [
                    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\":\"Rome, IT\"}}\n</tool_call>",
                    "{\"name\": \"find_place\", \"arguments\": {\"query\":\"a quiet cafe\"}}",
                ]
            ),
        ]
    }

    enum Kind {
        case json(String)
        case tag(String)
    }

    @Test func rawStopBitsMatchCanStop() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let directories = try #require(stopProbeDirectories)
        let grammars = try Self.grammars()
        for directory in directories {
            let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
            let configuration = ModelConfiguration(
                id: directory.lastPathComponent, eosTokenIds: Self.configEOSIds(in: directory))
            let stops = GuidedGenerationLoop.stopTokenIDs(
                tokenizer: tokenizer, configuration: configuration
            ).sorted()
            let vocabulary = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)
            let grammarTokenizer = try GrammarTokenizer(
                vocab: vocabulary.vocab, vocabType: vocabulary.vocabType,
                eosTokenId: Int32(tokenizer.eosTokenId ?? 0), stopTokenIds: stops)
            let speller = Speller(tokenizer: tokenizer, stops: Set(stops))
            let name = directory.deletingLastPathComponent().lastPathComponent

            for grammar in grammars where Self.grammarFilter.map({ grammar.name == $0 }) ?? true {
                var leaks: [String] = []
                var leakyCompilations = Set<Int>()
                var mismatches = 0
                var steps = 0
                for compilation in 0 ..< Self.compilations {
                    for (outputIndex, output) in grammar.outputs.enumerated() {
                        let constraint: GrammarConstraint
                        switch grammar.kind {
                        case .json(let schema):
                            constraint = try GrammarConstraint(
                                tokenizer: grammarTokenizer, jsonSchema: schema)
                        case .tag(let tag):
                            constraint = try GrammarConstraint(
                                tokenizer: grammarTokenizer, structuralTag: tag)
                        }
                        var remaining = Array(output.utf8)
                        var position = 0
                        while true {
                            let raw = try constraint.computeRawMask()
                            let mask = try constraint.computeMask()
                            let canStop = try constraint.canStop()
                            let rawBits = stopBits(
                                raw, stops: stops, vocabSize: grammarTokenizer.vocabSize)
                            let bits = stopBits(
                                mask, stops: stops, vocabSize: grammarTokenizer.vocabSize)
                            steps += 1
                            if rawBits == true && !canStop {
                                let done = Array(output.utf8).dropLast(remaining.count)
                                let consumed = done.suffix(24)
                                leaks.append(
                                    "c\(compilation) o\(outputIndex) p\(position) after '\(String(decoding: consumed, as: UTF8.self))'"
                                )
                                leakyCompilations.insert(compilation)
                            }
                            if bits != canStop { mismatches += 1 }
                            if remaining.isEmpty { break }
                            guard
                                let token = speller.next(
                                    for: remaining, mask: raw, vocabSize: grammarTokenizer.vocabSize
                                )
                            else {
                                Issue.record(
                                    "\(name) \(grammar.name): no allowed token for \(String(decoding: remaining.prefix(20), as: UTF8.self))"
                                )
                                break
                            }
                            _ = try constraint.commitToken(Int32(token.id))
                            remaining.removeFirst(token.length)
                            position += 1
                        }
                        #expect(
                            try constraint.canStop(),
                            "\(name) \(grammar.name) output \(outputIndex)")
                    }
                }
                print(
                    "[STOPPROBE] \(name) \(grammar.name): vocab \(grammarTokenizer.vocabSize), stops \(stops), steps \(steps), raw leaks \(leaks.count) in \(leakyCompilations.count)/\(Self.compilations) compilations, mask mismatches \(mismatches)"
                )
                for leak in leaks where leak.hasPrefix("c\(leakyCompilations.min() ?? 0) ") {
                    print("[STOPPROBE]   \(leak)")
                }
                #expect(mismatches == 0, "\(name) \(grammar.name)")
                // Upstream xgrammar: stop bits set where the grammar cannot end.
                withKnownIssue(
                    "xgrammar leaves stop bits set in cached bitset masks", isIntermittent: true
                ) {
                    #expect(leaks.isEmpty, "\(name) \(grammar.name): \(leaks.count) raw leaks")
                }
            }
        }
    }

    /// A model that always prefers the stop tokens, then the next token of a
    /// valid itinerary. Where xgrammar leaves stop bits set inside a string,
    /// the run must still finish the JSON, and a cap there is incomplete.
    @Test func loopFinishesWhereRawStopBitsLeak() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let directories = try #require(stopProbeDirectories)
        let output = Self.itinerary(production: false)
        for directory in directories {
            let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
            let configuration = ModelConfiguration(
                id: directory.lastPathComponent, eosTokenIds: Self.configEOSIds(in: directory))
            let stops = GuidedGenerationLoop.stopTokenIDs(
                tokenizer: tokenizer, configuration: configuration
            ).sorted()
            let vocabulary = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)
            let grammarTokenizer = try GrammarTokenizer(
                vocab: vocabulary.vocab, vocabType: vocabulary.vocabType,
                eosTokenId: Int32(tokenizer.eosTokenId ?? 0), stopTokenIds: stops)
            let name = directory.deletingLastPathComponent().lastPathComponent
            func constraint() throws -> GrammarConstraint {
                try GrammarConstraint(
                    tokenizer: grammarTokenizer, jsonSchema: TestFixtures.itinerarySchemaConstrained
                )
            }

            // The token script of the valid output, and where the raw mask leaks.
            let speller = Speller(tokenizer: tokenizer, stops: Set(stops))
            let walker = try constraint()
            var script: [Int] = []
            var leaks: [Int] = []
            var remaining = Array(output.utf8)
            while !remaining.isEmpty {
                let raw = try walker.computeRawMask()
                if stopBits(raw, stops: stops, vocabSize: grammarTokenizer.vocabSize) == true,
                    try !walker.canStop()
                {
                    leaks.append(script.count)
                }
                let token = try #require(
                    speller.next(for: remaining, mask: raw, vocabSize: grammarTokenizer.vocabSize))
                _ = try walker.commitToken(Int32(token.id))
                script.append(token.id)
                remaining.removeFirst(token.length)
            }
            print("[STOPPROBE] \(name) loop: \(script.count) tokens, raw leaks at \(leaks)")

            let full = try Self.runScripted(
                script: script, stops: stops, maxTokens: script.count + 8,
                constraint: try constraint(), tokenizer: tokenizer, configuration: configuration,
                vocabSize: grammarTokenizer.vocabSize)
            #expect(full.error == nil, "\(name): \(String(describing: full.error))")
            #expect(full.text == output, "\(name)")

            if let leak = leaks.first {
                let capped = try Self.runScripted(
                    script: script, stops: stops, maxTokens: leak,
                    constraint: try constraint(), tokenizer: tokenizer,
                    configuration: configuration, vocabSize: grammarTokenizer.vocabSize)
                #expect(
                    (capped.error as? GuidedGenerationError) == .incompleteOutput,
                    "\(name): cap at \(leak) gave \(String(describing: capped.error))")
            }
        }
    }

    private static func runScripted(
        script: [Int], stops: [Int], maxTokens: Int, constraint: GrammarConstraint,
        tokenizer: any MLXLMCommon.Tokenizer, configuration: ModelConfiguration, vocabSize: Int
    ) throws -> (text: String, error: (any Error)?) {
        let model = ScriptedModel(script: script, stops: stops, width: vocabSize, promptLength: 3)
        let context = ModelContext(
            configuration: configuration, model: model, processor: NoInputProcessor(),
            tokenizer: tokenizer)
        var text = ""
        do {
            try Stream.withNewDefaultStream(device: .cpu) {
                _ = try GuidedGenerationLoop.run(
                    input: LMInput(tokens: MLXArray([Int32(1), 2, 3])), context: context,
                    constraint: constraint, maxTokens: maxTokens, vocabSize: vocabSize
                ) { delta in
                    text += delta
                    return true
                }
            }
        } catch {
            return (text, error)
        }
        return (text, nil)
    }

    /// `eos_token_id` from `config.json` and `generation_config.json`.
    private static func configEOSIds(in directory: URL) -> Set<Int> {
        var ids = Set<Int>()
        for file in ["config.json", "generation_config.json"] {
            guard let data = try? Data(contentsOf: directory.appendingPathComponent(file)),
                let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            if let id = object["eos_token_id"] as? Int { ids.insert(id) }
            if let list = object["eos_token_id"] as? [Int] { ids.formUnion(list) }
        }
        return ids
    }

    /// A valid itinerary with 3 days of 3 activities, keys in schema order.
    private static func itinerary(production: Bool) -> String {
        func activity(_ day: Int, _ index: Int) -> String {
            #"{"type":"sightseeing","title":"Stop \#(day).\#(index)","description":"A short walk"}"#
        }
        func day(_ day: Int) -> String {
            let activities = (1 ... 3).map { activity(day, $0) }.joined(separator: ",")
            return production
                ? #"{"activities":[\#(activities)],"subtitle":"Day \#(day)","destination":"Fuji","title":"Day \#(day) plan"}"#
                : #"{"title":"Day \#(day) plan","subtitle":"Day \#(day)","destination":"Fuji","activities":[\#(activities)]}"#
        }
        let days = (1 ... 3).map(day).joined(separator: ",")
        return production
            ? #"{"rationale":"Calm pace","days":[\#(days)],"title":"Fuji trip","destinationName":"Mount Fuji","description":"Three days"}"#
            : #"{"title":"Fuji trip","destinationName":"Mount Fuji","description":"Three days","rationale":"Calm pace","days":[\#(days)]}"#
    }
}

/// Logits that put the stop tokens first and the next script token second.
private final class ScriptedModel: Module, MLXLMCommon.LanguageModel {
    let script: [Int]
    let stops: [Int]
    let width: Int
    let promptLength: Int
    var received = 0

    init(script: [Int], stops: [Int], width: Int, promptLength: Int) {
        self.script = script
        self.stops = stops
        self.width = width
        self.promptLength = promptLength
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult { .tokens(input.text) }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        received += input.tokens.size
        var row = [Float](repeating: 0, count: width)
        for stop in stops where stop < width { row[stop] = 30 }
        let next = received - promptLength
        if next >= 0, next < script.count { row[script[next]] = 20 }
        return LMOutput(logits: MLXArray(row).reshaped(1, 1, width))
    }
}

private struct NoInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        fatalError("the test supplies the prepared input")
    }
}

/// Spells bytes with the longest token the mask allows.
private struct Speller {
    private var byFirstByte: [UInt8: [(id: Int, bytes: [UInt8])]] = [:]

    init(tokenizer: any MLXLMCommon.Tokenizer, stops: Set<Int>) {
        let data = TokenizerVocabExtractor.extract(from: tokenizer)
        var offset = 0
        for (id, length) in data.tokenLens.enumerated() {
            let bytes = Array(data.tokenBytes[offset ..< offset + Int(length)])
            offset += Int(length)
            guard let first = bytes.first, !stops.contains(id) else { continue }
            byFirstByte[first, default: []].append((id, bytes))
        }
        for key in byFirstByte.keys {
            byFirstByte[key]!.sort { $0.bytes.count > $1.bytes.count }
        }
    }

    func next(for remaining: [UInt8], mask: MaskResult, vocabSize: Int) -> (id: Int, length: Int)? {
        guard let first = remaining.first else { return nil }
        for candidate in byFirstByte[first] ?? []
        where candidate.bytes.count <= remaining.count
            && remaining.starts(with: candidate.bytes)
            && GuidedGenerationLoop.maskAllows(mask, tokenId: candidate.id, vocabSize: vocabSize)
        {
            return (candidate.id, candidate.bytes.count)
        }
        return nil
    }
}

/// The declared stop bits of `mask`: all set, all clear, or mixed (nil).
private func stopBits(_ mask: MaskResult, stops: [Int], vocabSize: Int) -> Bool? {
    let bits = stops.filter { $0 < vocabSize }.map {
        GuidedGenerationLoop.maskAllows(mask, tokenId: $0, vocabSize: vocabSize)
    }
    if bits.allSatisfy({ $0 }) { return true }
    if bits.allSatisfy({ !$0 }) { return false }
    return nil
}

#endif
