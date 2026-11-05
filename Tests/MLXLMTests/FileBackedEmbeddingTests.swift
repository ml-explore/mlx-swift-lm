// Copyright © 2026 Apple Inc.

#if canImport(Darwin)
import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXScriptedLM
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class FileBackedEmbeddingTests: XCTestCase {
    private final class Model: Module, BaseLanguageModel, InputOnlyEmbeddingsProviding {
        let lookupOnly: Bool
        var inputOnlyEmbeddings: [Embedding] { lookupOnly ? [embedding] : [] }
        @ModuleInfo(key: "embedding") var embedding: Embedding

        init(_ embedding: Embedding, lookupOnly: Bool = true) {
            self.lookupOnly = lookupOnly
            _embedding.wrappedValue = embedding
            super.init()
        }
    }

    private final class AliasedModel: Module, InputOnlyEmbeddingsProviding {
        var inputOnlyEmbeddings: [Embedding] { [first] }
        @ModuleInfo(key: "first") var first: Embedding
        @ModuleInfo(key: "second") var second: Embedding

        init(_ embedding: Embedding) {
            _first.wrappedValue = embedding
            _second.wrappedValue = embedding
            super.init()
        }
    }

    private final class CustomEmbedding: QuantizedEmbedding {}

    private func embedding(
        dtype: DType = .bfloat16, bits: Int = 4, groupSize: Int = 64
    ) -> QuantizedEmbedding {
        let values = withRandomState(MLXRandom.RandomState(seed: 42)) {
            MLXRandom.normal([257, 256]).asType(dtype)
        }
        return QuantizedEmbedding(weight: values, groupSize: groupSize, bits: bits)
    }

    private func assertIdentical(
        _ actual: MLXArray, _ expected: MLXArray,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(
            actual.asData(access: .copy).data, expected.asData(access: .copy).data,
            file: file, line: line)
    }

    func testLoaderSelectsDeclaredSparseTablesAndHonorsResidentOverride() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = Model(embedding())
        try save(
            arrays: Dictionary(uniqueKeysWithValues: source.parameters().flattened()),
            url: directory.appendingPathComponent("model.safetensors"))
        let automatic = Model(embedding())
        try loadWeights(modelDirectory: directory, model: automatic)
        XCTAssertTrue(automatic.embedding is FileBackedQuantizedEmbedding)
        assertIdentical(
            automatic.embedding(MLXArray([0, 256])), source.embedding(MLXArray([0, 256])))
        let firstEmbedding = automatic.embedding
        let firstOutput = firstEmbedding(MLXArray([0, 256])).asData(access: .copy).data
        source.apply { $0.dtype.isFloatingPoint ? $0 * 2 : $0 }
        let replacement = directory.appendingPathComponent("replacement.safetensors")
        try save(
            arrays: Dictionary(uniqueKeysWithValues: source.parameters().flattened()),
            url: replacement)
        let original = directory.appendingPathComponent("model.safetensors")
        try FileManager.default.removeItem(at: original)
        try FileManager.default.moveItem(at: replacement, to: original)
        XCTAssertEqual(firstEmbedding(MLXArray([0, 256])).asData(access: .copy).data, firstOutput)
        try loadWeights(modelDirectory: directory, model: automatic)
        XCTAssertTrue(automatic.embedding is FileBackedQuantizedEmbedding)
        XCTAssertFalse(automatic.embedding === firstEmbedding)
        assertIdentical(
            automatic.embedding(MLXArray([0, 256])), source.embedding(MLXArray([0, 256])))
        try loadWeights(modelDirectory: directory, model: automatic, embeddingStorage: .resident)
        XCTAssertTrue(type(of: automatic.embedding) == QuantizedEmbedding.self)
        assertIdentical(
            automatic.embedding(MLXArray([0, 256])), source.embedding(MLXArray([0, 256])))
        let resident = Model(embedding())
        try loadWeights(modelDirectory: directory, model: resident, embeddingStorage: .resident)
        XCTAssertTrue(type(of: resident.embedding) == QuantizedEmbedding.self)
        let dense = Model(embedding(), lookupOnly: false)
        try loadWeights(modelDirectory: directory, model: dense)
        XCTAssertTrue(type(of: dense.embedding) == QuantizedEmbedding.self)
    }

    func testStoragePolicySurvivesConfigurationResolution() {
        let directory = FileManager.default.temporaryDirectory
        let configuration = ModelConfiguration(id: "test/model", embeddingStorage: .resident)
        let resolved = configuration.resolved(
            modelDirectory: directory, tokenizerDirectory: directory)
        XCTAssertEqual(resolved.embeddingStorage, .resident)
        XCTAssertEqual(ModelConfiguration(directory: directory).embeddingStorage, .automatic)
        XCTAssertNotEqual(configuration, ModelConfiguration(id: "test/model"))
    }

    func testModelsDeclareOnlyUntiedInputEmbeddings() throws {
        for tied in [false, true] {
            let data = Data(
                """
                {"model_type":"qwen3_5","hidden_size":256,"num_hidden_layers":1,"intermediate_size":512,
                 "num_attention_heads":4,"num_key_value_heads":2,"head_dim":64,
                 "rms_norm_eps":0.000001,"vocab_size":257,"full_attention_interval":1,
                 "num_experts":2,"num_experts_per_tok":1,"decoder_sparse_step":1,
                 "mlp_only_layers":[],"moe_intermediate_size":256,
                 "max_position_embeddings":1024,"original_max_position_embeddings":1024,
                 "tie_word_embeddings":\(tied)}
                """.utf8)
            let decoder = JSONDecoder()
            let models: [any InputOnlyEmbeddingsProviding] = [
                LlamaModel(try decoder.decode(LlamaConfiguration.self, from: data)),
                Qwen2Model(try decoder.decode(Qwen2Configuration.self, from: data)),
                Qwen3Model(try decoder.decode(Qwen3Configuration.self, from: data)),
                Qwen3MoEModel(try decoder.decode(Qwen3MoEConfiguration.self, from: data)),
                Phi3Model(try decoder.decode(Phi3Configuration.self, from: data)),
                Qwen35TextModel(try decoder.decode(Qwen35TextConfiguration.self, from: data)),
                Qwen35Model(try decoder.decode(Qwen35Configuration.self, from: data)),
            ]
            for model in models {
                XCTAssertEqual(model.inputOnlyEmbeddings.count, tied ? 0 : 1)
                if let input = model.inputOnlyEmbeddings.first {
                    XCTAssertTrue(model.namedModules().contains { $0.1 === input })
                }
            }
        }
    }

    func testGemma4DeclaresPerLayerAndUntiedInputTables() throws {
        for tied in [false, true] {
            let data = Data(
                """
                {"model_type":"gemma4_text","hidden_size":8,"num_hidden_layers":2,
                 "intermediate_size":16,"num_attention_heads":2,"num_key_value_heads":1,
                 "head_dim":4,"global_head_dim":4,"vocab_size":12,"vocab_size_per_layer_input":12,
                 "hidden_size_per_layer_input":4,"num_kv_shared_layers":0,"sliding_window":8,
                 "sliding_window_pattern":1,"max_position_embeddings":32,"rms_norm_eps":0.000001,
                 "rope_traditional":false,"use_double_wide_mlp":false,"enable_moe_block":false,
                 "attention_k_eq_v":false,"layer_types":["full_attention","full_attention"],
                 "rope_parameters":{},"tie_word_embeddings":\(tied)}
                """.utf8)
            let decoder = JSONDecoder()
            let text = Gemma4TextModel(
                try decoder.decode(MLXLLM.Gemma4TextConfiguration.self, from: data))
            let vision = Gemma4TextLanguageModel(
                try decoder.decode(MLXVLM.Gemma4TextConfiguration.self, from: data))
            let models: [any InputOnlyEmbeddingsProviding] = [text, vision]
            for model in models {
                let names = model.inputOnlyEmbeddings.compactMap { embedding in
                    model.namedModules().first { $0.1 === embedding }?.0
                }
                let expected =
                    ["model.embed_tokens_per_layer"] + (tied ? [] : ["model.embed_tokens"])
                XCTAssertEqual(Set(names), Set(expected))
                XCTAssertEqual(names.count, expected.count)
            }
        }
    }

    func testPhiMoEDeclaresItsInputTable() throws {
        let data = Data(
            """
            {"model_type":"phimoe","vocab_size":257,"hidden_size":256,"intermediate_size":512,
             "num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,
             "max_position_embeddings":1024,"original_max_position_embeddings":1024,
             "rms_norm_eps":0.000001,"num_local_experts":2,"num_experts_per_tok":1,"rope_theta":10000}
            """.utf8)
        let model = PhiMoEModel(try JSONDecoder().decode(PhiMoEConfiguration.self, from: data))
        XCTAssertEqual(model.inputOnlyEmbeddings.count, 1)
        XCTAssertTrue(model.namedModules().contains { $0.1 === model.inputOnlyEmbeddings.first })
    }

    func testLookupAndLinearProjectionPreserveAllBits() throws {
        let directory = FileManager.default.temporaryDirectory
        for dtype in [DType.float16, .bfloat16, .float32] {
            for bits in [2, 3, 4, 5, 6, 8] {
                let original = embedding(dtype: dtype, bits: bits)
                let model = Model(original)
                model.train(false)
                XCTAssertEqual(try prepareEmbeddingStorage(in: model, directory: directory), 1)
                XCTAssertFalse(model.embedding.training)
                XCTAssertTrue(model.embedding.trainableParameters().flattened().isEmpty)
                for indices in [
                    MLXArray(Int32(0)), MLXArray([0, 256, 19, 19, -1, -257]),
                    MLXArray([0, 256, 19, 19, 17, 4]).reshaped(2, 3),
                ] {
                    assertIdentical(model.embedding(indices), original(indices))
                }
                let input = original(MLXArray([0, 1]))
                assertIdentical(model.embedding.asLinear(input), original.asLinear(input))
                for (name, value) in original.parameters().flattened() {
                    let copied = Dictionary(
                        uniqueKeysWithValues: model.embedding.parameters().flattened())
                    let actual = try XCTUnwrap(copied[name])
                    assertIdentical(actual, value)
                }
            }
        }
    }

    func testAdditionalAffineGroupSizesPreserveAllBits() throws {
        for groupSize in [32, 128] {
            for dtype in [DType.float16, .bfloat16, .float32] {
                for bits in [2, 3, 4, 5, 6, 8] {
                    let original = embedding(dtype: dtype, bits: bits, groupSize: groupSize)
                    let model = Model(original)
                    try prepareEmbeddingStorage(
                        in: model, directory: FileManager.default.temporaryDirectory)
                    let replacement = try XCTUnwrap(model.embedding as? QuantizedEmbedding)
                    XCTAssertEqual(replacement.groupSize, groupSize)
                    assertIdentical(
                        replacement(MLXArray([0, 256, -1])), original(MLXArray([0, 256, -1])))
                    for (name, expected) in original.parameters().flattened() {
                        let actual = Dictionary(
                            uniqueKeysWithValues: replacement.parameters().flattened())
                        assertIdentical(try XCTUnwrap(actual[name]), expected)
                    }
                }
            }
        }
    }

    private struct TestTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer { PseudoWordTokenizer() }
    }

    func testQwen35FactoriesApplyStoragePolicyOnlyToUntiedInputTables() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        for tied in [false, true] {
            let config = Data(
                """
                {"model_type":"qwen3_5", "quantization":{"group_size":32,"bits":4},
                 "text_config":{"model_type":"qwen3_5_text","hidden_size":256,
                 "num_hidden_layers":1,"intermediate_size":512,"num_attention_heads":4,
                 "num_key_value_heads":2,"head_dim":64,"vocab_size":257,"full_attention_interval":1,
                 "tie_word_embeddings":\(tied),"rope_parameters":{"type":"default",
                 "mrope_section":[8,12,12],"partial_rotary_factor":1.0,"rope_theta":100000}},
                 "vision_config":{"model_type":"qwen3_5","depth":1,"hidden_size":32,
                 "intermediate_size":64,"out_hidden_size":256,"num_heads":4,"patch_size":2,
                 "spatial_merge_size":2,"temporal_patch_size":1,"num_position_embeddings":16}}
                """.utf8)
            try config.write(to: directory.appendingPathComponent("config.json"))
            let decoded = try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: config)
            let source = withRandomState(MLXRandom.RandomState(seed: 42)) { Qwen35(decoded) }
            source.update(parameters: source.parameters().mapValues { $0.asType(.bfloat16) })
            let original = QuantizedEmbedding(
                weight: source.languageModel.model.embedTokens.weight, groupSize: 32, bits: 4)
            try source.update(
                modules: ModuleChildren.unflattened([
                    ("language_model.model.embed_tokens", original)
                ]), verify: [])
            try save(
                arrays: Dictionary(uniqueKeysWithValues: source.parameters().flattened()),
                metadata: ["format": "mlx"],
                url: directory.appendingPathComponent("model.safetensors"))
            var sequences: [[Int]] = []
            for policy in [EmbeddingStoragePolicy.automatic, .resident] {
                let configuration = ModelConfiguration(
                    directory: directory, embeddingStorage: policy
                )
                .resolved(modelDirectory: directory, tokenizerDirectory: directory)
                let text = try await LLMModelFactory.shared._load(
                    configuration: configuration, tokenizerLoader: TestTokenizerLoader())
                let vision = try await VLMModelFactory.shared._load(
                    configuration: configuration, tokenizerLoader: TestTokenizerLoader())
                XCTAssertTrue(text.model is Qwen35Model)
                XCTAssertTrue(vision.model is Qwen35)
                for (index, context) in [text, vision].enumerated() {
                    XCTAssertEqual(context.configuration.embeddingStorage, policy)
                    let provider = try XCTUnwrap(context.model as? any InputOnlyEmbeddingsProviding)
                    XCTAssertEqual(provider.inputOnlyEmbeddings.count, tied ? 0 : 1)
                    let table = try XCTUnwrap(
                        context.model.namedModules().first {
                            $0.0 == "language_model.model.embed_tokens"
                        }?.1 as? Embedding)
                    XCTAssertEqual(
                        table is FileBackedQuantizedEmbedding, !tied && policy == .automatic)
                    assertIdentical(
                        table(MLXArray([0, 256, -1])), original(MLXArray([0, 256, -1])))
                    var iterator = try TokenIterator(
                        input: LMInput(tokens: MLXArray([1, 17, 24])), model: context.model,
                        parameters: GenerateParameters(maxTokens: 4, temperature: 0))
                    var tokens: [Int] = []
                    while let token = iterator.next() { tokens.append(token) }
                    XCTAssertEqual(tokens.count, 4)
                    if policy == .automatic {
                        sequences.append(tokens)
                    } else {
                        XCTAssertEqual(tokens, sequences[index])
                    }
                }
            }
        }
    }

    func testCompiledLookupUsesCurrentIndices() throws {
        let original = embedding()
        let model = Model(original)
        try prepareEmbeddingStorage(in: model, directory: FileManager.default.temporaryDirectory)
        let lookup = CompiledTrace<Model> { model, arguments in
            [model.embedding(arguments[0])]
        }
        for indices in [MLXArray([0, 256]), MLXArray([19, 19]), MLXArray([-1, 17])] {
            assertIdentical(lookup(model, indices), original(indices))
        }
    }

    func testSharedModulesStayShared() throws {
        let model = AliasedModel(embedding())
        let directory = FileManager.default.temporaryDirectory
        XCTAssertEqual(try prepareEmbeddingStorage(in: model, directory: directory), 1)
        XCTAssertTrue(model.first === model.second)
    }

    func testTrainableAndCustomEmbeddingsAreUnchanged() throws {
        let trainable = embedding()
        trainable.unfreeze()
        let model = Model(trainable)
        XCTAssertEqual(
            try prepareEmbeddingStorage(
                in: model, directory: FileManager.default.temporaryDirectory), 0)
        XCTAssertTrue(model.embedding === trainable)
        let custom = CustomEmbedding(
            weight: trainable.weight, scales: trainable.scales, biases: trainable.biases,
            groupSize: trainable.groupSize, bits: trainable.bits)
        custom.freeze()
        let customModel = Model(custom)
        XCTAssertEqual(
            try prepareEmbeddingStorage(
                in: customModel, directory: FileManager.default.temporaryDirectory), 0)
        XCTAssertTrue(customModel.embedding === custom)
    }

    func testFailureKeepsOriginalEmbedding() {
        let original = embedding()
        let model = Model(original)
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        XCTAssertThrowsError(try prepareEmbeddingStorage(in: model, directory: missing))
        XCTAssertTrue(model.embedding === original)
    }

    func testNoncontiguousArraysAndTemporaryFileCleanup() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = MLXArray((0 ..< 35).map(Float.init)).reshaped(5, 7).T
        let mapped = try fileBackedArray(source, directory: directory)
        assertIdentical(mapped, source)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
        assertIdentical(mapped + 1, source + 1)
    }

    func testFrozenFloatingTablesPreserveLookupAndParameters() throws {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let original = Embedding(
                weight: MLXArray((0 ..< 3 * 256).map { Float($0) / 256 }).reshaped(3, 256).asType(
                    dtype))
            original.freeze()
            let model = Model(original)
            XCTAssertEqual(
                try prepareEmbeddingStorage(
                    in: model, directory: FileManager.default.temporaryDirectory), 1)
            XCTAssertTrue(model.embedding is FileBackedEmbedding)
            for indices in [MLXArray(Int32(0)), MLXArray([0, 2, 2, -1]).reshaped(2, 2)] {
                assertIdentical(model.embedding(indices), original(indices))
            }
            assertIdentical(model.embedding.weight, original.weight)
            assertIdentical(model.embedding.weight + 1, original.weight + 1)
            assertIdentical(
                model.embedding.asLinear(original(MLXArray([0]))),
                original.asLinear(original(MLXArray([0]))))
        }
        let trainable = Embedding(embeddingCount: 257, dimensions: 256)
        let model = Model(trainable)
        XCTAssertEqual(
            try prepareEmbeddingStorage(
                in: model, directory: FileManager.default.temporaryDirectory), 0)
        XCTAssertTrue(model.embedding === trainable)
    }

    func testLoaderMapsFrozenFloatingTables() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        for dtype in [DType.float16, .bfloat16, .float32] {
            let source = Model(
                Embedding(
                    weight: MLXArray((0 ..< 257 * 256).map { Float($0) / 256 }).reshaped(257, 256)
                        .asType(dtype)))
            source.embedding.freeze()
            let url = directory.appendingPathComponent("model.safetensors")
            try saveCheckpoint(source, url: url)
            let model = Model(Embedding(embeddingCount: 257, dimensions: 256))
            model.embedding.freeze()
            let storage = try XCTUnwrap(
                CheckpointTensorStorage(model: model, policy: .automatic, urls: [url]))
            XCTAssertEqual(storage.mappings.count, 1)
            let checkpoint = try loadModelCheckpoint(urls: [url], mappings: storage.mappings)
            storage.record(checkpoint.weights)
            XCTAssertNotNil(
                try storage.array(matching: XCTUnwrap(checkpoint.weights["embedding.weight"])))
            try loadWeights(modelDirectory: directory, model: model)
            XCTAssertTrue(model.embedding is FileBackedEmbedding)
            assertIdentical(model.embedding.weight, source.embedding.weight)
            assertIdentical(
                model.embedding(MLXArray([0, 256, -1])), source.embedding(MLXArray([0, 256, -1])))
        }
    }

    func testOversizedBackingFailsBeforeReadingTheArray() throws {
        let array = MLXArray.zeros([131073, 16384], dtype: .uint8)
        let active = Memory.activeMemory
        XCTAssertThrowsError(
            try fileBackedArray(array, directory: FileManager.default.temporaryDirectory))
        XCTAssertEqual(Memory.activeMemory, active)
    }

    private func saveCheckpoint(_ model: Model, url: URL, offset: Int = 0) throws {
        try save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()), url: url)
        let original = try Data(contentsOf: url)
        let count = original.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            .littleEndian
        let padding = (offset - Int((count + 8) % 8) + 8) % 8
        var headerLength = (count + UInt64(padding)).littleEndian
        var rebased = withUnsafeBytes(of: &headerLength) { Data($0) }
        rebased.append(original.subdata(in: 8 ..< (8 + Int(count))))
        rebased.append(Data(repeating: 32, count: padding))
        rebased.append(original.subdata(in: (8 + Int(count)) ..< original.count))
        try rebased.write(to: url)
    }

    func testCheckpointMappingSelectsAlignedStorageAndPreservesAllBits() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("model.safetensors")
        for dtype in [DType.float16, .bfloat16, .float32] {
            for bits in [2, 3, 4, 5, 6, 8] {
                for offset in 0 ..< 8 {
                    try autoreleasepool {
                        let original = Model(embedding(dtype: dtype, bits: bits))
                        try saveCheckpoint(original, url: url, offset: offset)
                        let model = Model(embedding(dtype: dtype, bits: bits))
                        let storage = try XCTUnwrap(
                            CheckpointTensorStorage(
                                model: model, policy: .automatic, urls: [url]))
                        var checkpoint = try loadModelCheckpoint(
                            urls: [url], mappings: storage.mappings)
                        storage.record(checkpoint.weights)
                        let loadedWeight = try XCTUnwrap(checkpoint.weights["embedding.weight"])
                        let retained = try storage.array(matching: loadedWeight)
                        XCTAssertEqual(
                            retained != nil, offset.isMultiple(of: MemoryLayout<UInt64>.alignment))
                        let loadedAddress = loadedWeight.asData(access: .noCopy).data
                            .withUnsafeBytes {
                                $0.baseAddress.map(UInt.init(bitPattern:))
                            }
                        if let retained {
                            let retainedAddress = retained.asData(access: .noCopy).data
                                .withUnsafeBytes {
                                    $0.baseAddress.map(UInt.init(bitPattern:))
                                }
                            XCTAssertEqual(loadedAddress, retainedAddress)
                        }
                        try applyCheckpoint(checkpoint, to: model)
                        checkpoint.weights.removeAll()
                        XCTAssertEqual(
                            try prepareEmbeddingStorage(
                                in: model, directory: directory, checkpoint: storage), 1)
                        XCTAssertTrue(model.embedding is FileBackedQuantizedEmbedding)
                        for indices in [
                            MLXArray(Int32(0)), MLXArray([0, 256, 19, 19, -1, -257]),
                            MLXArray([0, 256, 19, 19, 17, 4]).reshaped(2, 3),
                        ] {
                            assertIdentical(model.embedding(indices), original.embedding(indices))
                        }
                        let actual = Dictionary(
                            uniqueKeysWithValues: model.embedding.parameters().flattened())
                        for (name, expected) in original.embedding.parameters().flattened() {
                            let value = try XCTUnwrap(actual[name])
                            assertIdentical(value, expected)
                            assertIdentical(value + 1, expected + 1)
                            assertIdentical(
                                value.view(dtype: .uint64) + MLXArray(UInt64(1)),
                                expected.view(dtype: .uint64) + MLXArray(UInt64(1)))
                            let remainder = value.asData(access: .noCopy).data.withUnsafeBytes {
                                $0.baseAddress.map { Int(UInt(bitPattern: $0) % 8) }
                            }
                            XCTAssertEqual(remainder, 0)
                        }
                        let input = original.embedding(MLXArray([0, 1]))
                        assertIdentical(
                            model.embedding.asLinear(input), original.embedding.asLinear(input))
                        let lookup = CompiledTrace<Model> { model, arguments in
                            [model.embedding(arguments[0])]
                        }
                        for indices in [MLXArray([0, 256]), MLXArray([19, -1])] {
                            assertIdentical(lookup(model, indices), original.embedding(indices))
                        }
                    }
                }
            }
        }
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(atPath: directory.path),
            ["model.safetensors"])
    }

    func testEarlyMappingRequiresKnownAffineQuantization() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("model.safetensors")
        let source = Model(embedding())
        try saveCheckpoint(source, url: url, offset: 0)
        let model = Model(Embedding(embeddingCount: 257, dimensions: 256))
        guard
            let unknown = CheckpointTensorStorage(
                model: model, policy: .automatic, urls: [url])
        else { throw XCTSkip("File-backed embeddings require a Metal device") }
        XCTAssertTrue(unknown.mappings.isEmpty)
        for mode in ["affine", "mxfp4"] {
            let quantization = try JSONDecoder().decode(
                BaseConfiguration.Quantization.self,
                from: Data("{\"group_size\":64,\"bits\":4,\"mode\":\"\(mode)\"}".utf8))
            let policy = BaseConfiguration.PerLayerQuantization(
                quantization: quantization, perLayerQuantization: [:])
            let model = Model(Embedding(embeddingCount: 257, dimensions: 256))
            let storage = try XCTUnwrap(
                CheckpointTensorStorage(
                    model: model, policy: .automatic, urls: [url], quantization: policy))
            XCTAssertEqual(storage.mappings.count, mode == "affine" ? 3 : 0)
            if mode == "affine" {
                var checkpoint = try loadModelCheckpoint(urls: [url], mappings: storage.mappings)
                storage.record(checkpoint.weights)
                checkpoint.perLayerQuantization = policy
                try applyCheckpoint(checkpoint, to: model)
                XCTAssertEqual(
                    try prepareEmbeddingStorage(
                        in: model, directory: directory.appendingPathComponent("missing"),
                        checkpoint: storage), 1)
                assertIdentical(
                    model.embedding(MLXArray([0, 256])), source.embedding(MLXArray([0, 256])))
            }
        }
    }

    func testCheckpointStorageMatchesRenamesAndRejectsChangedSources() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("model.safetensors")
        let model = Model(embedding())
        try saveCheckpoint(model, url: url, offset: 0)
        guard
            let storage = CheckpointTensorStorage(
                model: model, policy: .automatic, urls: [url])
        else { throw XCTSkip("File-backed embeddings require a Metal device") }
        let checkpoint = try loadModelCheckpoint(urls: [url])
        storage.record(checkpoint.weights)
        let renamed = try checkpoint.mapNames {
            $0.replacingOccurrences(of: "embedding", with: "renamed")
        }
        let source = try XCTUnwrap(renamed.weights["renamed.weight"])
        let mapped = try XCTUnwrap(try storage.array(matching: source))
        assertIdentical(mapped, source)
        XCTAssertNil(try storage.array(matching: source + 1))
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data([0]))
        try handle.close()
        XCTAssertNil(try storage.array(matching: source))
    }

    func testCheckpointMappingSurvivesClosingAndUnlinkingSource() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("model.safetensors")
        let model = Model(embedding())
        try saveCheckpoint(model, url: url, offset: 0)
        var storage = CheckpointTensorStorage(model: model, policy: .automatic, urls: [url])
        guard storage != nil else {
            throw XCTSkip("File-backed embeddings require a Metal device")
        }
        let checkpoint = try loadModelCheckpoint(urls: [url])
        storage?.record(checkpoint.weights)
        let source = try XCTUnwrap(checkpoint.weights["embedding.weight"])
        let mapped = try XCTUnwrap(try storage?.array(matching: source))
        storage = nil
        try FileManager.default.removeItem(at: url)
        assertIdentical(mapped, source)
        assertIdentical(mapped + 1, source + 1)
    }
}
#endif
