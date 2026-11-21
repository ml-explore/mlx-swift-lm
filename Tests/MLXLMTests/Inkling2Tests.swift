// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

// The kernels are checked against plain MLX computations of the same math, and the model
// against itself: a full-sequence forward must match token-by-token decoding through the
// caches, which exercises the sliding-window ring, the per-pass caches of the loop and the
// convolution states.

// MARK: - Configuration

@Suite("Inkling2 configuration")
struct Inkling2ConfigurationTests {

    /// `config.json` of `mlx-community/Inkling-2-Mini-4bit`, trimmed to the fields the port reads.
    static let miniJSON = """
        {
          "model_type": "inkling2_loop",
          "text_config": {
            "hidden_size": 2304, "num_hidden_layers": 24, "vocab_size": 201024,
            "unpadded_vocab_size": 200058, "num_attention_heads": 24, "num_key_value_heads": 8,
            "head_dim": 128, "swa_head_dim": 128, "swa_num_attention_heads": 24,
            "swa_num_key_value_heads": 8, "sliding_window_size": 1024,
            "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
            "d_rel": 16, "rel_extent": 1024, "log_scaling_n_floor": 8192,
            "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06,
            "qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0,
            "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 6, "n_routed_experts": 128,
            "num_experts_per_tok": 8, "intermediate_size": 1152, "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0, "route_scale": 8.0, "final_logit_softcapping": 30.0,
            "loop_config": {
              "core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64,
              "is_varloop": false
            }
          },
          "vision_config": {
            "native_patch_size": 20, "adapter_merge_size": 2, "projection_hidden_dim": 4096,
            "hidden_size": 1024, "num_hidden_layers": 23, "num_attention_heads": 16
          }
        }
        """

    @Test("The loop core runs once per pass between the prelude and the coda")
    func layerVisits() throws {
        let config = try JSONDecoder().decode(
            Inkling2Configuration.self, from: Data(Self.miniJSON.utf8))
        let text = config.textConfiguration
        let visits = text.layerVisits
        #expect(visits.count == 48)
        #expect(Array(visits.prefix(7)) == [0, 1, 2, 3, 4, 5, 6])
        #expect(Array(visits[18 ..< 20]) == [6, 7])
        #expect(Array(visits.suffix(6)) == Array(18 ..< 24))
        #expect(config.imageTokenId == 200054)
        #expect(config.visionConfiguration?.blockSize == 40)
        #expect(text.isSliding(0) && !text.isSliding(5) && !text.isSliding(23))
    }

    @Test("The processor configuration reads the vision settings of config.json")
    func processorConfiguration() throws {
        let config = try JSONDecoder().decode(
            Inkling2ProcessorConfiguration.self, from: Data(Self.miniJSON.utf8))
        #expect(config.patchSize == 20)
        #expect(config.mergeSize == 2)
        #expect(config.imageTokenId == 200054)
    }

    @Test("Expert formats follow the tensor shapes")
    func expertFormats() {
        let experts = 4
        let nvfp4: [String: MLXArray] = [
            "e.weight": MLXArray.zeros([experts, 8, 64 / 8], dtype: .uint32),
            "e.scales": MLXArray.zeros([experts, 8, 64 / 16], dtype: .uint8),
            "e.global_scale": MLXArray.ones([experts]),
        ]
        #expect(
            Inkling2ExpertFormat(path: "e", weights: nvfp4, inputDims: 64)
                == Inkling2ExpertFormat(mode: .nvfp4, bits: 4, groupSize: 16, hasGlobalScale: true))

        let mxfp8: [String: MLXArray] = [
            "e.weight": MLXArray.zeros([experts, 8, 64 / 4], dtype: .uint32),
            "e.scales": MLXArray.zeros([experts, 8, 64 / 32], dtype: .uint8),
        ]
        #expect(
            Inkling2ExpertFormat(path: "e", weights: mxfp8, inputDims: 64)
                == Inkling2ExpertFormat(mode: .mxfp8, bits: 8, groupSize: 32, hasGlobalScale: false)
        )
        #expect(Inkling2ExpertFormat(path: "missing", weights: mxfp8, inputDims: 64) == nil)
    }
}

// MARK: - Kernels

/// Attention with a relative bias and a sink, materialized in float32.
///
/// - Parameter distance: `[L, S]`, each query's distance to each key.
private func referenceRelativeAttention(
    q: MLXArray, k: MLXArray, v: MLXArray, rel: MLXArray, sinks: MLXArray, scale: Float,
    window: Int, distance: MLXArray
) -> MLXArray {
    let (b, hq, l, d) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
    let (hkv, s) = (k.dim(1), k.dim(2))
    let extent = rel.dim(-1)
    let group = hq / hkv
    let kk = repeated(k.asType(.float32), count: group, axis: 1)
    let vv = repeated(v.asType(.float32), count: group, axis: 1)
    var logits = matmul(q.asType(.float32), kk.swappedAxes(-1, -2)) * scale
    let index = broadcast(clip(distance, min: 0, max: extent - 1), to: [b, hq, l, s])
    var bias = takeAlong(rel.asType(.float32).transposed(0, 2, 1, 3), index, axis: -1)
    bias = which(distance .< extent, bias, MLXArray(Float(0)))
    var visible = distance .>= 0
    if window > 0 {
        visible = logicalAnd(visible, distance .< window)
    }
    logits = which(visible, logits + bias, MLXArray(-Float.infinity))
    let sinkLogits = sinks.asType(.float32).transposed(0, 2, 1)[.ellipsis, .newAxis]
    let weights = softmax(concatenated([logits, sinkLogits], axis: -1), axis: -1)[
        .ellipsis, ..<s]
    return matmul(weights, vv).transposed(0, 2, 1, 3).reshaped(b, l, hq, d)
}

private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

@Suite("Inkling2 kernels")
struct Inkling2KernelTests {

    @Test("Depth attention matches the softmax over the sources", arguments: [1, 2, 4])
    func depthAttention(queriesPerGroup: Int) {
        MLXRandom.seed(1)
        let blocks = MLXRandom.normal([3, 2, 5, 256])
        let partial = MLXRandom.normal([2, 5, 256])
        let queries = MLXRandom.normal([4, 256])
        let scale = 1 / Float(256).squareRoot()
        let out = inkling2DepthAttention(
            blocks: blocks, partial: partial, queries: queries, scale: scale,
            queriesPerGroup: queriesPerGroup)

        let sources = movedAxis(
            concatenated([blocks, partial.expandedDimensions(axis: 0)]), source: 0,
            destination: -2)
        let weights = softmax(matmul(sources, queries.transposed()) * scale, axis: -2)
        let expected = matmul(weights.swappedAxes(-1, -2), sources)
        #expect(out.shape == [2, 5, 4, 256])
        // MLX runs float32 matmuls in TF32 on GPUs with neural accelerators.
        #expect(maxDifference(out, expected) < 1e-2)
    }

    @Test(
        "Prefill attention matches the reference",
        arguments: [(window: 0, cached: 0), (window: 16, cached: 0), (window: 16, cached: 23)])
    func prefillAttention(window: Int, cached: Int) {
        MLXRandom.seed(2)
        let (b, hq, hkv, l, d, extent) = (1, 4, 2, 40, 64, 24)
        let s = cached + l
        let q = (MLXRandom.normal([b, hq, l, d]) * 0.3).asType(.bfloat16)
        let k = (MLXRandom.normal([b, hkv, s, d]) * 0.3).asType(.bfloat16)
        let v = MLXRandom.normal([b, hkv, s, d]).asType(.bfloat16)
        let rel = MLXRandom.normal([b, l, hq, extent]).asType(.bfloat16)
        let sinks = MLXRandom.normal([b, l, hq]).asType(.bfloat16)

        let out = inkling2RelativeAttention(
            q: q, k: k, v: v, rel: rel, sinks: sinks, scale: 0.125, window: window,
            keyStart: MLXArray([Int32(0)]))
        let queries = MLXArray.arange(l, dtype: .int32) + Int32(s - l)
        let distance = queries[0..., .newAxis] - MLXArray.arange(s, dtype: .int32)[.newAxis]
        let expected = referenceRelativeAttention(
            q: q, k: k, v: v, rel: rel, sinks: sinks, scale: 0.125, window: window,
            distance: distance)
        #expect(maxDifference(out, expected) < 2e-2)
    }

    @Test(
        "Decode attention matches the reference",
        arguments: [
            (keys: 37, ring: false, window: 0),
            (keys: 37, ring: true, window: 0),
            (keys: 64, ring: true, window: 48),
            (keys: 3000, ring: false, window: 0),
        ])
    func decodeAttention(keys s: Int, ring: Bool, window: Int) {
        MLXRandom.seed(3)
        let (b, hq, hkv, d, extent) = (2, 6, 2, 64, 40)
        let q = (MLXRandom.normal([b, hq, 1, d]) * 0.3).asType(.bfloat16)
        // The cache view is a slice of a longer buffer, as KVCacheSimple returns it.
        let kBuffer = (MLXRandom.normal([b, hkv, s + 5, d]) * 0.3).asType(.bfloat16)
        let vBuffer = MLXRandom.normal([b, hkv, s + 5, d]).asType(.bfloat16)
        let k = kBuffer[0..., 0..., ..<s, 0...]
        let v = vBuffer[0..., 0..., ..<s, 0...]
        let relative = (MLXRandom.normal([b, hq, 8]) * 0.25).asType(.bfloat16)
        let table = MLXRandom.normal([8, extent]).asType(.bfloat16)
        let rel = matmul(relative.asType(.float32), table.asType(.float32)).expandedDimensions(
            axis: 1)
        let sinks = MLXRandom.normal([b, 1, hq]).asType(.bfloat16)

        // A ring holds temporal position t at slot (ringIndex + t) % S.
        let ringIndex = ring ? 11 : nil
        var ringK = k
        var ringV = v
        if let ringIndex {
            let order = MLXArray((0 ..< s).map { Int32(($0 - ringIndex + s) % s) })
            ringK = k.take(order, axis: 2)
            ringV = v.take(order, axis: 2)
        }
        let gate = (MLXRandom.normal([b, 1, hq * d]) * 0.5).asType(.bfloat16)
        let out = inkling2DecodeAttention(
            q: q.squeezed(axis: 2), k: ringK, v: ringV, relative: relative, table: table,
            sinks: sinks, gate: gate, scale: 0.125, ringIndex: ringIndex, window: window)

        let distance = (MLXArray(Int32(s - 1)) - MLXArray.arange(s, dtype: .int32))[.newAxis]
        let expected = referenceRelativeAttention(
            q: q, k: k, v: v, rel: rel, sinks: sinks, scale: 0.125, window: window,
            distance: distance)
        let gated =
            expected.asType(.float32).reshaped(b, hq * d)
            * sigmoid(gate.asType(.float32)).reshaped(b, hq * d)
        #expect(out.shape == [b, hq * d])
        #expect(maxDifference(out, gated) < 1e-2)
    }

    @Test("Expert routing picks the top bias-corrected scores")
    func expertRouting() {
        MLXRandom.seed(4)
        let logits = MLXRandom.normal([3, 2, 128]) * 3
        let bias = MLXRandom.normal([128]) * 0.1
        let (indices, weights) = inkling2ExpertRouting(
            logits: logits, bias: bias, topK: 8, scoreEps: 1e-20, routeScale: 8)

        let scores = sigmoid(logits)
        let order = argSort(-(scores + bias), axis: -1)[.ellipsis, ..<8]
        let selected = takeAlong(scores, order, axis: -1)
        let expected = selected / selected.sum(axis: -1, keepDims: true) * 8
        #expect(
            indices.asType(.int32).asArray(Int32.self) == order.asType(.int32).asArray(Int32.self))
        #expect(maxDifference(weights, expected) < 1e-5)
    }
}

// MARK: - Model

@Suite("Inkling2 model")
struct Inkling2ModelTests {

    /// Six layers in blocks of two; layers 2 and 3 loop twice. Layers 3 and 5 attend to all
    /// positions, the others to a window of 8.
    static let tinyJSON = """
        {
          "model_type": "inkling2_loop",
          "text_config": {
            "hidden_size": 64, "num_hidden_layers": 6, "vocab_size": 50,
            "unpadded_vocab_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 32, "sliding_window_size": 8, "local_layer_ids": [0, 1, 2, 4],
            "d_rel": 8, "rel_extent": 16, "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-06, "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0,
            "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 2, "n_routed_experts": 4,
            "num_experts_per_tok": 2, "intermediate_size": 32, "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0, "route_scale": 2.0, "final_logit_softcapping": 30.0,
            "loop_config": {
              "core_start": 2, "core_end": 4, "num_passes": 2, "gate_rank": 8,
              "is_varloop": false
            }
          }
        }
        """

    /// A tiny model with random weights and NVFP4 experts.
    static func makeModel() throws -> Inkling2 {
        let config = try JSONDecoder().decode(
            Inkling2Configuration.self, from: Data(tinyJSON.utf8))
        let model = Inkling2(config)
        MLXRandom.seed(5)
        var parameters = [String: MLXArray]()
        for (key, value) in model.parameters().flattened() where value.dtype == .float32 {
            parameters[key] = MLXRandom.normal(value.shape) * 0.3
        }
        for (key, value) in model.parameters().flattened() where key.hasSuffix(".global_scale") {
            parameters[key] = MLXArray.ones(value.shape)
        }
        for (key, value) in model.parameters().flattened()
        where key.hasSuffix("_proj.weight") && value.dtype == .uint32 {
            let prefix = String(key.dropLast(".weight".count))
            let (rows, packed) = (value.dim(1), value.dim(2))
            let dense = MLXRandom.normal([value.dim(0), rows, packed * 8]) * 0.3
            let (weight, scales, _) = MLX.quantized(
                dense, groupSize: 16, bits: 4, mode: .nvfp4)
            parameters[key] = weight
            parameters["\(prefix).scales"] = scales
        }
        try model.update(
            parameters: ModuleParameters.unflattened(parameters), verify: .noUnusedKeys)
        eval(model)
        return model
    }

    @Test("Rotating cache capacity cannot preserve a prefix")
    func rotatingCacheCapacity() throws {
        let model = try Self.makeModel()
        let capacity = try KVCacheConfiguration.Capacity(maxTokens: 16, preservedPrefixTokens: 4)
        let parameters = GenerateParameters(kvCache: .init(capacity: capacity))
        #expect(throws: VLMError.self) { try model.newCache(parameters: parameters) }
        let tailOnly = try KVCacheConfiguration.Capacity(maxTokens: 16, preservedPrefixTokens: 0)
        let caches = try model.newCache(parameters: .init(kvCache: .init(capacity: tailOnly)))
        #expect(!caches.isEmpty)
    }

    @Test("Decoding through the caches matches the full-sequence forward")
    func incrementalDecoding() throws {
        let model = try Self.makeModel()
        let tokens = MLXArray((0 ..< 21).map { Int32(($0 * 7 + 3) % 48) }).reshaped(1, -1)
        let full = model(tokens, cache: nil)
        #expect(full.shape == [1, 21, 48])

        let cache = try model.newCache(parameters: nil)
        #expect(cache.count == 8)
        // Prefill 5 tokens, then decode the rest one at a time past the window of 8.
        let prefill = model(tokens[0..., ..<5], cache: cache)
        // The tolerance allows for TF32 float32 matmuls; a wrong cache is off by far more.
        #expect(maxDifference(prefill, full[0..., ..<5]) < 1e-2)
        for position in 5 ..< 21 {
            let step = model(tokens[0..., position ..< (position + 1)], cache: cache)
            #expect(
                maxDifference(step, full[0..., position ..< (position + 1)]) < 1e-2,
                "position \(position)")
        }
    }

    @Test("Chunked prefill matches one prefill")
    func chunkedPrefill() throws {
        let model = try Self.makeModel()
        let tokens = MLXArray((0 ..< 30).map { Int32(($0 * 5 + 1) % 48) }).reshaped(1, -1)
        let whole = try model.prepare(
            LMInput(tokens: tokens), cache: model.newCache(parameters: nil), state: nil,
            prefill: .init())
        let chunked = try model.prepare(
            LMInput(tokens: tokens), cache: model.newCache(parameters: nil), state: nil,
            prefill: .init(stepSize: 7))
        guard case .logits(let a) = whole, case .logits(let b) = chunked else {
            Issue.record("prepare did not return logits")
            return
        }
        #expect(a.logits.shape == [1, 1, 48])
        #expect(maxDifference(a.logits, b.logits) < 1e-2)
    }
}

// MARK: - Stream protocol

private final class InklingTestTokenizer: Tokenizer, @unchecked Sendable {
    static let controls = [
        "<|message_model|>": 200_001, "<|content_text|>": 200_004,
        "<|content_model_end_sampling|>": 200_006, "<|content_thinking|>": 200_008,
        "<|end_message|>": 200_010, "<|content_invoke_tool_json|>": 200_049,
    ]
    private var nextID = 1
    private var textToID = InklingTestTokenizer.controls
    private var idToText = Dictionary(
        uniqueKeysWithValues: InklingTestTokenizer.controls.map { ($1, $0) })

    func id(_ text: String) -> Int {
        if let id = textToID[text] { return id }
        let id = nextID
        nextID += 1
        textToID[text] = id
        idToText[id] = text
        return id
    }

    func ids(_ pieces: String...) -> [Int] { pieces.map(id) }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [id(text)] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { idToText[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { textToID[token] }
    func convertIdToToken(_ id: Int) -> String? { idToText[id] }
    var bosToken: String? { nil }
    var eosToken: String? { "<|content_model_end_sampling|>" }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite("Inkling2 stream protocol")
struct Inkling2StreamProtocolTests {

    private func events(
        _ tokenizer: InklingTestTokenizer, _ tokens: [Int],
        tools: [[String: any Sendable]]? = nil
    ) throws -> [TokenStreamEvent] {
        var decoder = try #require(
            ToolCallFormat.inkling.makeProtocolTokenStreamDecoder(
                tokenizer: tokenizer, tools: tools, stopStrings: []))
        var events = [TokenStreamEvent]()
        for token in tokens {
            _ = decoder.push(token) {
                events.append($0)
                return true
            }
        }
        _ = decoder.finish {
            events.append($0)
            return true
        }
        return events
    }

    private func text(_ events: [TokenStreamEvent]) -> (reasoning: String, response: String) {
        (
            events.compactMap { if case .reasoning(let t) = $0 { t } else { nil } }.joined(),
            events.compactMap { if case .response(let t) = $0 { t } else { nil } }.joined()
        )
    }

    private static let weatherTool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "get_weather",
            "parameters": [
                "type": "object",
                "properties": ["city": ["type": "string"] as [String: any Sendable]],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    @Test("Thinking and text frames become reasoning and response without markers")
    func reasoningAndResponse() throws {
        let tokenizer = InklingTestTokenizer()
        let tokens = tokenizer.ids(
            "<|message_model|>", "<|content_thinking|>", "Plan", " it.", "<|end_message|>",
            "<|message_model|>", "<|content_text|>", "Hello", " there.", "<|end_message|>",
            "<|content_model_end_sampling|>")
        let events = try events(tokenizer, tokens)
        let (reasoning, response) = text(events)
        #expect(reasoning == "Plan it.")
        #expect(response == "Hello there.")
        #expect(!events.contains { if case .protocolError = $0 { true } else { false } })
    }

    @Test("A tool frame becomes a tool call")
    func toolCall() throws {
        let tokenizer = InklingTestTokenizer()
        let tokens = tokenizer.ids(
            "<|message_model|>", "get_weather", "<|content_invoke_tool_json|>",
            #"{"name":"get_weather","args":{"city":"Rome"}}"#, "<|end_message|>",
            "<|content_model_end_sampling|>")
        let events = try events(tokenizer, tokens, tools: [Self.weatherTool])
        let calls = events.compactMap { if case .toolCall(let c) = $0 { c } else { nil } }
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
        #expect(calls.first?.function.arguments["city"] == .string("Rome"))
        #expect(text(events).response.isEmpty)
    }

    @Test("Undeclared, malformed and unfinished tool frames are rejected")
    func rejectedToolCalls() throws {
        let tokenizer = InklingTestTokenizer()
        func rejection(_ tokens: [Int]) throws -> RejectedToolCall.Reason? {
            try events(tokenizer, tokens, tools: [Self.weatherTool]).compactMap {
                if case .rejectedToolCall(let r) = $0 { r.reason } else { nil }
            }.first
        }
        #expect(
            try rejection(
                tokenizer.ids(
                    "<|message_model|>", "rm", "<|content_invoke_tool_json|>",
                    #"{"name":"rm","args":{}}"#, "<|end_message|>")) == .undeclaredTool)
        #expect(
            try rejection(
                tokenizer.ids(
                    "<|message_model|>", "get_weather", "<|content_invoke_tool_json|>",
                    #"{"name": "#, "<|end_message|>")) == .malformedSyntax)
        #expect(
            try rejection(
                tokenizer.ids(
                    "<|message_model|>", "get_weather", "<|content_invoke_tool_json|>",
                    #"{"name":"get_weather","#)) == .incompleteOutput)
    }

    @Test("The text parser reads a tool frame")
    func textParser() {
        let call = InklingToolCallParser().parse(
            content: #"<|content_invoke_tool_json|>{"name":"f","args":{"n":2}}<|end_message|>"#,
            tools: nil)
        #expect(call?.function.name == "f")
        #expect(call?.function.arguments["n"] == .int(2))
    }
}

// MARK: - Processor

@Suite("Inkling2 processor")
struct Inkling2ProcessorTests {

    @Test("Images are padded to whole blocks and patchified block by block")
    func patches() throws {
        let processor = Inkling2Processor(
            Inkling2ProcessorConfiguration(), tokenizer: InklingTestTokenizer())
        let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(
            to: CGRect(x: 0, y: 0, width: 50, height: 30))
        let (patches, grid) = try processor.preprocess(image: image, processing: nil)
        // ceil(30 / 40) x ceil(50 / 40) blocks of 2 x 2 patches of 20 x 20 x 3.
        #expect(grid.h == 2 && grid.w == 4)
        #expect(patches.shape == [8, 1200])
        let padding = (Float(-1) / 255 - 0.5) / 0.5
        // The last patch of the second block is right of the image.
        #expect(abs(patches[7].min().item(Float.self) - padding) < 1e-6)
        // The first pixel is red.
        #expect(abs(patches[0, 0].item(Float.self) - 1) < 1e-3)
        #expect(abs(patches[0, 1].item(Float.self) + 1) < 1e-3)
    }
}
