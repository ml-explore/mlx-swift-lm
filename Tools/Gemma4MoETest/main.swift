import Foundation
import MLX
import BenchmarkHelpers
import MLXLLM
import MLXLMCommon

let args = CommandLine.arguments
guard args.count >= 3 else { print("Usage: TQBench <test> <model_path> [scheme]"); exit(1) }
let test = args[1]; let modelPath = args[2]; let scheme = args.count > 3 ? args[3] : "none"

let container = try await loadModelContainer(from: URL(fileURLWithPath: modelPath), using: NoOpTokenizerLoader())

if test == "ppl" {
    let data = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: "/tmp/ppl_test_data.json"))) as! [String: Any]
    let ids = (data["ids"] as! [NSNumber]).map { Int32($0.intValue) }
    try await container.perform { context in
        let model = context.model; var cache = model.newCache(parameters: nil)
        let halfLen = ids.count / 2
        let prefillOut = model(MLXArray(Array(ids[0..<halfLen]))[.newAxis, .ellipsis], cache: cache)
        eval(prefillOut, cache)
        if scheme != "none" { maybeQuantizeKVCache(cache: &cache, kvBits: nil, quantizedKVStart: 0, kvScheme: scheme); eval(cache) }
        var totalLoss: Float = 0; var count = 0
        for i in halfLen..<(ids.count - 1) {
            let logits = model(MLXArray([ids[i]])[.newAxis, .ellipsis], cache: cache); eval(logits, cache)
            let posLogits = logits[0, -1]; let maxL = posLogits.max(); eval(maxL)
            let shifted = posLogits - maxL; let lse = log(exp(shifted).sum())
            let logProb = shifted[Int(ids[i + 1])] - lse; eval(logProb)
            totalLoss -= logProb.item(Float.self); count += 1
        }
        print("PPL:\(String(format: "%.2f", exp(totalLoss / Float(count))))")
    }
} else if test == "niah" {
    let data = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: "/tmp/niah_test_data.json"))) as! [String: Any]
    let ids = (data["ids"] as! [NSNumber]).map { Int32($0.intValue) }; let eosId = Int32((data["eos"] as! NSNumber).intValue)
    try await container.perform { context in
        let model = context.model; var cache = model.newCache(parameters: nil)
        let output = model(MLXArray(ids)[.newAxis, .ellipsis], cache: cache); eval(output, cache)
        if scheme != "none" { maybeQuantizeKVCache(cache: &cache, kvBits: nil, quantizedKVStart: 0, kvScheme: scheme); eval(cache) }
        let first = argMax(output[0, -1], axis: -1); eval(first)
        var generated: [Int32] = [first.item(Int32.self)]; var cur = MLXArray([generated[0]])[.newAxis, .ellipsis]
        for _ in 0..<30 { let out = model(cur, cache: cache); eval(out); let next = argMax(out[0, -1], axis: -1); eval(next); generated.append(next.item(Int32.self)); cur = MLXArray([generated.last!])[.newAxis, .ellipsis]; if generated.last! == eosId { break } }
        print("NIAH:\(generated.map{String($0)}.joined(separator: ","))")
    }
} else if test == "mem" {
    try await container.perform { context in
        let model = context.model; var cache = model.newCache(parameters: nil)
        let ctx = Int(args.count > 4 ? args[4] : "512")!
        let tokens = MLXArray(Array(repeating: Int32(1), count: ctx))[.newAxis, .ellipsis]
        let output = model(tokens, cache: cache); eval(output, cache)
        if scheme != "none" { maybeQuantizeKVCache(cache: &cache, kvBits: nil, quantizedKVStart: 0, kvScheme: scheme); eval(cache) }
        // Decode 5 tokens to trigger compression
        var cur = MLXArray([Int32(1)])[.newAxis, .ellipsis]
        for _ in 0..<5 { let out = model(cur, cache: cache); eval(out, cache); cur = MLXArray([Int32(1)])[.newAxis, .ellipsis] }
        var turboMem = 0; var turboCount = 0
        for c in cache { if let tc = c as? TurboQuantKVCache { turboMem += tc.memoryBytes; turboCount += 1 } }
        print("MEM:\(turboMem/1024)\t\(turboCount)\t\(cache.count)")
    }
}
