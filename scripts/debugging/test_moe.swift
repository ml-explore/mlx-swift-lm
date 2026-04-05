import Foundation
import MLX
import MLXLLM
import MLXLMCommon

@main
struct TestMoE {
    static func main() async throws {
        let config = ModelConfiguration(id: "mlx-community/gemma-4-26b-a4b-it-4bit")
        let modelContainer = try await LLMModelFactory.shared.loadContainer(from: HubClient.default, using: TokenizersLoader(), configuration: config)
        let context = try await modelContainer.context()
        let prompt = "What is 2+2?"
        let tokens = context.tokenizer.encode(text: prompt)
        print("Prompt tokens:", tokens)

        let input = MLXArray(tokens).reshaped(1, -1)
        MLX.eval(input)

        if let gemma4 = context.model as? Gemma4Model {
            print("Successfully cast to Gemma4Model!")
            let logits = gemma4(input, cache: nil)
            MLX.eval(logits)
            let logitsF32 = logits.asType(.float32)
            print("Logits shape:", logitsF32.shape)
            print("Max index:", MLX.argMax(logitsF32, axis: -1).item(Int.self))
            print("Max prob:", MLX.max(logitsF32).item(Float.self))
        }
    }
}
