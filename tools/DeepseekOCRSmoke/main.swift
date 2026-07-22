import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers

@main
enum DeepseekOCRSmoke {
    static func main() async throws {
        let env = ProcessInfo.processInfo.environment
        let modelID = env["MODEL_ID"] ?? "majentik/Unlimited-OCR-MLX-6bit"
        let prompt = env["PROMPT"] ?? "document parsing."
        let maxTokens = Int(env["MAX_TOKENS"] ?? "2048") ?? 2048
        let mode = env["MODE"] ?? "gundam"

        guard let imagePath = env["IMAGE"], !imagePath.isEmpty else {
            throw SmokeError.missingEnvironment("IMAGE")
        }
        guard let outputPath = env["OUT_FILE"], !outputPath.isEmpty else {
            throw SmokeError.missingEnvironment("OUT_FILE")
        }

        let imageURL = URL(fileURLWithPath: imagePath).standardizedFileURL
        let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil)

        print("Model: \(modelID)")
        print("Image: \(imageURL.path)")
        print("Prompt: \(prompt)")

        let container = try await VLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: modelID)
        ) { progress in
            guard progress.totalUnitCount > 0 else { return }
            let percent = Int(
                (Double(progress.completedUnitCount) / Double(progress.totalUnitCount)) * 100)
            print("Download progress: \(percent)%")
        }

        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: maxTokens, temperature: 0.0),
            additionalContext: [DeepseekOCRProcessor.modeContextKey: mode])

        let response = try await session.respond(
            to: prompt,
            image: .url(imageURL))

        try response.write(to: outputURL, atomically: true, encoding: .utf8)
        print("Wrote \(outputURL.path) chars: \(response.count)")
    }
}

private enum SmokeError: LocalizedError {
    case missingEnvironment(String)

    var errorDescription: String? {
        switch self {
        case .missingEnvironment(let name):
            return "Missing required environment variable: \(name)"
        }
    }
}
