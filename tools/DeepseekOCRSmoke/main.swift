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
        let modelDir = env["MODEL_DIR"]
        let prompt = env["PROMPT"] ?? "document parsing. "
        let maxTokens = Int(env["MAX_TOKENS"] ?? "2048") ?? 2048
        let modeRaw = env["MODE"] ?? DeepseekOCRProcessor.Mode.gundam.rawValue
        guard let mode = DeepseekOCRProcessor.Mode(rawValue: modeRaw) else {
            throw SmokeError.invalidMode(modeRaw)
        }

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
        if let modelDir, !modelDir.isEmpty {
            print("Model dir: \(modelDir)")
        }
        print("Image: \(imageURL.path)")
        print("Prompt: \(prompt)")
        print("Mode: \(mode.rawValue) (gundam=1024+tiles, base=640 single view)")

        let tokenizerLoader = #huggingFaceTokenizerLoader()
        let container: ModelContainer
        if let modelDir, !modelDir.isEmpty {
            // Local overlay (e.g. shard-name alias) — keep MODEL_ID for logging only.
            let directory = URL(fileURLWithPath: modelDir, isDirectory: true)
                .standardizedFileURL
            container = try await VLMModelFactory.shared.loadContainer(
                from: directory,
                using: tokenizerLoader)
        } else {
            container = try await VLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: tokenizerLoader,
                configuration: ModelConfiguration(id: modelID)
            ) { progress in
                guard progress.totalUnitCount > 0 else { return }
                let percent = Int(
                    (Double(progress.completedUnitCount) / Double(progress.totalUnitCount)) * 100)
                print("Download progress: \(percent)%")
            }
        }

        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: maxTokens, temperature: 0.0),
            // Do not pre-resize: DeepseekOCR gundam tiling needs the native page size
            // (ChatSession's default 512² best-fit shrinks 800×400 → 512×256 and skips crops).
            processing: .init(),
            additionalContext: DeepseekOCRProcessor.modeContext(mode))

        let response = try await session.respond(
            to: prompt,
            image: .url(imageURL))

        try response.write(to: outputURL, atomically: true, encoding: .utf8)
        print("Wrote \(outputURL.path) chars: \(response.count)")
    }
}

private enum SmokeError: LocalizedError {
    case missingEnvironment(String)
    case invalidMode(String)

    var errorDescription: String? {
        switch self {
        case .missingEnvironment(let name):
            return "Missing required environment variable: \(name)"
        case .invalidMode(let raw):
            let allowed = DeepseekOCRProcessor.Mode.allCases.map(\.rawValue).joined(separator: ", ")
            return "Invalid MODE=\(raw); expected one of: \(allowed)"
        }
    }
}
