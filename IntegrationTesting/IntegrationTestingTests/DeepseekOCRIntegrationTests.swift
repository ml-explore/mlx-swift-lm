// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

// MARK: - DeepSeek-OCR (deepseekocr) IntegrationTesting example
//
// Track A / TASK-028. Proves first-class DeepSeek-OCR via `VLMRegistry.deepseekOCR5bit`
// (`model_type=deepseekocr`).
//
// Cache-gated on `mlx-community/DeepSeek-OCR-5bit`, or force with
// `MLX_RUN_DEEPSEEK_OCR_INTEGRATION=1` (Hub download). Default CI never pulls
// multi-GB weights. OCR content quality is not asserted — load + prepare +
// short generate must succeed on the DeepSeek path.

private let deepseekOCRModelId = "mlx-community/DeepSeek-OCR-5bit"

private var shouldRunDeepseekOCRIntegration: Bool {
    ProcessInfo.processInfo.environment["MLX_RUN_DEEPSEEK_OCR_INTEGRATION"] == "1"
        || hfSnapshotDir(modelId: deepseekOCRModelId) != nil
}

@Suite(
    .serialized,
    .timeLimit(.minutes(15)),
    .enabled(if: shouldRunDeepseekOCRIntegration))
struct DeepseekOCRIntegrationTests {

    @Test
    func loadsDeepseekOCRAndRunsOCR() async throws {
        let tokenizerLoader = #huggingFaceTokenizerLoader()
        let container: ModelContainer
        if let dir = hfSnapshotDir(modelId: deepseekOCRModelId),
            ProcessInfo.processInfo.environment["MLX_RUN_DEEPSEEK_OCR_INTEGRATION"] != "1"
        {
            container = try await VLMModelFactory.shared.loadContainer(
                from: dir, using: tokenizerLoader)
        } else {
            container = try await VLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: tokenizerLoader,
                configuration: VLMRegistry.deepseekOCR5bit)
        }

        let isDeepseek = await container.perform { $0.model is DeepseekOCR }
        #expect(isDeepseek, "expected DeepseekOCR for deepseekocr path")

        // Synthetic page — asserts path health, not OCR accuracy.
        let page = VisionTestImages.solidColor(.white, size: 640)
        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: 32, temperature: 0),
            // Keep native resolution (ChatSession default 512² best-fit breaks tiling).
            processing: .init(),
            additionalContext: DeepseekOCRProcessor.modeContext(.base))

        let text = try await session.respond(
            to: "Free OCR.",
            image: .ciImage(page))
        #expect(!text.isEmpty, "DeepSeek-OCR produced empty output")
    }
}
