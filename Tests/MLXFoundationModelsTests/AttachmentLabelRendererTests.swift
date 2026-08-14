// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite("Attachment label rendering")
struct AttachmentLabelRendererTests {

    @Test
    func testNoImagesProducesNoLegend() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(AttachmentLabelRenderer.default.legend(for: []) == nil)
    }

    @Test
    func testUnlabeledImagesProduceNoLegend() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(AttachmentLabelRenderer.default.legend(for: [nil]) == nil)
        #expect(AttachmentLabelRenderer.default.legend(for: [nil, nil]) == nil)
    }

    @Test
    func testSingleLabeledImageNamesItInTheSingular() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            AttachmentLabelRenderer.default.legend(for: ["Photo_A1B2C3"])
                == "The image above is [Photo_A1B2C3].")
    }

    @Test
    func testMultipleLabeledImagesAreNamedInOrder() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            AttachmentLabelRenderer.default.legend(for: ["Photo_A1B2C3", "Photo_D4E5F6"])
                == "The 2 images above are, in order: [Photo_A1B2C3], [Photo_D4E5F6].")
    }

    @Test
    func testUnlabeledImageKeepsItsPositionInTheLegend() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // Ordering is the only thing tying a name to a picture, so a gap has to
        // occupy its slot rather than be omitted.
        #expect(
            AttachmentLabelRenderer.default.legend(for: [nil, "Photo_D4E5F6"])
                == "The 2 images above are, in order: (unlabeled), [Photo_D4E5F6].")
    }

    @Test
    func testThreeImagesUseCommaSeparatedOrder() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(
            AttachmentLabelRenderer.default.legend(
                for: ["Photo_A1B2C3", "Photo_D4E5F6", "Photo_7788AA"])
                == "The 3 images above are, in order: [Photo_A1B2C3], [Photo_D4E5F6], [Photo_7788AA].")
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
