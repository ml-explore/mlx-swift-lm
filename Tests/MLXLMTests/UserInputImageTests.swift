// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Testing

/// `UserInput.Image` is a struct, so the factories must produce what the enum
/// cases produced, and a label must default to none.
@Suite("UserInput.Image label")
struct UserInputImageTests {

    @Test("A URL image defaults to no label")
    func urlImageHasNoLabelByDefault() {
        let image = UserInput.Image.url(URL(fileURLWithPath: "/tmp/example.png"))
        #expect(image.label == nil)
        guard case .url(let url) = image.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/example.png")
    }

    @Test("A factory carries the label through to the value")
    func factoryCarriesTheLabel() {
        let image = UserInput.Image.url(
            URL(fileURLWithPath: "/tmp/example.png"), label: "Photo_A1B2C3")
        #expect(image.label == "Photo_A1B2C3")
    }

    @Test("A CIImage source round-trips through asCIImage")
    func ciImageSourceRoundTrips() throws {
        let source = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 2))
        let image = UserInput.Image.ciImage(source, label: "A")
        let recovered = try image.asCIImage()
        #expect(recovered.extent.width == 4)
        #expect(recovered.extent.height == 2)
        #expect(image.label == "A")
    }

    @Test("An array source becomes an image of the same size")
    func arraySourceBecomesAnImage() throws {
        let array = MLXArray.ones([3, 3, 4]) * 255
        let image = UserInput.Image.array(array)
        let recovered = try image.asCIImage()
        #expect(recovered.extent.width == 4)
        #expect(recovered.extent.height == 3)
        #expect(image.label == nil)
    }

    @Test("A factory can be passed where a function of one argument is expected")
    func factoryWorksAsAFunctionValue() {
        let urls = [URL(fileURLWithPath: "/tmp/a.png"), URL(fileURLWithPath: "/tmp/b.png")]
        let images = urls.map(UserInput.Image.url)
        #expect(images.count == 2)
        #expect(images.allSatisfy { $0.label == nil })
    }

    @Test("A label can be set after construction")
    func labelIsMutable() {
        var image = UserInput.Image.url(URL(fileURLWithPath: "/tmp/example.png"))
        image.label = "B"
        #expect(image.label == "B")
    }
}
