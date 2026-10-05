// Copyright © 2026 Apple Inc.

import AVFoundation
import CoreMedia
import Foundation
import MLXLMCommon
import Testing

@Suite("UserInput.Video source")
struct UserInputVideoTests {

    @Test("The url factory makes a url source")
    func urlFactoryMakesAURLSource() {
        let video = UserInput.Video.url(URL(fileURLWithPath: "/tmp/example.mov"))
        guard case .url(let url) = video.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/example.mov")
    }

    @Test("The avAsset factory makes an avAsset source")
    func avAssetFactoryMakesAnAVAssetSource() {
        let asset = AVURLAsset(url: URL(fileURLWithPath: "/tmp/asset.mov"))
        let video = UserInput.Video.avAsset(asset)
        guard case .avAsset(let sourceAsset) = video.source else {
            Issue.record("expected an avAsset source")
            return
        }
        #expect((sourceAsset as? AVURLAsset)?.url.path == "/tmp/asset.mov")
    }

    @Test("The frames factory makes a frames source")
    func framesFactoryMakesAFramesSource() {
        let frame = UserInput.VideoFrame(
            image: .url(URL(fileURLWithPath: "/tmp/frame.png")),
            timeStamp: CMTime(value: 1200, timescale: 600))
        let video = UserInput.Video.frames([frame])
        guard case .frames(let frames) = video.source else {
            Issue.record("expected a frames source")
            return
        }
        #expect(frames.count == 1)
        #expect(frames.first?.timeStamp.seconds == 2)
    }

    @Test("A factory works as an implicit member")
    func factoryWorksAsAnImplicitMember() {
        let video: UserInput.Video = .url(URL(fileURLWithPath: "/tmp/implicit.mov"))
        guard case .url(let url) = video.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/implicit.mov")
    }

    @Test("A factory can be passed where a function of one argument is expected")
    func factoryWorksAsAFunctionValue() {
        let urls = [URL(fileURLWithPath: "/tmp/a.mov"), URL(fileURLWithPath: "/tmp/b.mov")]
        let videos = urls.map(UserInput.Video.url)
        let paths = videos.compactMap { video -> String? in
            guard case .url(let url) = video.source else { return nil }
            return url.path
        }
        #expect(paths == ["/tmp/a.mov", "/tmp/b.mov"])
    }

    @Test("A factory works as a function value without a type context")
    func factoryWorksWithoutATypeContext() {
        let make = UserInput.Video.url
        let video = make(URL(fileURLWithPath: "/tmp/bare.mov"))
        guard case .url(let url) = video.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/bare.mov")
    }

    @Test("The initializer stores the source")
    func initializerStoresTheSource() {
        let video = UserInput.Video(source: .url(URL(fileURLWithPath: "/tmp/init.mov")))
        guard case .url(let url) = video.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/init.mov")
    }
}
