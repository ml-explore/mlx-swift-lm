// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("UserInput.Audio source")
struct UserInputAudioTests {

    @Test("The url factory makes a url source")
    func urlFactoryMakesAURLSource() {
        let audio = UserInput.Audio.url(URL(fileURLWithPath: "/tmp/example.wav"))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/example.wav")
    }

    @Test("The array factory makes an array source")
    func arrayFactoryMakesAnArraySource() {
        let audio = UserInput.Audio.array(MLXArray([0.25, -0.5, 1] as [Float]))
        guard case .array(let array) = audio.source else {
            Issue.record("expected an array source")
            return
        }
        #expect(array.asArray(Float.self) == [0.25, -0.5, 1])
    }

    @Test("A factory works as an implicit member")
    func factoryWorksAsAnImplicitMember() {
        let audio: UserInput.Audio = .url(URL(fileURLWithPath: "/tmp/implicit.wav"))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/implicit.wav")
    }

    @Test("A factory can be passed where a function of one argument is expected")
    func factoryWorksAsAFunctionValue() {
        let urls = [URL(fileURLWithPath: "/tmp/a.wav"), URL(fileURLWithPath: "/tmp/b.wav")]
        let audios = urls.map(UserInput.Audio.url)
        let paths = audios.compactMap { audio -> String? in
            guard case .url(let url) = audio.source else { return nil }
            return url.path
        }
        #expect(paths == ["/tmp/a.wav", "/tmp/b.wav"])
    }

    @Test("A factory works as a function value without a type context")
    func factoryWorksWithoutATypeContext() {
        let make = UserInput.Audio.url
        let audio = make(URL(fileURLWithPath: "/tmp/bare.wav"))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/bare.wav")
    }

    @Test("The initializer stores the source")
    func initializerStoresTheSource() {
        let audio = UserInput.Audio(source: .url(URL(fileURLWithPath: "/tmp/init.wav")))
        guard case .url(let url) = audio.source else {
            Issue.record("expected a url source")
            return
        }
        #expect(url.path == "/tmp/init.wav")
    }
}
