// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXScriptedLM
import Testing

@testable import MLXVLM

// Tests of `LFM2VLProcessor` and its configuration. Expected values come from
// tools/lfm2_vl_test_vectors.py. The tokenizer is `LFM2VLTestTokens` from
// LFM2VLImageProcessingTests.swift.

@Suite struct LFM2VLProcessorPrepareTests {

    private func processor(_ settings: LFM2VLImageSettings = .init()) -> LFM2VLProcessor {
        LFM2VLProcessor(
            LFM2VLProcessorConfiguration(settings: settings), tokenizer: LFM2VLTestTokens.tokenizer)
    }

    private func text(_ string: String) -> [Int] {
        LFM2VLTestTokens.tokenizer.encode(text: string, addSpecialTokens: false)
    }

    /// The template's ids: `<|startoftext|>`, then each turn, then the assistant prompt.
    private func chat(_ turns: [(role: String, content: [Int])]) -> [Int] {
        let (start, end) = (LFM2VLTestTokens.id("<|im_start|>"), LFM2VLTestTokens.id("<|im_end|>"))
        var ids = [LFM2VLTestTokens.id("<|startoftext|>")]
        for turn in turns {
            ids += [start] + text("\(turn.role)\n")
            ids += turn.content
            ids += [end] + text("\n")
        }
        return ids + [start] + text("assistant\n")
    }

    @Test func textOnlyPromptsHaveNoImage() async throws {
        let input = try await processor().prepare(input: UserInput(prompt: "Hi"))
        #expect(input.image == nil)
        // A mask would stop `ChatSession` from reusing the prompt cache.
        #expect(input.text.mask == nil)
        #expect(input.text.tokens.asArray(Int.self) == chat([("user", text("Hi"))]))
    }

    @Test func imageTokenWithoutImagesThrows() async throws {
        // transformers raises when the text has `<image>` but no images come with it.
        await #expect(throws: VLMError.self) {
            _ = try await processor().prepare(input: UserInput(prompt: "Hi <image>"))
        }
    }

    /// `<image>` as the text tokens the processor writes outside user messages.
    private var imageText: [Int] { text("<") + text("image>") }

    @Test func imageTextInAReplyIsNotAPlaceholder() async throws {
        // `ChatSession` stores a reply that holds `<image>` and renders it on every later turn.
        let image = RGB8Image.testPattern(width: 64, height: 48).userImage
        let reply = text("Use the ") + imageText + text(" token")
        let cases: [[UserInput.Image]] = [[image], []]
        for images in cases {
            let input = UserInput(chat: [
                .user("A", images: images), .assistant("Use the <image> token"), .user("C"),
            ])
            let prepared = try await processor().prepare(input: input)
            let run = images.isEmpty ? [] : LFM2VLTestTokens.singleViewRun(70)
            let ids = prepared.text.tokens.asArray(Int.self)
            #expect(
                ids
                    == chat([
                        ("user", run + text("A")), ("assistant", reply), ("user", text("C")),
                    ]))
            #expect(prepared.image?.frames?.count == (images.isEmpty ? nil : 1))
            // The model reads the same text.
            #expect(
                LFM2VLTestTokens.tokenizer.decode(tokenIds: ids)
                    .contains("assistant\nUse the <image> token<|im_end|>"))
        }
    }

    @Test func extraImageTextOutsideUserMessagesIsText() async throws {
        let image = RGB8Image.testPattern(width: 64, height: 48).userImage
        let call = ToolCall(function: .init(name: "look", arguments: ["at": "<image>"]))
        let input = UserInput(chat: [
            .system("Never write <image>"), .user("A", images: [image]),
            .assistant("", toolCalls: [call]), .tool("<image><image>"), .user("C"),
        ])
        let prepared = try await processor().prepare(input: input)
        let ids = prepared.text.tokens.asArray(Int.self)
        #expect(ids.filter { $0 == LFM2VLTestTokens.image }.count == 70)
        #expect(prepared.image?.frames?.count == 1)
        let rendered = LFM2VLTestTokens.tokenizer.decode(tokenIds: ids)
        for expected in [
            "system\nNever write <image><|im_end|>", #""arguments":{"at":"<image>"}"#,
            "tool\n<image><image><|im_end|>",
        ] {
            #expect(rendered.contains(expected), "\(expected)")
        }

        // Raw messages, with the content as a string and as parts.
        let part: [String: any Sendable] = ["type": "text", "text": "<image> y"]
        let messages: [MLXLMCommon.Message] = [
            ["role": "user", "content": "A"], ["role": "assistant", "content": "x <image>"],
            ["role": "tool", "content": [part]], ["role": "user", "content": "C"],
        ]
        let raw = try await processor().prepare(input: UserInput(messages: messages))
        #expect(
            raw.text.tokens.asArray(Int.self)
                == chat([
                    ("user", text("A")), ("assistant", text("x ") + imageText),
                    ("tool", imageText + text(" y")), ("user", text("C")),
                ]))

        // Typed by the user, it still needs its image.
        let typed = UserInput(chat: [.user("A <image>"), .assistant("<image>"), .user("C")])
        await #expect(
            throws: VLMError.processing("The prompt has 1 image placeholders but no images")
        ) {
            _ = try await processor().prepare(input: typed)
        }
    }

    @Test func imageTextInAToolSpecIsText() async throws {
        let image = RGB8Image.testPattern(width: 64, height: 48).userImage
        let parameters: [String: any Sendable] = ["type": "string", "enum": ["<image>", "text"]]
        let function: [String: any Sendable] = [
            "name": "look", "description": "Reads <image> files",
            "parameters": ["kind": parameters],
        ]
        let tool: ToolSpec = ["type": "function", "function": function]
        let input = UserInput(chat: [.user("A", images: [image])], tools: [tool])
        let prepared = try await processor().prepare(input: input)
        let json = try ScriptedTemplateMessage.canonicalJSON([tool])
        let spec = json.components(separatedBy: "<image>").map(text)
        #expect(
            prepared.text.tokens.asArray(Int.self)
                == chat([
                    ("system", Array(spec.joined(separator: imageText))),
                    ("user", LFM2VLTestTokens.singleViewRun(70) + text("A")),
                ]))
    }

    @Test func imageTextBeforeACombiningMarkIsText() async throws {
        // The tokenizer matches `<image>` by code units, also when a mark joins its `>`.
        for mark in ["\u{0301}", "\u{FE0F}", "\u{200D}"] {
            let input = UserInput(chat: [.user("A"), .assistant("Use <image>\(mark)"), .user("C")])
            let prepared = try await processor().prepare(input: input)
            #expect(
                prepared.text.tokens.asArray(Int.self)
                    == chat([
                        ("user", text("A")), ("assistant", text("Use ") + imageText + text(mark)),
                        ("user", text("C")),
                    ]))
        }
    }

    @Test func imageTextKeepsItsImageWhenTheCountsMatch() async throws {
        // Raw messages can place an image with `<image>` in any message, as in transformers.
        let image = RGB8Image.testPattern(width: 64, height: 48).userImage
        let messages: [MLXLMCommon.Message] = [
            ["role": "system", "content": "<image> x"], ["role": "user", "content": "A"],
            ["role": "assistant", "content": "y <image>"], ["role": "user", "content": "C"],
        ]
        let input = UserInput(messages: messages, images: [image, image])
        let prepared = try await processor().prepare(input: input)
        let run = LFM2VLTestTokens.singleViewRun(70)
        #expect(
            prepared.text.tokens.asArray(Int.self)
                == chat([
                    ("system", run + text(" x")), ("user", text("A")),
                    ("assistant", text("y ") + run), ("user", text("C")),
                ]))
        #expect(prepared.image?.frames?.count == 2)
    }

    @Test func imageTextStaysAPlaceholderWhenTheRendersDoNotLineUp() async throws {
        // Without the marker token, or when the second render fails or differs elsewhere, the
        // first render is kept, and the old count error is thrown.
        let changes: [MarkerTemplate.Change] = [.throwing, .longer, .other]
        let tokenizers =
            [LFM2VLTestTokens.make(omitting: ["<|image_start|>"])]
            + changes.map { (change: MarkerTemplate.Change) -> ScriptedTokenizer in
                ScriptedTokenizer(corpus: ["A", "C"], template: MarkerTemplate(change: change))
            }
        let input = UserInput(chat: [.user("A"), .assistant("<image>"), .user("C")])
        for tokenizer in tokenizers {
            let processor = LFM2VLProcessor(
                LFM2VLProcessorConfiguration(settings: .init()), tokenizer: tokenizer)
            await #expect(
                throws: VLMError.processing("The prompt has 1 image placeholders but no images")
            ) {
                _ = try await processor.prepare(input: input)
            }
        }
    }

    @Test func videosThrow() async throws {
        // transformers' `Lfm2VlProcessor` has no video input.
        let video = UserInput.Video.url(URL(fileURLWithPath: "/tmp/video.mov"))
        await #expect(throws: VLMError.processing("LFM2-VL does not accept video")) {
            _ = try await processor().prepare(input: UserInput(prompt: "Hi", videos: [video]))
        }
    }

    @Test func imagesExpandInPromptOrder() async throws {
        // Three images of different sizes in one message: one `<image>` each, images first.
        let sizes = [(64, 48), (1000, 1000), (400, 600)]
        let input = UserInput(
            chat: [
                .user(
                    "Hi",
                    images: sizes.map { RGB8Image.testPattern(width: $0.0, height: $0.1).userImage }
                )
            ])
        let prepared = try await processor().prepare(input: input)

        var content = LFM2VLTestTokens.singleViewRun(70)
        content += LFM2VLTestTokens.tiledRun(columns: 2, rows: 2, thumbnailTokens: 256)
        content += LFM2VLTestTokens.singleViewRun(228)
        let expected = chat([("user", content + text("Hi"))])
        #expect(prepared.text.tokens.shape == [1, expected.count])
        #expect(prepared.text.tokens.asArray(Int.self) == expected)

        let image = try #require(prepared.image)
        #expect(image.pixels.shape == [7, 1024, 768])
        var frames: [[Int]] = [[1, 14, 20]]
        frames += Array(repeating: [1, 32, 32], count: 5)
        frames.append([1, 38, 24])
        #expect(image.frames?.map { [$0.t, $0.h, $0.w] } == frames)
        #expect(
            sha256Hex(image.pixels)
                == "84387c3801f03cccaf096d287528a258fe03ce15568d57de581d7e7da1394219")
    }

    @Test func imagesInSeveralMessagesKeepTheirOrder() async throws {
        let first = RGB8Image.testPattern(width: 64, height: 48)
        let second = RGB8Image.testPattern(width: 400, height: 600)
        let input = UserInput(
            chat: [
                .user("A", images: [first.userImage]),
                .assistant("B"),
                .user("C", images: [second.userImage]),
            ])
        let prepared = try await processor().prepare(input: input)
        let expected = chat([
            ("user", LFM2VLTestTokens.singleViewRun(70) + text("A")), ("assistant", text("B")),
            ("user", LFM2VLTestTokens.singleViewRun(228) + text("C")),
        ])
        #expect(prepared.text.tokens.asArray(Int.self) == expected)
        #expect(prepared.image?.frames?.map { [$0.h, $0.w] } == [[14, 20], [38, 24]])
    }

    /// `images` fitted in `size` as the processor fits them, with `filter`.
    private func fitted(
        _ images: [RGB8Image], in size: CGSize, filter: ResampleFilter = .bilinear
    ) -> [UserInput.Image] {
        images.map { (image: RGB8Image) -> UserInput.Image in
            let fit = MediaProcessing.bestFit(
                CGSize(width: image.width, height: image.height), in: size)
            return image.resized(width: Int(fit.width), height: Int(fit.height), filter: filter)
                .userImage
        }
    }

    @Test func resizeFitsImagesFirst() async throws {
        // transformers on 1024x768: 6 tiles and a thumbnail.
        let image = RGB8Image.testPattern(width: 1024, height: 768)
        var input = UserInput(chat: [.user("Hi", images: [image.userImage])], processing: .init())
        let tiled = try await processor().prepare(input: input)
        #expect(
            tiled.text.tokens.asArray(Int.self)
                == chat([
                    (
                        "user",
                        LFM2VLTestTokens.tiledRun(columns: 3, rows: 2, thumbnailTokens: 234)
                            + text("Hi")
                    )
                ]))
        #expect(
            tiled.image?.frames?.map { [$0.h, $0.w] }
                == Array(repeating: [32, 32], count: 6) + [[26, 36]])
        #expect(
            sha256Hex(try #require(tiled.image).pixels)
                == "89cd469414abb4b21b994679ede2071c72876ac210c76e97c0a889c12ea5ab6f")

        // As `MediaProcessing.apply`: 512x512 fits it to one 512x384 view.
        input.processing.resize = CGSize(width: 512, height: 512)
        let single = try await processor().prepare(input: input)
        #expect(
            single.text.tokens.asArray(Int.self)
                == chat([("user", LFM2VLTestTokens.singleViewRun(192) + text("Hi"))]))
        #expect(single.image?.frames?.map { [$0.h, $0.w] } == [[24, 32]])

        // The fitted image is laid out as usual, so a large size still tiles.
        for size in [CGSize(width: 300, height: 200), CGSize(width: 2048, height: 2048)] {
            input.processing.resize = size
            let prepared = try await processor().prepare(input: input)
            let expected = try await processor().prepare(
                input: UserInput(
                    chat: [.user("Hi", images: fitted([image], in: size))], processing: .init()))
            #expect(
                prepared.text.tokens.asArray(Int.self) == expected.text.tokens.asArray(Int.self),
                "\(size)")
            #expect(
                sha256Hex(try #require(prepared.image).pixels)
                    == sha256Hex(try #require(expected.image).pixels), "\(size)")
            let views = prepared.image?.frames?.count ?? 0
            #expect(size.width > 1024 ? views > 1 : views == 1, "\(size)")
        }

        input.processing.resize = .zero
        await #expect(throws: VLMError.self) {
            _ = try await processor().prepare(input: input)
        }
    }

    @Test func resizeFitsEveryImage() async throws {
        // Small images grow to fit, as with `MediaProcessing.apply`.
        let images = [(64, 48), (1000, 1000), (400, 600)].map {
            RGB8Image.testPattern(width: $0.0, height: $0.1)
        }
        let size = CGSize(width: 512, height: 512)
        let prepared = try await processor().prepare(
            input: UserInput(
                chat: [.user("Hi", images: images.map(\.userImage))],
                processing: .init(resize: size)))
        let expected = try await processor().prepare(
            input: UserInput(
                chat: [.user("Hi", images: fitted(images, in: size))], processing: .init()))
        #expect(prepared.text.tokens.asArray(Int.self) == expected.text.tokens.asArray(Int.self))
        #expect(
            sha256Hex(try #require(prepared.image).pixels)
                == sha256Hex(try #require(expected.image).pixels))
        #expect(prepared.image?.frames?.map { [$0.h, $0.w] } == [[24, 32], [32, 32], [32, 22]])
    }

    @Test func resizeUsesTheCheckpointFilter() async throws {
        var settings = LFM2VLImageSettings()
        settings.maxImageTokens = 128
        settings.resample = .bicubic
        let image = RGB8Image.testPattern(width: 1024, height: 768)
        let size = CGSize(width: 512, height: 512)
        let prepared = try await processor(settings).prepare(
            input: UserInput(
                chat: [.user("Hi", images: [image.userImage])], processing: .init(resize: size)))
        for filter in ResampleFilter.allCases {
            let expected = try await processor(settings).prepare(
                input: UserInput(
                    chat: [.user("Hi", images: fitted([image], in: size, filter: filter))],
                    processing: .init()))
            let same =
                sha256Hex(try #require(prepared.image).pixels)
                == sha256Hex(try #require(expected.image).pixels)
            #expect(same == (filter == .bicubic), "\(filter)")
        }
    }

    @Test func onlyTilingNeedsTileMarkers() async throws {
        // A tokenizer without the tile and thumbnail markers can run settings that do not tile:
        // transformers gives 1024x768 one 576x416 view with `do_image_splitting=False`.
        let markers = Set(LFM2VLTestTemplate().specialTokens.filter { $0.hasPrefix("<|img_") })
        let tokenizer = LFM2VLTestTokens.make(omitting: markers)
        var settings = LFM2VLImageSettings()
        settings.doImageSplitting = false
        let image = RGB8Image.testPattern(width: 1024, height: 768)
        let input = UserInput(chat: [.user("Hi", images: [image.userImage])])
        let prepared = try await LFM2VLProcessor(
            LFM2VLProcessorConfiguration(settings: settings), tokenizer: tokenizer
        ).prepare(input: input)
        let imageToken = try #require(tokenizer.convertTokenToId("<image>"))
        #expect(prepared.text.tokens.asArray(Int.self).filter { $0 == imageToken }.count == 234)
        #expect(prepared.image?.frames?.map { [$0.h, $0.w] } == [[26, 36]])

        // Settings that tile need them, with a resize too.
        let processor = LFM2VLProcessor(
            LFM2VLProcessorConfiguration(settings: .init()), tokenizer: tokenizer)
        for resize in [nil, CGSize(width: 512, height: 512)] {
            var tiled = input
            tiled.processing.resize = resize
            await #expect(
                throws: VLMError.processing("The tokenizer has no <|img_thumbnail|> token")
            ) {
                _ = try await processor.prepare(input: tiled)
            }
        }
    }

    @Test func badImagesThrowVLMError() async throws {
        let damaged = FileManager.default.temporaryDirectory
            .appendingPathComponent("lfm2vl-damaged-\(UUID().uuidString).png")
        try Data("not an image".utf8).write(to: damaged)
        defer { try? FileManager.default.removeItem(at: damaged) }
        await #expect(
            throws: VLMError.imageProcessingFailure("unsupported or damaged image data")
        ) {
            _ = try await processor().prepare(
                input: UserInput(prompt: "Hi", images: [.url(damaged)]))
        }

        // Too thin to tile: the 32x8576 single view has 1072 patches, more than 1024.
        let thin = RGB8Image.testPattern(width: 32, height: 9000).userImage
        await #expect(
            throws: VLMError.imageProcessingFailure(
                "a 32x9000 image needs 1072 patches in one view, more than 1024")
        ) {
            _ = try await processor().prepare(input: UserInput(prompt: "Hi", images: [thin]))
        }
    }

    @Test func imagesAreReadOneAtATime() async throws {
        // The first image is laid out before the second is decoded, so its error comes first.
        let damaged = FileManager.default.temporaryDirectory
            .appendingPathComponent("lfm2vl-damaged-\(UUID().uuidString).png")
        try Data("not an image".utf8).write(to: damaged)
        defer { try? FileManager.default.removeItem(at: damaged) }
        let thin = RGB8Image.testPattern(width: 32, height: 9000).userImage
        await #expect(
            throws: VLMError.imageProcessingFailure(
                "a 32x9000 image needs 1072 patches in one view, more than 1024")
        ) {
            _ = try await processor().prepare(
                input: UserInput(prompt: "Hi", images: [thin, .url(damaged)]))
        }
    }

    @Test func severalImagesKeepTheirTokensAndPixels() async throws {
        // (images, sha256 of the token ids, sha256 of the pixels), as they were when every image
        // was decoded before the first was processed.
        let sizes = [
            (1024, 768), (64, 48), (1000, 1000), (400, 600), (48, 80), (640, 480), (300, 1400),
            (886, 1182),
        ]
        let images = sizes.map { (size: (Int, Int)) -> RGB8Image in
            RGB8Image.testPattern(width: size.0, height: size.1)
        }
        let cases: [(Int, String, String)] = [
            (
                1, "988b37f3905f4013726be79bd6117203e101d53bdf8404d874016aa37f45142d",
                "89cd469414abb4b21b994679ede2071c72876ac210c76e97c0a889c12ea5ab6f"
            ),
            (
                4, "901b1e033e4742efdd37bb584233dc6782e70769744490e00914cbf300fc50ed",
                "e4a932a3b885d54f92199406f810cf719512bf9d3ff5e838fcbea4077c70254f"
            ),
            (
                8, "3fc58bfdfde474737d21ab7a83544e5d515f8ac564a6020e7557d4ac090a3a78",
                "cd5dfd6f22cfc3c386a490267fd727395f428106208e1a4ba2d8ed301ee63a34"
            ),
        ]
        let size = CGSize(width: 512, height: 512)
        for (count, tokens, pixels) in cases {
            let some = Array(images.prefix(count))
            let input = UserInput(chat: [.user("Hi", images: some.map(\.userImage))])
            let prepared = try await processor().prepare(input: input)
            #expect(sha256Hex(prepared.text.tokens) == tokens, "\(count)")
            #expect(sha256Hex(try #require(prepared.image).pixels) == pixels, "\(count)")

            // With a resize, as the fitted images without one.
            let resized = try await processor().prepare(
                input: UserInput(
                    chat: [.user("Hi", images: some.map(\.userImage))],
                    processing: .init(resize: size)))
            let fittedInput = try await processor().prepare(
                input: UserInput(chat: [.user("Hi", images: fitted(some, in: size))]))
            #expect(
                sha256Hex(resized.text.tokens) == sha256Hex(fittedInput.text.tokens), "\(count)")
            #expect(
                sha256Hex(try #require(resized.image).pixels)
                    == sha256Hex(try #require(fittedInput.image).pixels), "\(count)")
        }
    }

    @Test func tooManyPlaceholdersThrow() async throws {
        // A literal `<image>` in the prompt has no image to go with it.
        let input = UserInput(
            prompt: "Hi <image>", images: [RGB8Image.testPattern(width: 64, height: 48).userImage])
        await #expect(throws: VLMError.self) {
            _ = try await processor().prepare(input: input)
        }
    }
}

@Suite struct LFM2VLProcessorConfigurationTests {

    @Test func publicAccessorsShowTheSettings() throws {
        // mlx-community repos turn tiling off and ship `max_num_patches: 256`, which
        // transformers ignores.
        let configuration = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self,
            from: Data(
                #"""
                {"image_processor": {"do_image_splitting": false, "use_thumbnail": false,
                 "max_tiles": 4, "max_num_patches": 256, "image_std": 0.25, "tile_size": 384},
                 "processor_class": "Lfm2VlProcessor"}
                """#.utf8))
        #expect(configuration.maxTiles == 4 && configuration.tileSize == 384)
        #expect(configuration.imageMean == [0.5, 0.5, 0.5])
        #expect(configuration.imageStd == [0.25, 0.25, 0.25])
        let std = configuration.imageStdTuple
        #expect([std.0, std.1, std.2] == [0.25, 0.25, 0.25])
        #expect(configuration.encoderPatchSize == 16 && configuration.downsampleFactor == 2)
        #expect(!configuration.settings.doImageSplitting && !configuration.settings.useThumbnail)
        #expect(configuration.settings.maxNumPatches == 1024)
    }

    @Test func encodingRoundTrips() throws {
        var settings = LFM2VLImageSettings()
        settings.resample = .bicubic
        settings.imageMean = [0.25, 0.5, 0.75]
        settings.maxImageTokens = 128
        settings.maxPixelsTolerance = 1.5
        settings.doPad = false
        settings.doImageSplitting = false
        settings.minTiles = 3
        settings.maxTiles = 7
        settings.useThumbnail = false
        let data = try JSONEncoder().encode(LFM2VLProcessorConfiguration(settings: settings))
        #expect(
            try JSONDecoder().decode(LFM2VLProcessorConfiguration.self, from: data).settings
                == settings)
    }

    @Test func factoryResolvesLikeMLXVLM() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "LFM2VLProcessorTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        func write(_ name: String, _ json: String) throws {
            try Data(json.utf8).write(to: directory.appending(component: name))
        }
        func resolvedSettings(
            _ registry: VLMProcessorLoadingRegistry = .shared
        ) async throws -> LFM2VLImageSettings {
            let resolved = try await resolveProcessorConfiguration(
                from: directory,
                context: VLMProcessorLoadingContext(
                    modelId: "test", modelType: "lfm2_vl", configurationData: Data("{}".utf8)),
                registry: registry)
            #expect(resolved.processorType == "Lfm2VlProcessor")
            return try JSONDecoder().decode(
                LFM2VLProcessorConfiguration.self, from: resolved.data
            ).settings
        }

        // Flat keys in `processor_config.json` override `preprocessor_config.json`, and the
        // tiling settings keep their defaults. `encoder_patch_size` wins over `patch_size`.
        try write(
            "preprocessor_config.json",
            """
            {"processor_class": "Lfm2VlProcessor", "resample": 3, "max_image_tokens": 128,
             "image_mean": [0.25, 0.25, 0.25], "do_image_splitting": false,
             "encoder_patch_size": 16, "patch_size": 14}
            """)
        try write(
            "processor_config.json",
            #"{"processor_class": "Lfm2VlProcessor", "max_image_tokens": 200, "max_tiles": 4}"#)
        var expected = LFM2VLImageSettings()
        expected.resample = .bicubic
        expected.maxImageTokens = 200
        expected.imageMean = [0.25, 0.25, 0.25]
        #expect(try await resolvedSettings() == expected)

        // A nested `image_processor` overrides the same way, with any registry.
        try write(
            "processor_config.json",
            """
            {"processor_class": "Lfm2VlProcessor", "use_image_special_tokens": true,
             "image_processor": {"resample": 2, "min_image_tokens": 32}}
            """)
        expected = LFM2VLImageSettings()
        expected.maxImageTokens = 128
        expected.imageMean = [0.25, 0.25, 0.25]
        expected.minImageTokens = 32
        #expect(try await resolvedSettings() == expected)
        #expect(try await resolvedSettings(VLMProcessorLoadingRegistry()) == expected)

        // The LFM2.5-VL-3B MLX repos: no `resample` reads as bicubic, and tiling stays on.
        try FileManager.default.removeItem(
            at: directory.appending(component: "preprocessor_config.json"))
        try write(
            "processor_config.json",
            """
            {"processor_class": "Lfm2VlProcessor",
             "image_processor": {"do_image_splitting": false, "use_thumbnail": false}}
            """)
        expected = LFM2VLImageSettings()
        expected.resample = .bicubic
        #expect(try await resolvedSettings() == expected)

        // A file that does not parse is named in the error.
        try write("processor_config.json", "{")
        let error = await #expect(throws: ProcessorConfigError.self) {
            try await resolvedSettings()
        }
        #expect(error?.filename == "processor_config.json")
    }

    @Test func remoteCodeReposReadBothFiles() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "LFM2VLProcessorTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        // The files of mlx-community/LFM2-VL-1.6B-4bit without `auto_map`, and with
        // `use_thumbnail` false as in the 450M repo. The thumbnail stays on.
        try Data(
            #"""
            {"do_convert_rgb": null, "do_normalize": true, "do_rescale": true, "do_resize": false,
             "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "Siglip2ImageProcessor",
             "image_std": [0.5, 0.5, 0.5], "input_data_format": "channels_last",
             "max_num_patches": 1024, "patch_size": 16, "processor_class": "Lfm2VlProcessor",
             "resample": 2, "rescale_factor": 0.00392156862745098, "return_tensors": "pt"}
            """#.utf8
        ).write(to: directory.appending(component: "preprocessor_config.json"))
        try Data(
            #"""
            {"do_image_splitting": true, "downsample_factor": 2, "encoder_patch_size": 16,
             "max_image_tokens": 256, "max_num_patches": 1024, "max_pixels_tolerance": 1.5,
             "max_tiles": 10, "min_image_tokens": 64, "min_tiles": 2,
             "processor_class": "Lfm2VlProcessor", "tile_size": 512,
             "use_image_special_tokens": true, "use_thumbnail": false}
            """#.utf8
        ).write(to: directory.appending(component: "processor_config.json"))
        let resolved = try await resolveProcessorConfiguration(
            from: directory,
            context: VLMProcessorLoadingContext(
                modelId: "test", modelType: "lfm2_vl", configurationData: Data("{}".utf8)),
            registry: .shared)
        let settings = try JSONDecoder().decode(
            LFM2VLProcessorConfiguration.self, from: resolved.data
        ).settings
        var expected = LFM2VLImageSettings()
        expected.maxPixelsTolerance = 1.5
        #expect(settings == expected)
        // With the shipped 1.5, an 800x600 image is tiled.
        let layout = try LFM2VLImageLayout.plan(width: 800, height: 600, settings: settings)
        #expect(layout.tiled && layout.gridColumns == 3 && layout.gridRows == 2)
    }
}

/// `LFM2VLTestTemplate`, but a chat that holds `<|image_start|>` as text renders as `change`
/// says.
private struct MarkerTemplate: ScriptedChatTemplate {
    enum Change { case throwing, longer, other }
    let change: Change
    let base = LFM2VLTestTemplate()

    var specialTokens: [String] { base.specialTokens }
    var corpus: [String] { base.corpus }
    var endOfTurnToken: String? { base.endOfTurnToken }

    func render(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [String] {
        let segments = try base.render(
            messages: messages, tools: tools, additionalContext: additionalContext)
        guard segments.contains("<|image_start|>") else { return segments }
        switch change {
        case .throwing: throw ScriptedTemplateError.unsupported("<|image_start|>")
        case .longer: return segments + ["\n"]
        case .other: return ["<|im_start|>"] + segments.dropFirst()
        }
    }
}
