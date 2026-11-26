import Foundation
import MLX
import MLXLMCommon

struct LFM3OmniProcessorConfiguration: Decodable, Sendable {
    let vision: LFM3VLProcessorConfiguration
    let audioToken: String
    struct Features: Decodable, Sendable {
        let rate: Int
        let bins: Int
        let fft: Int
        let window: Int
        let hop: Int
        let stride: Int
        let floor: Float
        let offset: Float
        let divisor: Float
        enum CodingKeys: String, CodingKey {
            case rate = "sampling_rate"
            case bins = "feature_size"
            case fft = "n_fft"
            case window = "win_length"
            case hop = "hop_length"
            case stride = "encoder_stride"
            case floor = "log_floor"
            case offset = "normalization_offset"
            case divisor = "normalization_divisor"
        }
    }
    enum CodingKeys: String, CodingKey {
        case features = "feature_extractor"
        case audioToken = "audio_token"
    }
    init(from decoder: Decoder) throws {
        vision = try LFM3VLProcessorConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        audioToken = try c.decode(String.self, forKey: .audioToken)
        let f = try c.decode(Features.self, forKey: .features)
        guard f.rate == 16000, f.bins == 128, f.fft == 400, f.window == 400,
            f.hop == 160, f.stride == 8, f.floor == -12, f.offset == 4, f.divisor == 4,
            audioToken == "<|audio|>"
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "Unsupported LFM3-Omni audio frontend configuration"))
        }
    }
}

struct LFM3OmniMessageGenerator: MessageGenerator {
    func generate(message: Chat.Message) -> Message {
        var result: Message = [
            "role": message.role.rawValue,
            "content": contentParts(for: message, layout: .imagesThenVideosThenAudioThenText),
        ]
        addReasoning(to: &result, for: message)
        addToolMetadata(to: &result, for: message)
        return result
    }
}

struct LFM3OmniProcessor: UserInputProcessor {
    let config: LFM3OmniProcessorConfiguration
    let tokenizer: any Tokenizer

    init(_ config: LFM3OmniProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func prepare(input: UserInput) async throws -> LMInput {
        if input.additionalContext?["enable_thinking"] as? Bool == true {
            throw VLMError.processing("LFM3-Omni does not support thinking mode")
        }
        var context = input.additionalContext ?? [:]
        context["enable_thinking"] = false
        let prompt = try tokenizer.applyChatTemplate(
            messages: LFM3OmniMessageGenerator().generate(
                from: input.removingSpecialTokenLabels(using: tokenizer)),
            tools: input.tools, additionalContext: context)
        guard let audioToken = tokenizer.convertTokenToId(config.audioToken),
            let imageToken = tokenizer.convertTokenToId("<|image|>"),
            let videoToken = tokenizer.convertTokenToId("<|video|>"),
            let startToken = tokenizer.convertTokenToId("<|vision_start|>"),
            let endToken = tokenizer.convertTokenToId("<|vision_end|>")
        else { throw VLMError.processing("LFM3-Omni tokenizer is missing media markers") }
        var clips = [MLXArray]()
        for audio in input.audios {
            var processing = input.processing.audio
            switch audio.source {
            case .array:
                guard processing.sampleRate == 16000, processing.channels == 1 else {
                    throw VLMError.processing("LFM3-Omni sample arrays must be mono at 16 kHz")
                }
            case .url:
                processing.sampleRate = 16000
                processing.channels = 1
            }
            let samples = try await audio.asMLXArray(processing: processing)
            clips.append(try LFM3OmniAudioFeatures()(samples))
        }
        let media = Set([imageToken, videoToken, audioToken])
        var unwrapped = [Int]()
        var position = 0
        while position < prompt.count {
            if position + 2 < prompt.count, prompt[position] == startToken,
                media.contains(prompt[position + 1]), prompt[position + 2] == endToken
            {
                unwrapped.append(prompt[position + 1])
                position += 3
            } else {
                unwrapped.append(prompt[position])
                position += 1
            }
        }
        let audioPositions = unwrapped.indices.filter { unwrapped[$0] == audioToken }
        let videoPositions = unwrapped.indices.filter { unwrapped[$0] == videoToken }
        guard audioPositions.count == clips.count else {
            throw VLMError.processing(
                "LFM3-Omni audio placeholders do not match the supplied clips")
        }
        let useSoundtracks = context["use_audio_in_video"] as? Bool ?? false
        var paired = [Int: Int]()
        if useSoundtracks {
            var used = Set<Int>()
            for (index, position) in videoPositions.enumerated() {
                guard
                    let clip = audioPositions.indices.first(where: {
                        audioPositions[$0] > position && !used.contains($0)
                    })
                else {
                    throw VLMError.processing(
                        "Each LFM3-Omni video needs a following audio clip for its soundtrack")
                }
                paired[index] = clip
                used.insert(clip)
            }
        }
        let soundtrackClips = Set(paired.values)
        let counts = clips.map { ($0.dim(1) + 7) / 8 }
        var expanded = [Int]()
        var clipIndex = 0
        var videoIndex = 0
        var clipOrder = [Int]()
        for token in unwrapped {
            if token == audioToken {
                if !soundtrackClips.contains(clipIndex) {
                    expanded.append(startToken)
                    expanded.append(contentsOf: repeatElement(audioToken, count: counts[clipIndex]))
                    expanded.append(endToken)
                    clipOrder.append(clipIndex)
                }
                clipIndex += 1
            } else if token == videoToken {
                expanded.append(contentsOf: [startToken, token, endToken])
                if let clip = paired[videoIndex] { clipOrder.append(clip) }
                videoIndex += 1
            } else if token == imageToken {
                expanded.append(contentsOf: [startToken, token, endToken])
            } else {
                expanded.append(token)
            }
        }
        let visual = try await LFM3VLProcessor(config.vision, tokenizer: tokenizer).prepare(
            input: input, prompt: expanded,
            videoExpansion: { index, groups in
                guard let clip = paired[index] else { return groups.flatMap { $0.tokens } }
                var bounds = [0]
                for group in groups.dropFirst() {
                    guard group.startTime.isFinite, group.startTime >= 0 else {
                        throw VLMError.processing("Invalid LFM3-Omni soundtrack timestamp")
                    }
                    let boundary = Int(
                        min(
                            Double(counts[clip]), (group.startTime / 0.08).rounded(.toNearestOrEven)
                        ))
                    bounds.append(max(bounds.last!, boundary))
                }
                bounds.append(counts[clip])
                var tokens = [Int]()
                for (index, group) in groups.enumerated() {
                    tokens.append(contentsOf: group.tokens)
                    tokens.append(
                        contentsOf: repeatElement(
                            audioToken, count: bounds[index + 1] - bounds[index]))
                }
                return tokens
            })
        clips = clipOrder.map { clips[$0] }
        var audio: LMInput.ProcessedAudio?
        if let longest = clips.map({ $0.dim(1) }).max() {
            let features = stacked(
                clips.map {
                    concatenated([$0, MLXArray.zeros([128, longest - $0.dim(1)])], axis: 1)
                })
            let mask =
                MLXArray(0 ..< longest)[.newAxis, 0...]
                .< MLXArray(clips.map { $0.dim(1) })[0..., .newAxis]
            audio = .init(features: features, mask: mask)
        }
        return LMInput(
            text: visual.text,
            image: visual.image, video: visual.video, audio: audio)
    }
}
