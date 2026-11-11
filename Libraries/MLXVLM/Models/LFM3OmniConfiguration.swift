import Foundation
import MLXLMCommon

struct LFM3OmniConfiguration: Decodable, Sendable {
    let visionLanguage: LFM3VLConfiguration
    let audioEncoder: LFM3OmniAudioConfiguration
    let depth: LFM3Configuration
    let audioToken: Int
    let audioOutputToken: Int
    let audioStartToken: Int
    let mixedStartToken: Int
    let textEndToken: Int
    let audioEndCode: Int
    let codebookSize: Int
    let codebooks: Int
    let schedule: [Int]
    let textBlock: Int
    let audioBlock: Int

    enum CodingKeys: String, CodingKey {
        case audioEncoder = "audio_encoder_config"
        case depth = "depth_config"
        case audioToken = "audio_token_id"
        case audioOutputToken = "audio_out_token_id"
        case audioStartToken = "audio_start_token_id"
        case mixedStartToken = "mixed_start_token_id"
        case textEndToken = "text_end_token_id"
        case audioEndCode = "audio_eos_code"
        case codebookSize = "codebook_size"
        case codebooks = "num_codebooks"
        case schedule = "maskgit_schedule"
        case textBlock = "interleaved_n_text"
        case audioBlock = "interleaved_n_audio"
    }

    init(from decoder: Decoder) throws {
        visionLanguage = try LFM3VLConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        audioEncoder = try c.decode(LFM3OmniAudioConfiguration.self, forKey: .audioEncoder)
        depth = try c.decode(LFM3Configuration.self, forKey: .depth)
        audioToken = try c.decode(Int.self, forKey: .audioToken)
        audioOutputToken = try c.decode(Int.self, forKey: .audioOutputToken)
        audioStartToken = try c.decode(Int.self, forKey: .audioStartToken)
        mixedStartToken = try c.decode(Int.self, forKey: .mixedStartToken)
        textEndToken = try c.decode(Int.self, forKey: .textEndToken)
        audioEndCode = try c.decode(Int.self, forKey: .audioEndCode)
        codebookSize = try c.decode(Int.self, forKey: .codebookSize)
        codebooks = try c.decode(Int.self, forKey: .codebooks)
        schedule = try c.decode([Int].self, forKey: .schedule)
        textBlock = try c.decode(Int.self, forKey: .textBlock)
        audioBlock = try c.decode(Int.self, forKey: .audioBlock)
        let tokens = [
            audioToken, audioOutputToken, audioStartToken, mixedStartToken, textEndToken,
            visionLanguage.imageToken, visionLanguage.videoToken,
        ]
        guard codebookSize > 1, codebooks > 0,
            codebooks <= Int.max / codebookSize, codebooks <= Int.max / depth.hiddenSize,
            audioEndCode >= 0, audioEndCode < codebookSize,
            textBlock > 0, audioBlock > 0, !schedule.isEmpty,
            Set(tokens).count == tokens.count,
            tokens.allSatisfy({ $0 >= 0 && $0 < visionLanguage.text.vocabularySize }),
            depth.layerTypes.allSatisfy({ $0 == .fullAttention })
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "Invalid LFM3-Omni speech configuration"))
        }
        var remaining = codebooks
        for count in schedule {
            guard count > 0, count <= remaining else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: decoder.codingPath,
                        debugDescription: "LFM3-Omni mask schedule must partition its codebooks"))
            }
            remaining -= count
        }
        guard remaining == 0 else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "LFM3-Omni mask schedule leaves unrevealed codebooks"))
        }
    }
}
