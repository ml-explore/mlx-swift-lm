// Copyright © 2026 Apple Inc.

import Foundation
import MLX

#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

#if canImport(AVFoundation)
extension UserInput.AudioFormat {
    public var asAudioFormatID: AudioFormatID {
        switch self {
        case .linearPCM:
            return kAudioFormatLinearPCM
        }
    }
}
#endif

extension UserInput.Audio {

    public func asMLXArray(processing: UserInput.AudioProcessing = .init()) async throws -> MLXArray
    {
        switch self {
        case .url(let url):
            #if canImport(AVFoundation)
            let asset = AVURLAsset(url: url)

            guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
                throw UserInputError.noAudioData(url)
            }

            let settings: [String: Any] = [
                AVFormatIDKey: processing.audioFormat.asAudioFormatID,
                AVSampleRateKey: processing.sampleRate,
                AVNumberOfChannelsKey: processing.channels,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsNonInterleaved: false,
            ]

            let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
            let reader = try AVAssetReader(asset: asset)
            reader.add(output)
            reader.startReading()

            var samples: [Float] = []

            while let sampleBuffer = output.copyNextSampleBuffer() {
                guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { continue }

                let byteCount = CMBlockBufferGetDataLength(blockBuffer)
                var chunk = [Float](repeating: 0, count: byteCount / MemoryLayout<Float>.size)
                CMBlockBufferCopyDataBytes(
                    blockBuffer, atOffset: 0, dataLength: byteCount, destination: &chunk)
                samples.append(contentsOf: chunk)
            }

            guard reader.status == .completed else {
                throw reader.error ?? UserInputError.unableToLoad(url)
            }

            return MLXArray(samples)
            #else
            fatalError("Audio processing is not supported on this platform.")
            #endif
        case .array(let array):
            return array
        }
    }
}
