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

            var sampleData = Data()
            if let duration = try? await asset.load(.duration) {
                let estimatedByteCount =
                    duration.seconds * processing.sampleRate * Double(processing.channels)
                    * Double(MemoryLayout<Float>.size)
                if estimatedByteCount.isFinite, estimatedByteCount > 0 {
                    // Avoid a single unbounded eager allocation for unusually
                    // long inputs while preventing geometric growth for common
                    // clips.
                    let reserveLimit = 64 * 1_024 * 1_024
                    sampleData.reserveCapacity(Int(min(estimatedByteCount, Double(reserveLimit))))
                }
            }

            while let sampleBuffer = output.copyNextSampleBuffer() {
                guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { continue }

                let byteCount = CMBlockBufferGetDataLength(blockBuffer)
                guard byteCount > 0 else { continue }
                let writeOffset = sampleData.count
                sampleData.append(contentsOf: repeatElement(0, count: byteCount))
                let copyStatus = sampleData.withUnsafeMutableBytes { destination in
                    CMBlockBufferCopyDataBytes(
                        blockBuffer,
                        atOffset: 0,
                        dataLength: byteCount,
                        destination: destination.baseAddress!.advanced(by: writeOffset)
                    )
                }
                guard copyStatus == kCMBlockBufferNoErr else {
                    reader.cancelReading()
                    throw UserInputError.unableToLoad(url)
                }
            }

            guard reader.status == .completed else {
                throw reader.error ?? UserInputError.unableToLoad(url)
            }

            guard sampleData.count.isMultiple(of: MemoryLayout<Float>.size) else {
                throw UserInputError.unableToLoad(url)
            }
            let sampleCount = sampleData.count / MemoryLayout<Float>.size
            return MLXArray(sampleData, [sampleCount], type: Float32.self)
            #else
            fatalError("Audio processing is not supported on this platform.")
            #endif
        case .array(let array):
            return array
        }
    }
}
