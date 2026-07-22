// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLXLMCommon

final class LoadWeightsTests: XCTestCase {

    func testLoadWeightsUsesSafetensorsIndexWeightMapWhenPresent() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("mtp.safetensors", in: directory)
        try writeEmptyFile("optiq_vision.safetensors", in: directory)
        try """
        {
          "metadata": { "total_size": 1 },
          "weight_map": {
            "model.norm.weight": "model.safetensors"
          }
        }
        """.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model.safetensors"])
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("LoadWeightsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeEmptyFile(_ name: String, in directory: URL) throws {
        try Data().write(to: directory.appendingPathComponent(name))
    }
}
