// Copyright © 2026 Apple Inc.

import MLXLLM
import XCTest

final class LLMRegistryTests: XCTestCase {

    func testFalconH1RModelConfigurationIsRegistered() {
        XCTAssertTrue(LLMRegistry.shared.contains(id: "tiiuae/Falcon-H1R-7B"))

        let configuration = LLMRegistry.falconH1R7B
        XCTAssertEqual(configuration.name, "tiiuae/Falcon-H1R-7B")
    }
}
