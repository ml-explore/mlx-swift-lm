// Copyright © 2026 Apple Inc.

import MLXVLM
import XCTest

final class VLMRegistryTests: XCTestCase {

    func testGemma4VLMRegistryUsesTurnEndToken() {
        for configuration in [
            VLMRegistry.gemma4_E2B_it_4bit,
            VLMRegistry.gemma4_E4B_it_4bit,
            VLMRegistry.gemma4_31B_it_4bit,
            VLMRegistry.gemma4_26BA4B_it_4bit,
        ] {
            XCTAssertEqual(configuration.extraEOSTokens, ["<turn|>"])
        }
    }
}
