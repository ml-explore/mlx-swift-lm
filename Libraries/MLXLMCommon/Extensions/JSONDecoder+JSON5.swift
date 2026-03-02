// Copyright © 2026 Apple Inc.

import Foundation

public extension JSONDecoder {
    static func json5() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.allowsJSON5 = true
        return decoder
    }
}
