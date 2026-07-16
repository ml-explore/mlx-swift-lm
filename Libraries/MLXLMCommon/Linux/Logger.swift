// Copyright © 2026 Apple Inc.

#if canImport(os)

import os

typealias Logger = os.Logger

#else

final class Logger: Sendable {
    private let subsystem: String
    private let category: String

    init(subsystem: String, category: String) {
        self.subsystem = subsystem
        self.category = category
    }

    func info(_ message: String) {
        print("[INFO] [\(subsystem).\(category)] \(message)")
    }

    func error(_ message: String) {
        print("[ERROR] [\(subsystem).\(category)] \(message)")
    }
}

#endif
