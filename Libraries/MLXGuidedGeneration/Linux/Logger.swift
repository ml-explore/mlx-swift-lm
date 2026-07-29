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

    func debug(_ message: String) {
        print("[DEBUG] [\(subsystem).\(category)] \(message)")
    }

    func info(_ message: String) {
        print("[INFO] [\(subsystem).\(category)] \(message)")
    }

    func notice(_ message: String) {
        print("[NOTICE] [\(subsystem).\(category)] \(message)")
    }

    func warning(_ message: String) {
        print("[WARNING] [\(subsystem).\(category)] \(message)")
    }

    func error(_ message: String) {
        print("[ERROR] [\(subsystem).\(category)] \(message)")
    }

    func fault(_ message: String) {
        print("[FAULT] [\(subsystem).\(category)] \(message)")
    }
}

#endif
