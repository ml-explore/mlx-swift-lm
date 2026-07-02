// Copyright © 2026 Apple Inc.

import Foundation

#if os(Linux)

extension String {
    public init(localized resource: String) {
        self = resource
    }
}

#endif
