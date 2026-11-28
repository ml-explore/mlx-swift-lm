// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// Deprecated alias for `ChainedLogitProcessor`.
@available(*, deprecated, renamed: "ChainedLogitProcessor")
public typealias CompositeLogitProcessor = ChainedLogitProcessor

extension ChainedLogitProcessor {
    @available(*, deprecated, renamed: "init(processors:)")
    public init(_ processors: [any LogitProcessor]) {
        self.init(processors: processors)
    }
}
