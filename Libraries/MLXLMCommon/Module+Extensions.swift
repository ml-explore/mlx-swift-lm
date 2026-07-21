// Copyright © 2024 Apple Inc.

import MLXNN

extension Module {

    /// Compute the number of parameters in a possibly quantized model
    @available(*, deprecated, message: "use parameterCount (per module)")
    public func numParameters() -> Int {
        parameterCount
    }
}
