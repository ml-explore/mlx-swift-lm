// Copyright © 2024 Apple Inc.

import MLXNN

/// A logical count for a module whose storage differs from its model parameters.
package protocol ParameterCountProviding {
    var parameterCount: Int { get }
}

extension Module {

    /// Compute the number of parameters in a possibly quantized model
    public func numParameters() -> Int {
        if let provider = self as? any ParameterCountProviding {
            return provider.parameterCount
        }

        let parameters = filterMap(
            filter: Module.filterLocalParameters, map: Module.mapParameters()
        ).flattened()
        let localCount: Int
        if let quantized = self as? any Quantized {
            localCount = parameters.reduce(0) { count, parameter in
                switch parameter.0 {
                case "weight": count + parameter.1.size * 32 / quantized.bits
                case "bias": count + parameter.1.size
                default: count
                }
            }
        } else {
            localCount = parameters.reduce(0) { $0 + $1.1.size }
        }

        return children().flattenedValues().reduce(localCount) { $0 + $1.numParameters() }
    }
}
