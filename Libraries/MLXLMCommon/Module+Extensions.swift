// Copyright © 2024 Apple Inc.

import MLXNN

extension Module {

    /// Compute the number of parameters in a possibly quantized model
    ///
    /// Counts the parameters of every module, including those of a module that also has
    /// sub-modules. A quantized layer counts the values its packed weight holds, `32 / bits` per
    /// element, and its bias. Its scales, biases and global scales are not parameters.
    public func numParameters() -> Int {
        return modules().map {
            mod -> Int in
            let parameters = mod.filterMap(
                filter: Module.filterLocalParameters, map: Module.mapParameters()
            ).flattened()
            guard let quantized = mod as? any Quantized else {
                return parameters.reduce(0) { $0 + $1.1.size }
            }
            return parameters.reduce(0) { count, parameter in
                switch parameter.0 {
                case "weight": count + parameter.1.size * 32 / quantized.bits
                case "bias": count + parameter.1.size
                default: count
                }
            }
        }.reduce(0, +)
    }
}
