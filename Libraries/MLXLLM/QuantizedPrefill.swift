// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import MLXNN

private protocol PrefillModuleProperty {
    func acceptsPrefillReplacement(_ value: Any) -> Bool
}

extension ModuleInfo: PrefillModuleProperty {
    fileprivate func acceptsPrefillReplacement(_ value: Any) -> Bool { value is T }
}

/// Selects a projection route without evaluating GPU work. MLX owns device dispatch.
enum QuantizedPrefill {
    enum Decision: String, Sendable {
        case dense, outsidePrefill, unsupportedInput, unavailableDevice
        case unsupportedProjection, insufficientRows, unsupportedFormat, trainable
        case insufficientWorkspace
    }

    // Optional graph-construction diagnostics, not GPU execution counters.
    @TaskLocal static var observer: (@Sendable (Decision) -> Void)?

    private static func minimumRows(bits: Int) -> Int? {
        switch bits {
        case 2: 384
        case 3, 4, 5, 6, 8: 2048
        default: nil
        }
    }

    static func withPrefill<Result>(enabled: Bool = true, _ body: () throws -> Result) rethrows
        -> Result
    {
        try PrefillExecutionContext.$isActive.withValue(enabled, operation: body)
    }

    private static let recommendedBytes: Int = {
        #if canImport(Metal)
        GPU.maxRecommendedWorkingSetBytes() ?? 0
        #else
        0
        #endif
    }()

    static var isAvailable: Bool {
        Device.defaultDevice().deviceType == .gpu && recommendedBytes > 0
    }

    static func workspaceLimit(recommendedBytes: Int, memoryLimit: Int, activeBytes: Int) -> Int {
        let budget = max(0, min(recommendedBytes, memoryLimit))
        let headroom = budget - min(budget, max(0, activeBytes))
        // Spend at most 1/64 of the remaining allocator budget on one dense projection.
        return headroom / 64
    }

    static func prefersDense(
        rows: Int, inputs: Int, outputs: Int, bits: Int, workspaceLimit: Int
    ) -> Bool {
        // Require enough rows to amortize dequantization without predicting kernel tiles.
        guard let minimum = minimumRows(bits: bits), rows >= minimum,
            inputs > 0, outputs > 0, workspaceLimit > 0
        else { return false }
        return outputs <= workspaceLimit / 2 / inputs
    }

    static func usesDense(
        _ linear: Linear, rows: Int, inputs: Int, dtype: DType, workspaceLimit: Int? = nil
    ) -> Bool {
        decision(linear, rows: rows, inputs: inputs, dtype: dtype, workspaceLimit: workspaceLimit)
            == .dense
    }

    static func decision(
        _ linear: Linear, rows: Int, inputs: Int, dtype: DType, workspaceLimit: Int? = nil
    ) -> Decision {
        guard
            ObjectIdentifier(type(of: linear)) == ObjectIdentifier(QuantizedLinear.self)
                || ObjectIdentifier(type(of: linear))
                    == ObjectIdentifier(PrefillQuantizedLinear.self),
            let linear = linear as? QuantizedLinear
        else { return .unsupportedProjection }
        guard let minimum = minimumRows(bits: linear.bits) else { return .unsupportedFormat }
        guard rows >= minimum else { return .insufficientRows }
        guard dtype == .bfloat16 || dtype == .float16, inputs > 0,
            linear.mode == .affine,
            [32, 64, 128].contains(linear.groupSize), inputs % linear.groupSize == 0,
            linear.globalScale == nil, linear.weight.ndim == 2,
            linear.weight.dtype == .uint32, linear.scales.ndim == 2,
            linear.weight.dim(1) == inputs / 32 * linear.bits,
            linear.scales.dtype == dtype, linear.biases?.dtype == dtype,
            linear.biases?.shape == linear.scales.shape,
            linear.shape.1 == inputs,
            linear.scales.shape == [linear.shape.0, inputs / linear.groupSize],
            linear.bias == nil
                || (linear.bias?.dtype == dtype && linear.bias?.shape == [linear.shape.0])
        else { return .unsupportedFormat }
        guard !linear.training, linear.trainableParameters().flattened().isEmpty
        else { return .trainable }

        let limit =
            workspaceLimit
            ?? self.workspaceLimit(
                recommendedBytes: recommendedBytes,
                memoryLimit: Memory.memoryLimit, activeBytes: Memory.activeMemory)
        return prefersDense(
            rows: rows, inputs: inputs, outputs: linear.shape.0,
            bits: linear.bits, workspaceLimit: limit) ? .dense : .insufficientWorkspace
    }

    static func decision(_ linear: Linear, input x: MLXArray) -> Decision {
        guard PrefillExecutionContext.isActive else { return .outsidePrefill }
        guard x.ndim == 3, x.dim(0) == 1, x.dim(1) > 1 else { return .unsupportedInput }
        guard isAvailable else { return .unavailableDevice }
        return decision(linear, rows: x.dim(1), inputs: x.dim(2), dtype: x.dtype)
    }

    static func dense(_ linear: QuantizedLinear, _ x: MLXArray) -> MLXArray {
        let weight = dequantized(
            linear.weight, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits, dtype: x.dtype)
        let output = matmul(x, weight.T)
        return linear.bias.map { output + $0 } ?? output
    }

    @discardableResult
    static func prepare(_ model: Module, rows: Int) throws -> Bool {
        guard rows >= 384, !model.training, isAvailable else { return false }
        try Task.checkCancellation()
        let limit = workspaceLimit(
            recommendedBytes: recommendedBytes,
            memoryLimit: Memory.memoryLimit, activeBytes: Memory.activeMemory)
        guard limit > 0 else { return false }
        var changed = false
        var canRoute = false
        defer {
            if changed { model.invalidateCompiledTraces() }
        }
        var visited = Set<ObjectIdentifier>()
        for owner in model.modules() where visited.insert(ObjectIdentifier(owner)).inserted {
            var properties: [String: any PrefillModuleProperty]?
            for (key, children) in owner.children().sorted(by: { $0.key < $1.key }) {
                try Task.checkCancellation()
                var replacing = false
                let replacement = children.mapValues { module -> Module in
                    guard let linear = module as? QuantizedLinear, linear.weight.ndim == 2,
                        usesDense(
                            linear, rows: rows, inputs: linear.shape.1, dtype: linear.scales.dtype,
                            workspaceLimit: limit)
                    else { return module }
                    if linear is PrefillQuantizedLinear {
                        canRoute = true
                        return module
                    }
                    replacing = true
                    return PrefillQuantizedLinear(linear)
                }
                guard replacing else { continue }
                if properties == nil { properties = replacementProperties(of: owner) }
                guard let property = properties?[key] else { continue }
                guard let value = directReplacementValue(replacement),
                    property.acceptsPrefillReplacement(value)
                else { continue }
                // MLX replaces containers wholesale. Keep every untouched sibling.
                // One direct property cannot leave a partially updated ancestor tree.
                do {
                    try owner.update(
                        modules: ModuleChildren(values: [key: replacement]), verify: .none)
                    changed = true
                    canRoute = true
                } catch {
                    // A custom setter may throw after writing. Refresh before invalidating traces.
                    changed = true
                    try owner.update(modules: ModuleChildren(), verify: .none)
                    if error is CancellationError { throw error }
                    // Keep coherent replacements, but use stock routing for this request.
                    return false
                }
            }
        }
        return canRoute
    }

    private static func directReplacementValue(_ children: NestedItem<String, Module>) -> Any? {
        switch children {
        case .value(let module):
            return module
        case .array(let children):
            var modules = [Module]()
            for child in children {
                guard case .value(let module) = child else { return nil }
                modules.append(module)
            }
            return modules
        case .dictionary(let children):
            var modules = [String: Module]()
            for (key, child) in children {
                guard case .value(let module) = child else { return nil }
                modules[key] = module
            }
            return modules
        case .none:
            return nil
        }
    }

    private static func replacementProperties(of owner: Module) -> [String:
        any PrefillModuleProperty]
    {
        var properties = [String: any PrefillModuleProperty]()
        var mirror: Mirror? = Mirror(reflecting: owner)
        while let current = mirror, current.subjectType != Module.self {
            for child in current.children {
                if let property = child.value as? any PrefillModuleProperty,
                    let (key, _) = ModuleValue.fromMirror(child)
                {
                    properties[key] = property
                }
            }
            mirror = current.superclassMirror
        }
        return properties
    }
}

/// Keeps packed parameters; each forward selects its route using the current input.
final class PrefillQuantizedLinear: QuantizedLinear {
    init(_ linear: QuantizedLinear) {
        super.init(
            weight: linear.weight, bias: linear.bias, scales: linear.scales,
            biases: linear.biases, groupSize: linear.groupSize, bits: linear.bits,
            mode: linear.mode, globalScale: linear.globalScale)
        freeze(keys: Array(linear.noGrad()))
        train(linear.training)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let decision = QuantizedPrefill.decision(self, input: x)
        QuantizedPrefill.observer?(decision)
        guard decision == .dense else { return super.callAsFunction(x) }
        return QuantizedPrefill.dense(self, x)
    }
}
