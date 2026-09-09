// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// MLX traces a compiled function once. Any MLXArray the body reads but does not
// take as an argument becomes a constant of that trace, so weights read through
// a captured module freeze at the values of the first call. Plain inference
// never notices. An adapter load, fuse, or unload, or a training step, keeps
// producing the old weights.
//
// The `compile` overloads below shadow MLX's stateless ones with a `@Sendable`
// body. `MLXArray` and `Module` are not `Sendable`, so that capture no longer
// builds. Bodies that read weights use `CompiledTrace`, which declares them as
// compile state.
//
// `MLX.compile(inputs:outputs:)` is untouched: declaring captured state there
// is correct, and it remains the way to compile a training step.

/// Compiles a function that reads nothing but its arguments.
///
/// Shadows MLX's `compile` with a `@Sendable` body, which cannot capture an
/// `MLXArray` or a `Module`. Use ``CompiledTrace`` for a body that has to read
/// module weights.
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable ([MLXArray]) -> [MLXArray]
) -> @Sendable ([MLXArray]) -> [MLXArray] {
    MLX.compile(shapeless: shapeless, body)
}

/// Overload of ``compile(shapeless:_:)-(Bool,([MLXArray])->[MLXArray])`` for one input and one
/// output.
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray) -> MLXArray
) -> @Sendable (MLXArray) -> MLXArray {
    MLX.compile(shapeless: shapeless, body)
}

/// Overload of ``compile(shapeless:_:)-(Bool,([MLXArray])->[MLXArray])`` for two inputs and one
/// output.
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    MLX.compile(shapeless: shapeless, body)
}

/// Overload of ``compile(shapeless:_:)-(Bool,([MLXArray])->[MLXArray])`` for three inputs and one
/// output.
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray {
    MLX.compile(shapeless: shapeless, body)
}

/// A compiled trace of a body that reads module weights.
///
/// The declared modules' weights become compile `inputs:`, so every call reads
/// their current values instead of the ones the first call happened to see.
/// Without that, a compiled forward silently ignores a loaded adapter or a
/// training step.
///
/// The body takes its owner as a parameter and is `@Sendable`, so it cannot
/// reach the weights any other way: capturing `self` does not compile.
///
/// ```swift
/// final class MoEBlock: Module, UnaryLayer {
///     // The body stays inside this module, so the default state covers it.
///     private let compiledForward = CompiledTrace<MoEBlock> { block, arguments in
///         [block.forward(arguments[0])]
///     }
///
///     func callAsFunction(_ x: MLXArray) -> MLXArray {
///         compiledForward(self, x)
///     }
/// }
/// ```
///
/// Compilation is lazy, so a module can create its traces in `init` and they
/// still read weights loaded later. Traces never retain their owner.
///
/// A trace is built from the modules present when it first ran, so replacing
/// modules invalidates it: see `Module.invalidateCompiledTraces()`.
public final class CompiledTrace<Owner: Module>: CompiledTraceInvalidating {

    /// The traced body. Receives the owner whose weights it may read.
    public typealias Body = @Sendable (Owner, [MLXArray]) -> [MLXArray]

    /// Returns the modules whose weights the body reads.
    public typealias StateProvider = @Sendable (Owner) -> [Module]

    private let state: StateProvider
    private let body: Body

    // Compiled on first call rather than at init, because weights load later.
    // The lock keeps two concurrent first calls from installing rival traces.
    private let lock = NSLock()
    private var function: (@Sendable ([MLXArray]) -> [MLXArray])?
    private var tracedOwner: ObjectIdentifier?

    /// - Parameters:
    ///   - state: the modules whose weights the body reads; the owner by
    ///     default, which covers a body that stays inside its own subtree
    ///   - body: the function to trace
    public init(state: @escaping StateProvider = { [$0] }, body: @escaping Body) {
        self.state = state
        self.body = body
    }

    /// Whether the trace has been compiled.
    public var isCompiled: Bool {
        lock.withLock { function != nil }
    }

    /// Drops the compiled trace; the next call re-reads the module tree.
    public func invalidate() {
        lock.withLock {
            function = nil
            tracedOwner = nil
        }
    }

    /// Runs the trace, compiling it on first use.
    public func callAsFunction(_ owner: Owner, _ arguments: [MLXArray]) -> [MLXArray] {
        let traced = lock.withLock {
            if let function {
                precondition(
                    tracedOwner == ObjectIdentifier(owner),
                    "a CompiledTrace belongs to the module it first ran on")
                return function
            }
            let function = compiled(owner)
            self.function = function
            self.tracedOwner = ObjectIdentifier(owner)
            return function
        }
        return traced(arguments)
    }

    /// Convenience for a single argument and a single result.
    public func callAsFunction(_ owner: Owner, _ argument: MLXArray) -> MLXArray {
        self(owner, [argument])[0]
    }

    private func compiled(_ owner: Owner) -> @Sendable ([MLXArray]) -> [MLXArray] {
        let body = self.body

        // A snapshot is enough: `update(parameters:)`, the path optimizers and
        // adapter weights take, writes through these same MLXArray objects.
        // Replacing modules makes new ones, which is what
        // `invalidateCompiledTraces()` is for. Re-reading the tree every call
        // would cost more than the trace saves (~35µs per decoder layer).
        let weights = state(owner).flatMap { $0.innerState() }

        // The one place in this package that captures a module in a compiled
        // body: `unowned` because the module stores the trace, and a strong
        // capture would cycle and leak the weights with it.
        return MLX.compile(inputs: [weights]) { [unowned owner] arguments in
            body(owner, arguments)
        }
    }
}

/// Something `Module.invalidateCompiledTraces()` can reset: a ``CompiledTrace``
/// or a container of them.
public protocol CompiledTraceInvalidating: AnyObject {
    func invalidate()
}

extension Module {

    /// Drops every compiled trace stored in this module tree.
    ///
    /// A ``CompiledTrace`` is built from the modules present when it first ran,
    /// so anything that replaces modules leaves it stale: loading, fusing, or
    /// unloading an adapter, quantizing a loaded model. Weight updates that
    /// keep the tree intact, such as training steps, do not need this.
    /// Traces are found by reflection, so storing one is all a module has to
    /// do: there is no conformance to remember.
    public func invalidateCompiledTraces() {
        visit { _, module in
            for value in module.items().flattenedValues() {
                if case .other(let value) = value, let trace = value as? CompiledTraceInvalidating {
                    trace.invalidate()
                }
            }
        }
    }
}
