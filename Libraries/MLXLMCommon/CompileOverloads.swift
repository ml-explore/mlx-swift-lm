// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// The rest of the stateless `compile` surface, mirroring MLX's own overload
// matrix (1 to 8 inputs, 1 to 4 results). See CompiledTrace.swift for what
// these are for: a `@Sendable` body cannot capture a Module or an MLXArray, so
// weights cannot be frozen into a trace by accident. A shape missing here would
// fall through to MLX's unchecked overload, which is why the matrix is complete
// rather than only the shapes models happen to use today.
//
// Hidden from DocC, as upstream hides the equivalent list.

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray) -> (MLXArray, MLXArray)
) -> @Sendable (MLXArray) -> (MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray) -> (MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray, MLXArray
        )
) -> @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray)
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray
        )
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray, MLXArray
        )
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray)
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray
        )
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray) {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray
        )
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray)
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray, MLXArray
        )
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray, MLXArray, MLXArray
    )
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray
        )
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray)
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray
        )
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray, MLXArray
    )
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray, MLXArray
        )
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray, MLXArray, MLXArray
    )
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
        -> MLXArray
) -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray {
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
        -> (MLXArray, MLXArray)
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray
    )
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
        -> (MLXArray, MLXArray, MLXArray)
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray, MLXArray
    )
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
        -> (MLXArray, MLXArray, MLXArray, MLXArray)
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray, MLXArray, MLXArray
    )
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (
            MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray
        ) -> MLXArray
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) ->
    MLXArray
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (
            MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray
        ) -> (MLXArray, MLXArray)
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) ->
    (MLXArray, MLXArray)
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (
            MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray
        ) -> (MLXArray, MLXArray, MLXArray)
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) ->
    (MLXArray, MLXArray, MLXArray)
{
    MLX.compile(shapeless: shapeless, body)
}

@_documentation(visibility: internal)
public func compile(
    shapeless: Bool = false,
    _ body:
        @escaping @Sendable (
            MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray
        ) -> (MLXArray, MLXArray, MLXArray, MLXArray)
)
    -> @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) ->
    (MLXArray, MLXArray, MLXArray, MLXArray)
{
    MLX.compile(shapeless: shapeless, body)
}
