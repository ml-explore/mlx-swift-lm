// swift-tools-version: 6.0
import PackageDescription

// Decode throughput of a checkout, built in release. MLX_SWIFT_LM_PATH points the
// benchmark at another checkout, so a change is timed against its base with one build tree.
let library = Context.environment["MLX_SWIFT_LM_PATH"] ?? "../.."

let package = Package(
    name: "DecodeBench",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "mlx-swift-lm", path: library)],
    targets: [
        .executableTarget(
            name: "decodebench",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "BenchmarkHelpers", package: "mlx-swift-lm"),
            ]
        )
    ]
)
