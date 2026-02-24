// swift-tools-version: 5.12
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mlx-swift-lm",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "MLXLLM",
            targets: ["MLXLLM"]),
        .library(
            name: "MLXVLM",
            targets: ["MLXVLM"]),
        .library(
            name: "MLXLMCommon",
            targets: ["MLXLMCommon"]),
        .library(
            name: "MLXEmbedders",
            targets: ["MLXEmbedders"]),
        .library(
            name: "MLXLMHuggingFace",
            targets: ["MLXLMHuggingFace"]),
        .library(
            name: "MLXEmbeddersHuggingFace",
            targets: ["MLXEmbeddersHuggingFace"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.6")),
        .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers.git", from: "0.1.0"),
        .package(
            url: "https://github.com/DePasqualeOrg/swift-huggingface.git",
            branch: "improve-cache-hit-performance"),
    ],
    targets: [
        .target(
            name: "MLXLLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-tokenizers"),
            ],
            path: "Libraries/MLXLLM",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXVLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-tokenizers"),
            ],
            path: "Libraries/MLXVLM",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-tokenizers"),
            ],
            path: "Libraries/MLXLMCommon",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-tokenizers"),
                .target(name: "MLXLMCommon"),
            ],
            path: "Libraries/MLXEmbedders",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXLMHuggingFace",
            dependencies: [
                "MLXLMCommon",
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Libraries/MLXLMHuggingFace",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXEmbeddersHuggingFace",
            dependencies: [
                "MLXEmbedders",
                "MLXLMHuggingFace",
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-tokenizers"),
            ],
            path: "Libraries/MLXEmbeddersHuggingFace",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "MLXLMTests",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-tokenizers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                "MLXLMCommon",
                "MLXLMHuggingFace",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
            ],
            path: "Tests/MLXLMTests",
            exclude: [
                "README.md"
            ],
            resources: [.process("Resources/1080p_30.mov"), .process("Resources/audio_only.mov")],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "MLXLMIntegrationTests",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-tokenizers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                "MLXLMCommon",
                "MLXLMHuggingFace",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                "MLXEmbeddersHuggingFace",
            ],
            path: "Tests/MLXLMIntegrationTests",
            exclude: [
                "README.md"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "Benchmarks",
            dependencies: [
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                "MLXLMCommon",
                "MLXLMHuggingFace",
                "MLXEmbeddersHuggingFace",
            ],
            path: "Tests/Benchmarks",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    // docc builder
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
