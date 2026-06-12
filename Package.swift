// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
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
            name: "MLXHuggingFace",
            targets: ["MLXHuggingFace"]),
        .library(
            name: "MLXFoundationModels",
            targets: ["MLXFoundationModels"]),
        .library(
            name: "MLXGuidedGeneration",
            targets: ["MLXGuidedGeneration"]),
        .library(
            name: "BenchmarkHelpers",
            targets: ["BenchmarkHelpers"]),
        .library(
            name: "IntegrationTestHelpers",
            targets: ["IntegrationTestHelpers"]),
    ],
    traits: [
        // Gates the MLXLanguageModel adapter for Apple's FoundationModels
        // framework. Default-on. Disabling the trait compiles MLXFoundationModels
        // to an effectively empty library (only MLXDownloadProgress survives):
        // the entire `MLXLanguageModel` / `MLXLanguageModel.Executor` surface
        // requires FoundationModels types that are not available on platforms
        // older than iOS/macOS/visionOS 27.0. Consumers targeting older floors
        // can still use this package for MLXLLM / MLXLMCommon / MLXEmbedders
        // etc. by turning the trait off.
        .trait(
            name: "FoundationModelsIntegration",
            description:
                "Enables the MLXLanguageModel adapter for Apple's FoundationModels framework. Disabling removes the MLXLanguageModel / MLXLanguageModel.Executor types."
        ),
        // Grammar-constrained generation via the vendored xgrammar library.
        // Default-on. Disabling the trait removes MLXFoundationModels's
        // dependency on CXGrammar so consumers who don't need guided
        // generation skip compiling the vendored C++ source tree.
        .trait(
            name: "GuidedGenerationSupport",
            description:
                "Enables grammar-constrained generation via xgrammar. When disabled, MLXFoundationModels still builds and provides chat / tool calling, but guided-output APIs are unavailable."
        ),
        .default(enabledTraits: ["FoundationModelsIntegration", "GuidedGenerationSupport"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/swiftlang/swift-syntax.git", "600.0.0" ..< "604.0.0"),
    ],
    targets: [
        .target(
            name: "MLXLLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLLM",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXVLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXVLM",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLMCommon",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .target(name: "MLXLMCommon"),
            ],
            path: "Libraries/MLXEmbedders",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "BenchmarkHelpers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/BenchmarkHelpers"
        ),
        .target(
            name: "IntegrationTestHelpers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/IntegrationTestHelpers",
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "MLXLMTests",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
            ],
            path: "Tests/MLXLMTests",
            exclude: [
                "README.md"
            ],
            resources: [.process("Resources/1080p_30.mov"), .process("Resources/audio_only.mov")]
        ),
        .macro(
            name: "MLXHuggingFaceMacros",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ],
            path: "Libraries/MLXHuggingFaceMacros"
        ),
        .target(
            name: "MLXHuggingFace",
            dependencies: [
                "MLXHuggingFaceMacros",
                "MLXLMCommon",
            ],
            path: "Libraries/MLXHuggingFace"
        ),
        // C++ bridge for xgrammar: vendored upstream C++17 source under
        // Sources/CXGrammar/xgrammar/ compiled directly by SPM, plus our
        // own shim.cc exposing the extern "C" API from xgrammar_c.h.
        //
        // Refresh the vendored tree with scripts/sync-xgrammar-source.sh.
        // The pinned upstream sha lives in Sources/CXGrammar/xgrammar/VERSION
        // and is mirrored in shim.cc's kXGrammarVersion.
        .target(
            name: "CXGrammar",
            path: "Sources/CXGrammar",
            exclude: [
                // Compiled via Sources/CXGrammar/grammar_functor_wrapper.cc to
                // provide out-of-class definitions for static const members that
                // clang ODR-uses through variadic templates.
                "xgrammar/cpp/grammar_functor.cc"
            ],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("xgrammar/include"),
                .headerSearchPath("xgrammar/cpp"),
                .headerSearchPath("xgrammar/3rdparty/picojson"),
                .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
                .define("XGRAMMAR_ENABLE_CPPTRACE", to: "0"),
                .define("XGRAMMAR_ENABLE_INTERNAL_CHECK", to: "0"),
                // Rename the vendored C++ namespaces at compile time so this
                // target's symbols cannot collide with another xgrammar in the
                // same binary (e.g. CoreAI's prebuilt copy). Token-level
                // substitution: it rewrites bare `xgrammar` / `picojson`
                // identifiers (namespace decls and `::` uses) but not header
                // names, string literals, `XGRAMMAR_*` macros, or `xg_*` tokens.
                .define("xgrammar", to: "mlx_xgrammar"),
                .define("picojson", to: "mlx_picojson"),
                // xgrammar throws -- exceptions must stay enabled.
                .unsafeFlags(["-std=c++17", "-fexceptions"]),
                // Vendored upstream source emits a curated set of warnings
                // under -Wall -Wextra. We silence only the ones produced by
                // unmodified upstream, and only on Apple platforms where
                // we compile.
                .unsafeFlags(
                    [
                        "-Wno-unused-parameter",
                        "-Wno-shadow",
                        "-Wno-sign-compare",
                        "-Wno-deprecated-declarations",
                        "-Wno-unused-but-set-variable",
                    ],
                    .when(platforms: [.macOS, .iOS, .visionOS, .tvOS])
                ),
            ],
            linkerSettings: [
                .linkedLibrary("c++")
            ]
        ),
        // Grammar-constrained ("guided") generation engine built on the
        // vendored xgrammar C++ via CXGrammar. Standalone: depends only on
        // CXGrammar + MLXLMCommon + MLX, with no FoundationModels coupling and
        // no @available floor beyond the package's macOS 14 / iOS 17 minimum.
        .target(
            name: "MLXGuidedGeneration",
            dependencies: [
                "MLXLMCommon",
                "CXGrammar",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/MLXGuidedGeneration"
        ),
        // Bridges Apple's FoundationModels framework to MLX-powered on-device
        // inference. Public surface is gated by @available(macOS 27 / iOS 27 /
        // visionOS 27, *) and #if canImport(FoundationModels), so the target
        // builds on every Xcode that compiles the rest of mlx-swift-lm. The
        // CXGrammar dependency is trait-conditional: with the
        // GuidedGenerationSupport trait disabled, the xgrammar backend is
        // not linked and grammar-constrained generation is unavailable.
        .target(
            name: "MLXFoundationModels",
            dependencies: [
                "MLXLMCommon",
                .target(
                    name: "MLXGuidedGeneration",
                    condition: .when(traits: ["GuidedGenerationSupport"])
                ),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Libraries/MLXFoundationModels"
        ),
        .testTarget(
            name: "MLXFoundationModelsTests",
            dependencies: [
                "MLXFoundationModels",
                "MLXLMCommon",
                .target(
                    name: "MLXGuidedGeneration",
                    condition: .when(traits: ["GuidedGenerationSupport"])
                ),
                // MLXLLM is linked here (not by MLXFoundationModels itself) so its
                // module-init registers a factory with MLXLMCommon's
                // ModelFactoryRegistry. Without it, loadModelContainer throws
                // .noModelFactoryAvailable before ever reaching the downloader,
                // which deadlocks AvailabilityTests' in-flight gate. Model-free:
                // the tests inject a stub downloader — no network, no real weights.
                "MLXLLM",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/MLXFoundationModelsTests"
        ),
        // FM-independent guided-generation tests. Depends only on the engine
        // and MLXLMCommon. No FoundationModels, no direct CXGrammar.
        .testTarget(
            name: "MLXGuidedGenerationTests",
            dependencies: [
                "MLXGuidedGeneration",
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/MLXGuidedGenerationTests"
        ),
        // Direct C-API tests for the CXGrammar shim. No FoundationModels
        // dependency; exercises the vendored xgrammar C++ library through
        // the shim's public C entry points.
        .testTarget(
            name: "CXGrammarTests",
            dependencies: ["CXGrammar"],
            path: "Tests/CXGrammarTests",
            // tokenizer_gemma3.json is read at runtime via a #filePath-relative
            // path (see goldensDirectory in the test sources), not bundled, so
            // the Fixtures tree is excluded from the build graph.
            exclude: ["Fixtures"]
        ),
    ]
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
