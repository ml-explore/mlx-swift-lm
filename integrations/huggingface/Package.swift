// swift-tools-version: 5.12
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-huggingface",
    products: [
    ],
    dependencies: [
        .package(url: "https://github.com/DePasqualeOrg/swift-huggingface-mlx", .branch("main")),
        .package(url: "https://github.com/DePasqualeOrg/swift-transformers-mlx", .branch("main")),
    ],
    targets: [
    ]
)
