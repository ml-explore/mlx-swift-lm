// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

extension LLMModelFactory {

    /// Download, quantize, and save a safetensors-backed LLM.
    ///
    /// This is the Swift equivalent of the common `mlx_lm.convert` workflow for models that
    /// are available as compatible safetensors. PyTorch `.bin` conversion is intentionally out
    /// of scope.
    public func convert(
        from downloader: any Downloader,
        configuration: ModelConfiguration,
        to outputDirectory: URL,
        options: ModelConversionOptions = .init(),
        useLatest: Bool = false,
        downloadProgressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
        progressHandler: @Sendable @escaping (ModelConversionProgress) -> Void = { _ in }
    ) async throws -> ModelConversionResult {
        let resolved = try await resolveForModelConversion(
            configuration: configuration,
            from: downloader,
            useLatest: useLatest,
            progressHandler: { progress in
                downloadProgressHandler(progress)
                progressHandler(
                    .init(
                        stage: .downloading,
                        fractionCompleted: progress.fractionCompleted,
                        message: progress.localizedDescription))
            })

        return try await convert(
            configuration: resolved,
            to: outputDirectory,
            options: options,
            progressHandler: progressHandler)
    }

    /// Download, quantize, and save a safetensors-backed LLM.
    public func convert(
        from downloader: any Downloader,
        configuration: ModelConfiguration,
        to outputDirectory: URL,
        bits: Int = 4,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        useLatest: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelConversionResult {
        try await convert(
            from: downloader,
            configuration: configuration,
            to: outputDirectory,
            options: .init(bits: bits, groupSize: groupSize, mode: mode),
            useLatest: useLatest,
            downloadProgressHandler: progressHandler)
    }

    /// Quantize and save a local safetensors-backed LLM.
    public func convert(
        from directory: URL,
        tokenizerDirectory: URL? = nil,
        to outputDirectory: URL,
        options: ModelConversionOptions = .init(),
        progressHandler: @Sendable @escaping (ModelConversionProgress) -> Void = { _ in }
    ) async throws -> ModelConversionResult {
        try await convert(
            configuration: .init(
                modelDirectory: directory,
                tokenizerDirectory: tokenizerDirectory ?? directory,
                name: directory.deletingLastPathComponent().lastPathComponent + "/"
                    + directory.lastPathComponent,
                defaultPrompt: "",
                extraEOSTokens: [],
                eosTokenIds: [],
                toolCallFormat: nil),
            to: outputDirectory,
            options: options,
            progressHandler: progressHandler)
    }

    /// Quantize and save a resolved local safetensors-backed LLM.
    public func convert(
        configuration: ResolvedModelConfiguration,
        to outputDirectory: URL,
        options: ModelConversionOptions = .init(),
        progressHandler: @Sendable @escaping (ModelConversionProgress) -> Void = { _ in }
    ) async throws -> ModelConversionResult {
        let modelDirectory = configuration.modelDirectory
        let configurationURL = modelDirectory.appendingPathComponent("config.json")

        let configData: Data
        do {
            configData = try Data(contentsOf: configurationURL)
        } catch {
            throw ModelFactoryError.configurationFileError(
                configurationURL.lastPathComponent, configuration.name, error)
        }
        if sourceConfigurationIsQuantized(configData) {
            throw ModelConversionError.sourceAlreadyQuantized(modelDirectory)
        }

        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let model: LanguageModel
        do {
            model = try await typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        return try MLXLMCommon.convert(
            modelDirectory: modelDirectory,
            tokenizerDirectory: configuration.tokenizerDirectory,
            model: model,
            to: outputDirectory,
            options: options,
            progressHandler: progressHandler)
    }
}
