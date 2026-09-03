// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

package struct InferenceStatePreparationFailure {
    package let modelType: String
    package let error: any Error
}

package struct InferenceStatePreparationReport {
    package let failures: [InferenceStatePreparationFailure]

    package var succeeded: Bool { failures.isEmpty }
}

private let inferenceStateLogger = Logger(
    subsystem: "mlx-swift-lm", category: "inference-state")

/// Prepare a language model after checkpoint loading or an explicit topology
/// update.
///
/// A failed optional optimization is logged and reported while the model
/// remains usable through its unfused path. `BaseLanguageModel` values outside
/// the inference lifecycle, such as rerankers, require no preparation.
@discardableResult
package func prepareInferenceState(
    in model: BaseLanguageModel
) -> InferenceStatePreparationReport {
    guard let languageModel = model as? any LanguageModel else {
        return InferenceStatePreparationReport(failures: [])
    }

    do {
        try languageModel.prepare()
        return InferenceStatePreparationReport(failures: [])
    } catch {
        let modelType = String(reflecting: type(of: languageModel))
        inferenceStateLogger.error(
            "Failed to prepare inference state for \(modelType): \(String(describing: error))")
        return InferenceStatePreparationReport(failures: [
            .init(modelType: modelType, error: error)
        ])
    }
}

/// Prepare derived state and realize a fully loaded model before publication.
///
/// All custom checkpoint loaders should finalize through this function so
/// inference-only optimizations are applied consistently.
///
/// Weights matching `deferredWeightPrefixes` are left lazy, see
/// ``DeferredWeightsProviding``. Everything else is realized here.
@discardableResult
package func materializeModelForInference(
    _ model: BaseLanguageModel,
    deferredWeightPrefixes: [String] = []
) -> InferenceStatePreparationReport {
    let report = prepareInferenceState(in: model)
    if deferredWeightPrefixes.isEmpty {
        eval(model)
    } else {
        let materialized = model.parameters().flattened()
            .compactMap { key, value in
                isDeferredWeight(key, prefixes: deferredWeightPrefixes) ? nil : value
            }
        eval(materialized)
    }
    return report
}
