// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

/// Singleton instance
let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

/// A test Suite that is serialized.  All inner types
/// will be serialized with respect to each other.
///
/// This is important because loading models is very resource
/// intensive and we don't want to try and load them all at once.
/// Test suites that load models should be inner types and can be
/// written like this (this particular pattern lets you write
/// the suite without excessive indentation):
///
/// ```swift
/// extension ModelLoadingSuite {
///     @Suite(.serialized) struct Gemma4EseriesLoadIntegrationTests {}
/// }
///
/// extension ModelLoadingSuite.Gemma4EseriesLoadIntegrationTests {
///
///     @Test func someTestThatLoadsAModel() { ...
/// ```
///
/// Load models through `models` or `TestModelManager.shared` so only one
/// model is in memory at a time.
@Suite(.serialized) struct ModelLoadingSuite {
}
