// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// An unspecified `reasoningLevel` must resolve through the chat template's own
/// `defaultOn`, and every caller needing a concrete on/off must resolve the same
/// way.
///
/// The regression these guard: the think-then-call gate read the optional as
/// `!= false`, i.e. "unspecified means think", while the prompt rendered
/// `?? defaultOn`. Those agree for every `defaultOn: true` family, so the split
/// was invisible until `ReasoningConfig.gemma4` shipped the first
/// `defaultOn: false` one. Gemma 4's 31B template then prefilled a *closed*
/// empty `<|channel>thought\n<channel|>`, no channel ever opened, and a
/// `.required` tool call silently never arrived.
///
/// RUNTIME REQUIREMENT: `MLXLanguageModel` is `@available(macOS 27.0, ...)`, so
/// every test here needs a macOS 27 *host*. Building against the macOS 27 SDK is
/// not enough - on an older host the `#available` guards return early and the
/// suite reports green without executing a single assertion. That holds for this
/// whole test target, not just this file. Treat a pass on macOS 26 as "compiled",
/// never as "covered".
@Suite
struct ThinkingEnabledResolutionTests {

    @Test func unspecifiedLevelDefersToTemplateDefault() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.Executor.thinkingEnabled(for: nil, defaultOn: true))
        #expect(!MLXLanguageModel.Executor.thinkingEnabled(for: nil, defaultOn: false))
    }

    @Test func explicitLevelOverridesTemplateDefault() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // A concrete level means "think" even where the template defaults off.
        #expect(MLXLanguageModel.Executor.thinkingEnabled(for: .light, defaultOn: false))
        #expect(MLXLanguageModel.Executor.thinkingEnabled(for: .deep, defaultOn: false))
        // ...and `no_think` means off even where the template defaults on.
        #expect(
            !MLXLanguageModel.Executor.thinkingEnabled(for: .custom("no_think"), defaultOn: true))
    }

    /// The gate and the prompt must never disagree: both resolve identically to
    /// what `ReasoningConfig` itself injects into the template.
    @Test func resolutionMatchesReasoningConfigAdditionalContext() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let levels: [ContextOptions.ReasoningLevel?] = [
            nil, .light, .moderate, .deep, .custom("no_think"), .custom("anything"),
        ]
        for defaultOn in [true, false] {
            let strategy = ReasoningPromptStrategy.templateFlag(
                key: "enable_thinking", defaultOn: defaultOn)
            for level in levels {
                let resolved = MLXLanguageModel.Executor.thinkingEnabled(
                    for: level, defaultOn: defaultOn)
                let injected = try strategy.additionalContext(
                    forThinkingEnabled: MLXLanguageModel.Executor.thinkingEnabled(for: level))
                #expect(injected?["enable_thinking"] as? Bool == resolved)
            }
        }
    }

    /// Gemma 4 is the config that exposed the split: its template writes
    /// `enable_thinking | default(false)`.
    @Test func gemma4DefaultsThinkingOff() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        guard
            case .templateFlag(let key, let defaultOn) =
                ReasoningConfig.gemma4.promptStrategy
        else {
            Issue.record("Gemma 4 is expected to use a template flag")
            return
        }
        #expect(key == "enable_thinking")
        #expect(defaultOn == false)
        #expect(!MLXLanguageModel.Executor.thinkingEnabled(for: nil, defaultOn: defaultOn))
    }
}

#endif
