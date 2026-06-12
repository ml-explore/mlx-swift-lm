// Copyright © 2025 Apple Inc.

#if GuidedGenerationSupport

    import Testing
    import Foundation
    import MLXLMCommon
    import FoundationModels
    @testable import MLXFoundationModels
    @testable import MLXGuidedGeneration

    /// Performance benchmarks for guided generation.
    ///
    /// Measures constrained vs unconstrained throughput, fast-forward token
    /// effectiveness, and grammar compilation time.
    @Suite(.serialized, .timeLimit(.minutes(10)))
    struct GuidedGenerationBenchmarkTests {

        /// Shared prompt used across runs.
        private static let benchmarkPrompt = "Generate a JSON object with a name and age."

        /// Number of timed iterations per configuration.
        private static let iterations = 3

        /// Max tokens for both paths.
        private static let benchmarkMaxTokens = 256

        /// Bounded object schema for benchmarks.
        private static let benchmarkSchema = """
            {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "maxLength": 20 },
                    "active": { "type": "boolean" },
                    "color": { "type": "string", "enum": ["red", "green", "blue"] }
                },
                "required": ["name", "active", "color"],
                "additionalProperties": false
            }
            """

        // MARK: - Constrained vs Unconstrained Throughput

        @Test
        func constrainedVsUnconstrainedThroughput() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

            try await warmup(container: container)

            var unconstrainedRuns: [RunResult] = []
            for _ in 0 ..< Self.iterations {
                let result = try await measureUnconstrained(container: container)
                unconstrainedRuns.append(result)
            }

            var constrainedRuns: [RunResult] = []
            for _ in 0 ..< Self.iterations {
                let result = try await measureConstrained(container: container)
                constrainedRuns.append(result)
            }

            let uMedianTime = median(unconstrainedRuns.map(\.seconds))
            let cMedianTime = median(constrainedRuns.map(\.seconds))
            let uMedianChars = median(unconstrainedRuns.map { Double($0.characterCount) })
            let cMedianChars = median(constrainedRuns.map { Double($0.characterCount) })
            let uMedianEvents = median(unconstrainedRuns.map { Double($0.textDeltaCount) })

            let uCharsPerSec = uMedianChars / uMedianTime
            let cCharsPerSec = cMedianChars / cMedianTime
            let uTokPerSec = uMedianEvents / uMedianTime

            print("")
            print("=== Constrained vs Unconstrained Benchmark ===")
            print("Unconstrained:")
            print("  Median wall time:   \(fmt(uMedianTime)) s")
            print("  Median chars:       \(Int(uMedianChars))")
            print("  Median textDeltas:  \(Int(uMedianEvents))")
            print("  Chars/s:            \(fmt(uCharsPerSec))")
            print("  Events/s (approx tok/s): \(fmt(uTokPerSec))")
            for (i, r) in unconstrainedRuns.enumerated() {
                print(
                    "  Run \(i): \(fmt(r.seconds)) s, \(r.characterCount) chars, \(r.textDeltaCount) events"
                )
            }
            print("Constrained (object schema):")
            print("  Median wall time:   \(fmt(cMedianTime)) s")
            print("  Median chars:       \(Int(cMedianChars))")
            print("  Chars/s:            \(fmt(cCharsPerSec))")
            for (i, r) in constrainedRuns.enumerated() {
                print(
                    "  Run \(i): \(fmt(r.seconds)) s, \(r.characterCount) chars, \(r.textDeltaCount) events"
                )
            }
            print(
                "Wall-time ratio (constrained / unconstrained): \(fmt(cMedianTime / uMedianTime))x")
            print("")

            #expect(uMedianChars > 0, "Unconstrained should produce characters")
            #expect(cMedianChars > 0, "Constrained should produce characters")
        }

        // MARK: - Fast-Forward Effectiveness

        @Test
        func fastForwardEffectiveness() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

            try await warmup(container: container)

            var constrainedRuns: [RunResult] = []
            var unconstrainedRuns: [RunResult] = []

            for _ in 0 ..< Self.iterations {
                let c = try await measureConstrained(container: container)
                constrainedRuns.append(c)
            }

            for _ in 0 ..< Self.iterations {
                let u = try await measureUnconstrained(container: container)
                unconstrainedRuns.append(u)
            }

            let cMedianTime = median(constrainedRuns.map(\.seconds))
            let cMedianChars = median(constrainedRuns.map { Double($0.characterCount) })
            let uMedianTime = median(unconstrainedRuns.map(\.seconds))
            let uMedianEvents = median(unconstrainedRuns.map { Double($0.textDeltaCount) })
            let uMedianChars = median(unconstrainedRuns.map { Double($0.characterCount) })

            let cCharsPerSec = cMedianChars / cMedianTime
            let uCharsPerSec = uMedianChars / uMedianTime
            let uTokPerSec = uMedianEvents / uMedianTime

            print("")
            print("=== Fast-Forward Effectiveness ===")
            print("Constrained (object schema, FF enabled):")
            print("  Median wall time: \(fmt(cMedianTime)) s")
            print("  Median chars:     \(Int(cMedianChars))")
            print("  Chars/s:          \(fmt(cCharsPerSec))")
            print("Unconstrained baseline:")
            print("  Median wall time: \(fmt(uMedianTime)) s")
            print("  Median chars:     \(Int(uMedianChars))")
            print("  Chars/s:          \(fmt(uCharsPerSec))")
            print("  Approx tok/s:     \(fmt(uTokPerSec))")
            print("")
            print("Interpretation:")
            print("  Constrained/Unconstrained wall-time ratio: \(fmt(cMedianTime / uMedianTime))x")
            print("")

            #expect(cMedianChars > 0, "Constrained should produce output")
            #expect(uMedianChars > 0, "Unconstrained should produce output")
        }

        // MARK: - Per-Token Latency Regression Gate
        //
        // Non-functional budget: per-token latency must not regress by more
        // than 5 % against the recorded baseline. Mechanically:
        //
        //   1. Measure `iterations` constrained runs against the bounded
        //      benchmark schema. Take the median wall-clock time and median
        //      character count; derive `perCharSeconds = seconds / chars`
        //      as a stable per-token proxy (character count is fixed by the
        //      schema; token count tracks it tightly for bounded JSON).
        //   2. Read the baseline payload from
        //      `Fixtures/goldens/per_token_baseline.json`. When the file is
        //      missing the test fails with a recording instruction rather
        //      than silently skipping.
        //   3. Compare `measured / baseline`; fail when the ratio exceeds
        //      1.05 (i.e. > 5 % regression). Improvements (ratio < 1.0) pass
        //      unconditionally.
        //
        // ## Recording the baseline
        //
        // Set `RECORD_C17_BASELINE=1` to switch the same test into recorder
        // mode. Recording measures the current backend and writes the
        // resulting JSON to two sinks:
        //
        //   - A `BEGIN_GOLDEN: per_token_baseline.json` /
        //     `END_GOLDEN: per_token_baseline.json` stdout block — the
        //     recovery path on device, where the source tree is read-only.
        //   - A direct write to `Fixtures/goldens/per_token_baseline.json`
        //     via `#filePath` resolution — the happy path on host runs.
        //
        // The recorder mode exits after writing; it does not assert the
        // gate against itself. After recording once, subsequent runs without
        // the env var become the real regression gate.
        //
        // ## Why per-character, not per-token-id
        //
        // `GuidedGenerationLoop.run` does return a generated-token count via
        // its `Int` return value, so we *could* gate on tokens. We stay on
        // characters because the bounded schema used here (name ≤ 20, enum
        // color, boolean active) makes character count deterministic across
        // runs to within a handful of characters and scales linearly with
        // token count. The 5 % budget absorbs the residual noise; the
        // regression gate still fires on any backend-level slowdown that
        // actually matters.

        @Test("per-token latency within ±5 % of recorded baseline")
        func testPerTokenLatencyWithinBudget() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)
            try await warmup(container: container)

            var runs: [RunResult] = []
            for _ in 0 ..< Self.iterations {
                let result = try await measureConstrained(container: container)
                runs.append(result)
            }

            let medianSeconds = median(runs.map(\.seconds))
            let medianChars = median(runs.map { Double($0.characterCount) })
            let perCharSeconds = medianSeconds / max(medianChars, 1.0)

            print("")
            print("=== Per-Token Latency Gate ===")
            print("Measured (median of \(Self.iterations) runs):")
            print("  wall time:    \(fmt(medianSeconds)) s")
            print("  chars:        \(Int(medianChars))")
            print("  per-char:     \(fmt(perCharSeconds * 1000.0)) ms/char")
            for (i, r) in runs.enumerated() {
                print(
                    "  run \(i): \(fmt(r.seconds)) s, \(r.characterCount) chars, \(r.textDeltaCount) events"
                )
            }

            // Recording mode — write the measurement as the new baseline and
            // return without asserting the gate against itself. This is how
            // the baseline fixture is produced.
            if ProcessInfo.processInfo.environment["RECORD_C17_BASELINE"] == "1" {
                try Self.writePerTokenBaseline(
                    medianSeconds: medianSeconds,
                    medianChars: medianChars,
                    perCharSeconds: perCharSeconds,
                    sampleRuns: runs
                )
                return
            }

            // Gate mode — load the baseline and compare. Missing baseline is
            // a first-class failure with a recording instruction rather than
            // a silent skip.
            guard let baseline = Self.loadPerTokenBaseline() else {
                Issue.record(
                    """
                    Per-token latency baseline missing from the test bundle \
                    (resource `per_token_baseline.json` under Fixtures/goldens/).

                    To record the baseline, run the benchmark suite with \
                    RECORD_C17_BASELINE=1 (on device: prefix with TEST_RUNNER_):

                        TEST_RUNNER_RECORD_C17_BASELINE=1 xcodebuild test-without-building \
                            -only-testing:MLXFoundationModelsTests/GuidedGenerationBenchmarkTests ...

                    On device, the write falls back to a BEGIN_GOLDEN / \
                    END_GOLDEN block in the test log — parse it out of the \
                    xcresult and commit the file to Fixtures/goldens/.
                    """
                )
                return
            }

            let ratio = perCharSeconds / baseline.perCharSeconds
            let regressionPercent = (ratio - 1.0) * 100.0

            print(
                "Baseline (recorded): perCharSeconds = \(fmt(baseline.perCharSeconds * 1000.0)) ms/char"
            )
            print("Ratio:               \(fmt(ratio))x (gate ≤ 1.05)")
            print("Δ:                   \(fmt(regressionPercent))%")
            print("")

            #expect(
                ratio <= 1.05,
                """
                Per-token latency regressed \(fmt(regressionPercent))% \
                (ratio \(fmt(ratio))x > 1.05x gate). \
                Baseline: \(fmt(baseline.perCharSeconds * 1000.0)) ms/char; \
                measured: \(fmt(perCharSeconds * 1000.0)) ms/char. \
                If this regression is intentional, re-record the baseline \
                with RECORD_C17_BASELINE=1 and justify in the PR.
                """
            )
        }

        // MARK: - Grammar Compilation Time

        @Test
        func grammarCompilationTime() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let container = try await loadTestModelContainer(id: TestFixtures.defaultModelID)

            let modelID = TestFixtures.defaultModelID
            let (xgTokenizer, hostTokenizer): (GrammarTokenizer, any Tokenizer) =
                try await container.perform { context in
                    let xg = try await MLXLanguageModel.makeXGTokenizer(
                        modelID: modelID,
                        tokenizer: context.tokenizer
                    )
                    return (xg, context.tokenizer)
                }

            let schema = """
                {
                  "type": "object",
                  "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" },
                    "active": { "type": "boolean" }
                  },
                  "required": ["name", "age", "active"],
                  "additionalProperties": false
                }
                """

            let iterations = 5
            var durations: [Duration] = []
            for _ in 0 ..< iterations {
                let start = ContinuousClock.now
                let constraint = try GrammarConstraint(
                    tokenizer: xgTokenizer,
                    jsonSchema: schema,
                    fastForward: true,
                    hostTokenizer: hostTokenizer
                )
                let elapsed = ContinuousClock.now - start
                durations.append(elapsed)
                _ = constraint
            }

            let medianMs = median(durations.map { $0.seconds * 1000.0 })

            print("")
            print("=== Grammar Compilation Time ===")
            for (i, d) in durations.enumerated() {
                print("  Run \(i): \(fmt(d.seconds * 1000.0)) ms")
            }
            print("  Median: \(fmt(medianMs)) ms")
            print("  Target: < 1500ms per compilation")
            print("")

            // 1500ms is generous for this device class (iPhone, iOS 27).
            // The first cold call typically takes ~850ms; steady-state (after
            // JIT/CPU warmup) settles around 500ms. The 1500ms gate catches
            // genuine algorithmic regressions (e.g. grammar-complexity blowup)
            // without being sensitive to device-class or build variation.
            #expect(
                medianMs < 1500.0,
                "Grammar compilation took \(fmt(medianMs)) ms, expected < 1500ms"
            )
        }

        // MARK: - Helpers

        /// Result of a single timed run.
        private struct RunResult {
            let seconds: Double
            let characterCount: Int
            let textDeltaCount: Int
        }

        /// Warm up the model.
        private func warmup(container: ModelContainer) async throws {
            try await container.perform { context in
                let userInput = UserInput(
                    chat: [.user("Hi")],
                    processing: .init()
                )
                let input = try await context.processor.prepare(input: userInput)
                let params = GenerateParameters(maxTokens: 1)
                for await _ in try generate(
                    input: input, parameters: params, context: context
                ) {}
            }
        }

        /// Run a single unconstrained generation and measure it.
        private func measureUnconstrained(
            container: ModelContainer
        ) async throws -> RunResult {
            try await container.perform { context in
                let userInput = UserInput(
                    chat: [.user(Self.benchmarkPrompt)],
                    processing: .init()
                )
                let input = try await context.processor.prepare(input: userInput)
                let params = GenerateParameters(maxTokens: Self.benchmarkMaxTokens)

                var charCount = 0
                var deltaCount = 0
                let start = ContinuousClock.now
                for await generation in try generate(
                    input: input, parameters: params, context: context
                ) {
                    switch generation {
                    case .chunk(let text):
                        charCount += text.count
                        deltaCount += 1
                    case .info, .toolCall:
                        break
                    }
                }
                let elapsed = ContinuousClock.now - start
                return RunResult(
                    seconds: elapsed.seconds,
                    characterCount: charCount,
                    textDeltaCount: deltaCount
                )
            }
        }

        /// Run a single constrained generation (bounded object schema) and measure it.
        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private func measureConstrained(
            container: ModelContainer
        ) async throws -> RunResult {
            try await container.perform { context in
                let xgTokenizer = try await MLXLanguageModel.makeXGTokenizer(
                    modelID: TestFixtures.defaultModelID,
                    tokenizer: context.tokenizer
                )
                let constraint = try GrammarConstraint(
                    tokenizer: xgTokenizer,
                    jsonSchema: Self.benchmarkSchema,
                    fastForward: true,
                    hostTokenizer: context.tokenizer
                )

                let userInput = UserInput(
                    chat: [.user(Self.benchmarkPrompt)],
                    processing: .init()
                )
                let input = try await context.processor.prepare(input: userInput)

                var charCount = 0
                var deltaCount = 0
                let start = ContinuousClock.now
                try GuidedGenerationLoop.run(
                    input: input,
                    context: context,
                    constraint: constraint,
                    maxTokens: Self.benchmarkMaxTokens,
                    vocabSize: Int(xgTokenizer.vocabSize)
                ) { text in
                    charCount += text.count
                    deltaCount += 1
                    return true
                }
                let elapsed = ContinuousClock.now - start
                return RunResult(
                    seconds: elapsed.seconds,
                    characterCount: charCount,
                    textDeltaCount: deltaCount
                )
            }
        }

        /// Median of an array.
        private func median(_ values: [Double]) -> Double {
            let sorted = values.sorted()
            let n = sorted.count
            guard n > 0 else { return 0 }
            if n % 2 == 0 {
                return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            }
            return sorted[n / 2]
        }

        /// Format a Double to 2 decimal places.
        private func fmt(_ value: Double) -> String {
            String(format: "%.2f", value)
        }

        // MARK: - Per-token latency baseline fixture I/O

        /// Decoded per-token latency baseline. `perCharSeconds` is the only
        /// field the gate consumes; the rest exists for provenance when the
        /// fixture is reviewed or diffed.
        private struct PerTokenBaseline {
            let perCharSeconds: Double
            let medianSeconds: Double
            let medianChars: Double
        }

        /// On-disk path for the *recorder* sink, resolved via `#filePath`.
        /// This points at the source tree on the host Mac; it's the right
        /// place for the recorder to write a checked-in fixture. On device,
        /// writes here fail silently (iOS sandbox) — the BEGIN_GOLDEN /
        /// END_GOLDEN stdout block is the recovery path.
        ///
        /// The gate *reads* the baseline through `Bundle.module` instead, so
        /// that device runs find the file inside the test bundle (where the
        /// `.process("Fixtures")` resource declaration in Package.swift
        /// copies it at build time).
        private static let perTokenBaselineSourcePath: URL = {
            let thisFile = URL(fileURLWithPath: #filePath)
            return
                thisFile
                .deletingLastPathComponent()
                .appendingPathComponent("Fixtures", isDirectory: true)
                .appendingPathComponent("goldens", isDirectory: true)
                .appendingPathComponent("per_token_baseline.json", isDirectory: false)
        }()

        /// Loads the baseline fixture from the bundled test resources.
        /// Returns nil when the resource is missing or malformed — the gate
        /// surfaces both cases as the same test failure with a recording
        /// instruction.
        private static func loadPerTokenBaseline() -> PerTokenBaseline? {
            guard
                let url = fixturesBundle.url(
                    forResource: "per_token_baseline",
                    withExtension: "json"
                ),
                let data = try? Data(contentsOf: url),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let perChar = json["perCharSeconds"] as? Double,
                let seconds = json["medianSeconds"] as? Double,
                let chars = json["medianChars"] as? Double
            else {
                return nil
            }
            return PerTokenBaseline(
                perCharSeconds: perChar,
                medianSeconds: seconds,
                medianChars: chars
            )
        }

        /// Writes the baseline payload to two sinks:
        ///
        ///   - `BEGIN_GOLDEN: per_token_baseline.json` /
        ///     `END_GOLDEN:` stdout block for device recovery.
        ///   - Best-effort direct write to the on-disk goldens dir for
        ///     host runs (silently skipped when the path is read-only).
        private static func writePerTokenBaseline(
            medianSeconds: Double,
            medianChars: Double,
            perCharSeconds: Double,
            sampleRuns: [RunResult]
        ) throws {
            let payload: [String: Any] = [
                "modelId": TestFixtures.defaultModelID,
                "schema": Self.benchmarkSchema,
                "prompt": Self.benchmarkPrompt,
                "maxTokens": Self.benchmarkMaxTokens,
                "iterations": Self.iterations,
                "medianSeconds": medianSeconds,
                "medianChars": medianChars,
                "perCharSeconds": perCharSeconds,
                "runs": sampleRuns.map { run -> [String: Any] in
                    [
                        "seconds": run.seconds,
                        "characterCount": run.characterCount,
                        "textDeltaCount": run.textDeltaCount,
                    ]
                },
            ]

            let data = try JSONSerialization.data(
                withJSONObject: payload,
                options: [.prettyPrinted, .sortedKeys]
            )
            guard let text = String(data: data, encoding: .utf8) else {
                Issue.record("per_token_baseline.json JSON was not valid UTF-8")
                return
            }

            print("BEGIN_GOLDEN: per_token_baseline.json")
            print(text)
            print("END_GOLDEN: per_token_baseline.json")

            let dir = perTokenBaselineSourcePath.deletingLastPathComponent()
            try? FileManager.default.createDirectory(
                at: dir,
                withIntermediateDirectories: true
            )
            do {
                try data.write(to: perTokenBaselineSourcePath, options: [.atomic])
                print("[baseline] wrote \(perTokenBaselineSourcePath.path)")
            } catch {
                print("[baseline] on-disk write skipped: \(error)")
            }
        }
    }

    // MARK: - Duration convenience

    extension Duration {
        /// Total seconds as a Double, combining the seconds and attoseconds components.
        fileprivate var seconds: Double {
            Double(components.seconds) + Double(components.attoseconds) / 1e18
        }
    }

#endif
