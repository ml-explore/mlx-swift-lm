// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXGuidedGeneration

@Suite
struct GuidedGenerationDiagnosticSinkTests {

    @Test
    func recordsSignalsInGenerationOrder() {
        let sink = GuidedGenerationDiagnosticSink()
        sink.recordSampledToken(10)
        sink.recordFastForwardToken(20)
        sink.recordSampledToken(11)
        sink.recordTermination(grammarTerminated: true, generatedTokenCount: 3)
        sink.recordBuffer(#"{"name":"x"}"#, incompleteOutput: false)
        sink.recordParse(parsedAsToolCall: true, parsedName: "x")

        #expect(sink.sampledTokenIDs == [10, 11])
        #expect(sink.fastForwardTokenIDs == [20])
        #expect(sink.grammarTerminated == true)
        #expect(sink.generatedTokenCount == 3)
        #expect(sink.finalBuffer == #"{"name":"x"}"#)
        #expect(sink.incompleteOutput == false)
        #expect(sink.parsedAsToolCall == true)
        #expect(sink.parsedName == "x")
    }

    @Test
    func currentIsNilByDefault() {
        #expect(GuidedGenerationDiagnosticSink.current == nil)
    }

    @Test
    func taskLocalBindingIsVisibleWithinScopeOnly() async {
        let sink = GuidedGenerationDiagnosticSink()
        await GuidedGenerationDiagnosticSink.$current.withValue(sink) {
            GuidedGenerationDiagnosticSink.current?.recordSampledToken(42)
        }
        #expect(sink.sampledTokenIDs == [42])
        #expect(GuidedGenerationDiagnosticSink.current == nil)
    }

    @Test
    func configuredEmitHookCancelsTheCallingTask() async {
        let sink = GuidedGenerationDiagnosticSink(cancelAfterEmitCount: 1)
        let wasCancelled = await Task {
            sink.recordEmit()
            return Task.isCancelled
        }.value

        #expect(sink.emitCount == 1)
        #expect(wasCancelled)
    }

    @Test
    func ordinaryEmitHookDoesNotCancelTheCallingTask() async {
        let sink = GuidedGenerationDiagnosticSink()
        let wasCancelled = await Task {
            sink.recordEmit()
            return Task.isCancelled
        }.value

        #expect(sink.emitCount == 1)
        #expect(!wasCancelled)
    }

    @Test
    func configuredReasoningCloseHookCancelsTheCallingTask() async {
        let sink = GuidedGenerationDiagnosticSink(cancelOnToolReasoningClose: true)
        let wasCancelled = await Task {
            sink.recordToolReasoningClose()
            return Task.isCancelled
        }.value

        #expect(sink.toolReasoningCloseCount == 1)
        #expect(wasCancelled)
    }

    @Test
    func ordinaryReasoningCloseHookDoesNotCancelTheCallingTask() async {
        let sink = GuidedGenerationDiagnosticSink()
        let wasCancelled = await Task {
            sink.recordToolReasoningClose()
            return Task.isCancelled
        }.value

        #expect(sink.toolReasoningCloseCount == 1)
        #expect(!wasCancelled)
    }
}
