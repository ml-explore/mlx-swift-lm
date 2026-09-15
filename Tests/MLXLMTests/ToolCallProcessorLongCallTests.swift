// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

/// `ToolCallProcessor` while collecting a tagged call — on the native path
/// and on the cross-dialect recovery path that declared tools enable — finds
/// the end tag by scanning the chunk plus the tag's overlap with the buffered
/// text, not the whole buffered call: a long call streamed token by token
/// costs time linear in its length, and the tag is still found wherever a
/// chunk boundary falls.
struct ToolCallProcessorLongCallTests {

    private static func call(withContent content: String) -> String {
        "<tool_call>\n<function=write>\n<parameter=content>\n\(content)\n</parameter>\n"
            + "</function>\n</tool_call>"
    }

    @Test("A long tagged call streamed one character at a time parses in linear time")
    func longCallStreamedByCharacterIsLinear() throws {
        let content = String(repeating: "x", count: 40_000)
        let processor = ToolCallProcessor(format: .qwen35)
        let start = DispatchTime.now().uptimeNanoseconds
        for character in Self.call(withContent: content) {
            _ = processor.processChunk(String(character))
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9

        #expect(processor.toolCalls.count == 1)
        let call = try #require(processor.toolCalls.first)
        #expect(call.function.name == "write")
        #expect(call.function.arguments["content"] == .string(content))
        // The whole-buffer scan took over ten seconds here.
        #expect(seconds < 3, "\(seconds) s")
    }

    @Test("A long tagged call streamed with declared tools parses in linear time")
    func longCallWithDeclaredToolsIsLinear() throws {
        let content = String(repeating: "x", count: 40_000)
        let contentSchema: [String: any Sendable] = ["type": "string"]
        let parameters: [String: any Sendable] = [
            "type": "object", "properties": ["content": contentSchema] as [String: any Sendable],
        ]
        let function: [String: any Sendable] = ["name": "write", "parameters": parameters]
        let tools: [[String: any Sendable]] = [["type": "function", "function": function]]
        let processor = ToolCallProcessor(format: .qwen35, tools: tools)
        let start = DispatchTime.now().uptimeNanoseconds
        for character in Self.call(withContent: content) {
            _ = processor.processChunk(String(character))
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9

        #expect(processor.toolCalls.count == 1)
        let call = try #require(processor.toolCalls.first)
        #expect(call.function.name == "write")
        #expect(call.function.arguments["content"] == .string(content))
        #expect(seconds < 3, "\(seconds) s")
    }

    @Test("An end tag split across chunks closes the call and returns the trailing text")
    func endTagSplitAcrossChunksCloses() throws {
        let processor = ToolCallProcessor(format: .qwen35)
        var outputs: [String?] = []
        for chunk in ["<tool_call>\n<function=f>\n</function>\n</tool", "_ca", "ll> after"] {
            outputs.append(processor.processChunk(chunk))
        }

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "f")
        #expect(outputs == [nil, nil, " after"])
    }

    @Test("An end tag arriving in the chunk that completes the start tag closes the call")
    func endTagInTheChunkThatCompletesTheStartTagCloses() throws {
        let processor = ToolCallProcessor(format: .qwen35)
        var outputs: [String?] = []
        for chunk in ["<tool", "_call>\n<function=f>\n</function>\n</tool_call>", " after"] {
            outputs.append(processor.processChunk(chunk))
        }

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "f")
        #expect(outputs == [nil, nil, " after"])
    }

    @Test("An end tag followed by another call in one chunk parses both")
    func endTagFollowedByAnotherCallInOneChunkParsesBoth() throws {
        let processor = ToolCallProcessor(format: .qwen35)
        for chunk in [
            "<tool_call>\n<function=f>\n", "</function>\n</tool_call><tool_call>\n<function=g>\n",
            "</function>\n</tool_call>",
        ] {
            _ = processor.processChunk(chunk)
        }

        #expect(processor.toolCalls.map(\.function.name) == ["f", "g"])
    }
}
