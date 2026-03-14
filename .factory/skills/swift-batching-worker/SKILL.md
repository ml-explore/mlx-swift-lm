---
name: swift-batching-worker
description: Implements continuous batching infrastructure, scheduler, prompt cache, model updates, and example app for mlx-swift-lm
---

# Swift Batching Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for all features in the continuous batching mission:
- BatchKVCache and batch masking infrastructure
- BatchTokenIterator (batch generation engine)
- InferenceScheduler with single-to-batch upgrade
- LRU prompt cache
- Model RoPE migration (applyRotaryPosition)
- Example app batch subcommand

## Reference Materials

Before starting work, read these reference files for domain knowledge:
- `skills/mlx-swift-lm/SKILL.md` — Core mlx-swift-lm skill with API reference
- `skills/mlx-swift-lm/references/kv-cache.md` — KV cache types and patterns
- `skills/mlx-swift-lm/references/generation.md` — Generation API patterns
- `skills/mlx-swift-lm/references/concurrency.md` — Thread safety patterns
- `.factory/library/architecture.md` — Architecture decisions for this mission

For Python reference implementation details, search for `BatchGenerator`, `BatchKVCache`, `LRUPromptCache` in the Python mlx-lm repo (https://github.com/ml-explore/mlx-lm/).

## Work Procedure

### 1. Read Feature Context
- Read the feature description, preconditions, expectedBehavior, and verificationSteps carefully
- Read `.factory/library/architecture.md` for architectural context
- Read relevant existing code files mentioned in preconditions
- Check `.factory/library/` for any accumulated knowledge from previous features

### 2. Write Tests First (TDD — Red Phase)
- Create test file(s) in `Tests/MLXLMTests/` following existing test conventions
- Write failing tests that cover the feature's expectedBehavior
- Tests MUST use mock models and synthetic data — NO model downloads
- For mock models, create minimal `LanguageModel` conforming types that return deterministic outputs
- Run `swift test --filter MLXLMTests` to confirm tests fail (red)
- If tests can't compile yet (new types don't exist), create minimal stubs first

### 3. Implement (Green Phase)
- New batching code goes in `Libraries/MLXLMCommon/Batching/` directory
- Follow existing code conventions (see existing files for style):
  - Use `public` access for API surface, `internal` for implementation details
  - Use Swift naming conventions (camelCase, descriptive names)
  - Match existing patterns for protocols, extensions, and type hierarchy
  - Use `@preconcurrency` and `Sendable` where needed (StrictConcurrency is enabled)
- For model modifications (applyRotaryPosition migration):
  - Change ONLY the RoPE call sites (~4 lines per model)
  - Do NOT restructure model code or change other logic
  - The helper function should be in `Libraries/MLXLMCommon/Batching/BatchPositionedCache.swift`
- Run `swift test --filter MLXLMTests` to confirm tests pass (green)

### 4. Verify
- Run `swift build` to ensure clean compilation
- Run `swift test --filter MLXLMTests` to confirm all tests pass (existing + new)
- For scheduler features: verify StrictConcurrency compliance (no warnings)
- For model migration: run `grep` to verify no old patterns remain
- Manually inspect key code paths for correctness

### 5. Update Library Knowledge
- Add any discovered patterns, gotchas, or decisions to `.factory/library/architecture.md`
- If a feature changes how things work, update the relevant library file

## Key Technical Notes

### BatchKVCache Design
- Left-padding strategy: shorter sequences padded with zeros on the left
- Track per-sequence `leftPadding: MLXArray` and `offset: MLXArray`
- `filter(batchIndices:)` — removes sequences, shifts to reduce padding
- `extend(other:)` — merges batches, right-justifies to longest
- `extract(idx:)` — returns single KVCacheSimple, strips padding
- `merge([KVCache])` — creates batch from individuals
- `makeMask()` — causal mask accounting for left-padding

### BatchPositionedKVCache Protocol
```swift
public protocol BatchPositionedKVCache: KVCache {
    var batchOffset: MLXArray { get }
}

public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?) -> MLXArray {
    if let batchCache = cache as? BatchPositionedKVCache {
        return rope(x, offset: batchCache.batchOffset)
    } else {
        return rope(x, offset: cache?.offset ?? 0)
    }
}
```

### InferenceScheduler
- Swift actor for thread safety
- Single request → TokenIterator (existing path, zero overhead)
- Second request → upgrade: migrate KVCacheSimple to BatchKVCache, start BatchTokenIterator
- `isBatchCompatible()` checks: no images/video, no MambaCache/CacheList, standard KVCacheSimple

### Mock Model for Tests
```swift
class MockLanguageModel: LanguageModel {
    var kvHeads: [Int] { [4] }
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        // Return deterministic logits based on input
        let logits = MLXArray.zeros([1, 1, vocabSize])
        return LMOutput(logits: logits)
    }
    // ... other required methods
}
```

## Example Handoff

```json
{
  "salientSummary": "Implemented BatchKVCache with left-padding, filter, extend, extract, merge, and makeMask operations. Wrote 15 unit tests covering all operations plus edge cases (empty batch, single sequence, round-trip). All tests pass, swift build clean.",
  "whatWasImplemented": "BatchKVCache struct in Libraries/MLXLMCommon/Batching/BatchKVCache.swift with full left-padding-based batching support. Includes filter(batchIndices:), extend(other:), extract(idx:), merge(_:), fromSingle(_:), makeMask(n:), and integration with createCausalMask. Also added BatchKVCacheTests.swift with 15 test cases.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "swift test --filter MLXLMTests",
        "exitCode": 0,
        "observation": "All 45 tests passed (30 existing + 15 new BatchKVCache tests)"
      },
      {
        "command": "swift build",
        "exitCode": 0,
        "observation": "Clean build, no warnings"
      },
      {
        "command": "grep -r 'class BatchKVCache' Libraries/",
        "exitCode": 0,
        "observation": "Found in Libraries/MLXLMCommon/Batching/BatchKVCache.swift"
      }
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {
        "file": "Tests/MLXLMTests/BatchKVCacheTests.swift",
        "cases": [
          {"name": "testInitWithLeftPadding", "verifies": "VAL-CACHE-001"},
          {"name": "testUpdateAdvancesOffset", "verifies": "VAL-CACHE-002"},
          {"name": "testFilterRetainsIndices", "verifies": "VAL-CACHE-003"},
          {"name": "testFilterShiftsPadding", "verifies": "VAL-CACHE-004"},
          {"name": "testExtendMergesBatch", "verifies": "VAL-CACHE-005"}
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Feature depends on batching infrastructure from a previous milestone that doesn't exist yet
- A model has a custom RoPE pattern not covered by `applyRotaryPosition` and needs guidance
- StrictConcurrency produces errors that require architectural decisions
- Existing tests fail for reasons unrelated to the current feature
- The mlx-swift-examples Xcode project requires changes beyond adding Swift files
