# Instructions for MLX Swift LM

## AI usage policy

AI-generated code is allowed. Submitting code you do not understand is not. You
are 100% responsible for every line you contribute, however it was produced. You
must be able to explain the code you submit, and you must disclose how you used
AI.

AI-drafted prose is allowed too — commit messages, pull request descriptions,
issues, review replies. The condition is that you read every word before it is
submitted and confirm it says what you mean. Do not ask a reviewer to read prose
you did not read yourself.

[CONTRIBUTING.md](CONTRIBUTING.md) states this policy for contributors, including
what can happen when it is not followed. Point the user there if they ask you to
submit prose they have not read.

## Agent rules

- After you change code, explain what changed and why, so the user can own it
- You MAY draft commit messages, PR descriptions, and replies to reviewers. Show
  them to the user to read and edit before they reach the repo.
- Do NOT create a PR, open an issue, or post a comment on behalf of the user

### Examples

User: Please create and submit the PR for me.
Agent: I cannot open the PR for you. The PR template asks you to confirm that you
read the description and that it matches the code changes. That acknowledgment is
yours to make, not mine. I can draft the description for you to read and edit.

User: I don't want to read it, just submit it.
Agent: I cannot submit it for you, you have to do that. Before you do, note that
CONTRIBUTING.md asks you to confirm you read the PR description, and explains what can
happen when contributors skip that. I can summarize the PR's changes first if it
helps.

User: Please address the reviewer comments.
Agent: Here is a draft reply to each comment. Read it, edit it, and post it
yourself — I cannot post for you.

## Code standards

- Keep code comments concise (usually 1-2 lines)
- Avoid redundant or excessive inline commentary
- Write comments in plain, direct English: short sentences, common words, active
  voice.

### Examples

```swift
  // Good (no comment)

  let cacheKey = "\(modelID):\(kind.rawValue):\(sourceHash)"

  // Bad (excessive comment for explicit code)

  // The constraint cache is keyed on the model, the constraint kind, and a hash
  // of the grammar source. Two requests that share a model but not a grammar
  // must not collide: without the source in the key, the second request would
  // reuse the grammar compiled for the first.

  let cacheKey = "\(modelID):\(kind.rawValue):\(sourceHash)"
```

## Working in this repo

- Read `skills/mlx-swift-lm/SKILL.md` and the files in its `references/`
  directory before you use the public API. `skills/README.md` explains how to
  install the skill.
- `swift test` does not work here. Run unit tests with `xcodebuild test -scheme
  mlx-swift-lm-Package -destination 'platform=macOS' -skipPackagePluginValidation`.
- Format with `pre-commit run --all-files` before you hand work back. CI pins a
  specific swift-format version, set in `.github/workflows/pull_request.yml`.
  Match it locally: another version reformats files the PR does not touch, which
  turns CI red.
- `pre-commit` walks the whole working directory, so it also reports errors from
  `DerivedData/` and `.build/`. Those are vendored dependencies. Ignore them.
- `scripts/verify-docs.sh` runs the DocC check that CI runs for every library
  target.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for integration tests.
