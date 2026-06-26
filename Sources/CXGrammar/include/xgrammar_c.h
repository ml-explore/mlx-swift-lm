/*
 * xgrammar_c.h -- public C interface exposed by the CXGrammar shim.
 *
 * The Swift bridge imports this header (and nothing from the vendored
 * C++ sources) through the module.modulemap alongside. It covers:
 *   - TokenizerInfo construction / lookup
 *   - GrammarCompiler + JSON schema compilation
 *   - GrammarMatcher: fill_next_token_bitmask, accept_token, is_terminated,
 *     fork, find_jump_forward_string
 *   - discriminated error statuses + xg_last_error_message
 */

#ifndef CXGRAMMAR_XGRAMMAR_C_H
#define CXGRAMMAR_XGRAMMAR_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Returns a pointer to the pinned upstream xgrammar commit sha, matching
 * the contents of Sources/CXGrammar/xgrammar/VERSION. The returned pointer
 * has static storage and must not be freed.
 */
const char *xg_version(void);

/*
 * Opaque handle wrapping `xgrammar::TokenizerInfo`. Construct with
 * xg_tokenizer_info_new; destroy with xg_tokenizer_info_free. Handles are
 * owned by the caller; passing one to another `xg_*` function does not
 * transfer ownership.
 */
typedef struct XGTokenizerInfo XGTokenizerInfo;

/*
 * Status code returned by every fallible shim function. Zero means
 * success; negative values indicate failure. Discriminated per-exception
 * codes:
 *   XG_ERR_INTERNAL                 -- catch-all fallback; no xgrammar
 *                                      exception matched.
 *   XG_ERR_INVALID_ARG              -- caller-supplied argument was
 *                                      rejected by xgrammar (e.g. a
 *                                      matcher rejects a token that
 *                                      the grammar disallows).
 *   XG_ERR_INVALID_JSON             -- xgrammar::InvalidJSONError.
 *   XG_ERR_INVALID_JSON_SCHEMA      -- xgrammar::InvalidJSONSchemaError.
 *   XG_ERR_INVALID_STRUCTURAL_TAG   -- xgrammar::InvalidStructuralTagError.
 * xg_last_error_message() returns a pointer to the failure message
 * recorded on the calling thread; use it to surface xgrammar's
 * `what()` to Swift.
 */
typedef int32_t XGStatus;
#define XG_OK                          ((int32_t)0)
#define XG_ERR_INTERNAL                ((int32_t)-1)
#define XG_ERR_INVALID_ARG             ((int32_t)-2)
#define XG_ERR_INVALID_JSON            ((int32_t)-3)
#define XG_ERR_INVALID_JSON_SCHEMA     ((int32_t)-4)
#define XG_ERR_INVALID_STRUCTURAL_TAG  ((int32_t)-5)

/*
 * Pointer to the last error message recorded on the calling thread,
 * or NULL if no failure has been observed on this thread. The pointer
 * has thread-local storage and remains valid until the next xg_*
 * function call on the same thread. Do not free.
 */
const char *xg_last_error_message(void);

/*
 * Vocabulary encoding, mirrors `xgrammar::VocabType`. RAW treats each
 * vocab string as its literal byte sequence; BYTE_FALLBACK expects the
 * byte-fallback convention used by SentencePiece-style tokenizers
 * (`<0x41>` for byte 0x41); BYTE_LEVEL expects GPT-2-style byte-level
 * encoding.
 */
typedef enum {
    XG_VOCAB_TYPE_RAW = 0,
    XG_VOCAB_TYPE_BYTE_FALLBACK = 1,
    XG_VOCAB_TYPE_BYTE_LEVEL = 2,
} XGVocabType;

/*
 * Construct an `XGTokenizerInfo` from a caller-owned vocab array.
 *
 * `vocab` points to `vocab_count` null-terminated UTF-8 strings. The
 * strings are copied; the array and its contents may be freed after this
 * call returns. `stop_token_ids` is optional — pass NULL with a count of
 * zero to omit; otherwise it points to `stop_token_ids_count` int32 token
 * ids treated as stop tokens. On success, `*out_info` is set to a freshly
 * allocated handle and `XG_OK` is returned. On failure, `*out_info` is
 * left untouched and a negative status is returned.
 */
XGStatus xg_tokenizer_info_new(
    const char *const *vocab,
    size_t vocab_count,
    XGVocabType vocab_type,
    const int32_t *stop_token_ids,
    size_t stop_token_ids_count,
    XGTokenizerInfo **out_info
);

/*
 * Release a handle returned by xg_tokenizer_info_new. Safe to call with
 * a NULL pointer.
 */
void xg_tokenizer_info_free(XGTokenizerInfo *info);

/*
 * Opaque handle wrapping `xgrammar::Grammar`. Construct via
 * `xg_grammar_from_json_schema` (JSON-schema source path); destroy
 * with `xg_grammar_free`.
 */
typedef struct XGGrammar XGGrammar;

/*
 * Compile a JSON-schema source string into an `XGGrammar`. Uses
 * `xgrammar::Grammar::FromJSONSchema` under the hood, which throws
 * `InvalidJSONError` on malformed JSON and `InvalidJSONSchemaError`
 * on a schema that parses but is unsupported or ill-formed. On
 * failure, `*out_grammar` is left untouched, a discriminated status
 * is returned, and the exception `what()` text is copied to the
 * thread-local buffer retrieved via `xg_last_error_message`.
 */
XGStatus xg_grammar_from_json_schema(
    const char *schema_json,
    XGGrammar **out_grammar
);

/*
 * Release a handle returned by xg_grammar_from_json_schema. Safe to
 * call with a NULL pointer.
 */
void xg_grammar_free(XGGrammar *grammar);

/*
 * Opaque handle wrapping `xgrammar::GrammarCompiler`. Binds a
 * tokenizer to a compile cache; every compiled grammar produced by
 * this compiler is bound to the same tokenizer. Construct with
 * `xg_grammar_compiler_new`; destroy with `xg_grammar_compiler_free`.
 * One compiler per tokenizer is sufficient — the compiler caches
 * compiled grammars internally.
 */
typedef struct XGGrammarCompiler XGGrammarCompiler;

/*
 * Opaque handle wrapping `xgrammar::CompiledGrammar`. A grammar that
 * has been compiled against a specific tokenizer and is ready to
 * drive a matcher. Construct via `xg_compile_json_schema` (or the
 * other compile entry points). Destroy with
 * `xg_compiled_grammar_free`.
 */
typedef struct XGCompiledGrammar XGCompiledGrammar;

/*
 * Construct an `XGGrammarCompiler` bound to the given tokenizer.
 *
 * `tokenizer_info` must be a handle returned by
 * `xg_tokenizer_info_new` and must outlive every compiled grammar
 * produced by this compiler. The compiler copies the tokenizer handle
 * internally (xgrammar's PIMPL + shared_ptr semantics) so the caller
 * keeps ownership of the original handle. Defaults mirror upstream:
 * `max_threads=8`, `cache_enabled=true`, `max_memory_bytes=-1`. On
 * success, `*out_compiler` is set to a freshly allocated handle and
 * `XG_OK` is returned; on failure `*out_compiler` is left untouched.
 */
XGStatus xg_grammar_compiler_new(
    XGTokenizerInfo *tokenizer_info,
    XGGrammarCompiler **out_compiler
);

/*
 * Release a handle returned by `xg_grammar_compiler_new`. Safe to
 * call with a NULL pointer. Does not free any `XGCompiledGrammar`
 * handles previously produced by this compiler — those remain valid
 * until individually freed.
 */
void xg_grammar_compiler_free(XGGrammarCompiler *compiler);

/*
 * Compile a JSON-schema source string into an `XGCompiledGrammar`
 * bound to the compiler's tokenizer. Uses
 * `xgrammar::GrammarCompiler::CompileJSONSchema` with upstream
 * defaults (any_whitespace=true, strict_mode=true, indent/separators/
 * max_whitespace unset). On schema failure the thread-local error
 * buffer is populated and a discriminated status (typically
 * `XG_ERR_INVALID_JSON_SCHEMA`) is returned; `*out_compiled` is left
 * untouched.
 */
XGStatus xg_compile_json_schema(
    XGGrammarCompiler *compiler,
    const char *schema_json,
    XGCompiledGrammar **out_compiled
);

/*
 * Release a handle returned by `xg_compile_json_schema`. Safe to call
 * with a NULL pointer.
 */
void xg_compiled_grammar_free(XGCompiledGrammar *compiled);

/*
 * Parse `ebnf_text` as an EBNF (GBNF) grammar and compile it against
 * the compiler's bound tokenizer in one call. Combines
 * `xgrammar::Grammar::FromEBNF(ebnf_text, root_rule_name)` with
 * `GrammarCompiler::CompileGrammar(grammar)` so the shim exposes a
 * single-call entry point parallel to `xg_compile_json_schema`.
 *
 * `root_rule_name` may be NULL or empty; the shim substitutes
 * xgrammar's default of "root". Pass "start" (or any custom rule
 * name) when your grammar uses a non-default top-level production.
 *
 * EBNF parse errors throw `xgrammar::LogFatalError` (not a
 * discriminated typed exception), which falls through the shim's
 * exception table to this call's default error, `XG_ERR_INTERNAL`.
 * The parser's line/column message is captured into the thread-local
 * buffer retrieved via `xg_last_error_message`, which surfaces on the
 * Swift side as `XGError.constraintCompilationFailed`.
 *
 * On success, `*out_compiled` is set to a freshly allocated handle
 * and `XG_OK` is returned. On failure, `*out_compiled` is left
 * untouched.
 */
XGStatus xg_compile_grammar_from_ebnf(
    XGGrammarCompiler *compiler,
    const char *ebnf_text,
    const char *root_rule_name,
    XGCompiledGrammar **out_compiled
);

/*
 * Parse `structural_tag_json` as xgrammar's structural-tag JSON format
 * and compile it against the compiler's bound tokenizer in one call.
 * Combines `xgrammar::Grammar::FromStructuralTag(json, nullopt)` with
 * `GrammarCompiler::CompileGrammar(grammar)` so the shim exposes a
 * single-call entry point parallel to `xg_compile_grammar_from_ebnf`.
 *
 * Used by the Qwen tool-calling pipeline: the wrapped-vs-bare
 * `<tool_call>...</tool_call>` envelope composes as an `or` of a
 * `tag`-wrapped `json_schema` and a bare `json_schema`, sharing the
 * same envelope schema between both arms. Structural tag is xgrammar's
 * first-class API for exactly this multi-format dispatch case; hand-
 * rolled GBNF would have to reimplement the JSON-schema-to-grammar
 * compile that `Grammar::FromJSONSchema` already does internally.
 *
 * Tokenizer info is passed as `nullopt`: the structural-tag body used
 * here contains only `const_string` and `json_schema` formats, neither
 * of which reference token ids or token strings. A future structural-
 * tag body that uses `token`, `token_dispatch`, or `token_triggered_
 * tags` formats will need a variant of this entry point that threads
 * the compiler's bound `TokenizerInfo` through to
 * `FromStructuralTag`'s second argument.
 *
 * Errors map via the shim's discriminated-status path. Malformed
 * structural-tag JSON surfaces as `XG_ERR_INVALID_STRUCTURAL_TAG`
 * (mapped from `xgrammar::InvalidStructuralTagError` in
 * `kExceptionMappings`); any other xgrammar throw falls through to
 * this call's default error of `XG_ERR_INTERNAL`. In both cases the
 * parser's message is captured into the thread-local buffer retrieved
 * via `xg_last_error_message`.
 *
 * On success, `*out_compiled` is set to a freshly allocated handle
 * and `XG_OK` is returned. On failure, `*out_compiled` is left
 * untouched.
 */
XGStatus xg_compile_structural_tag(
    XGGrammarCompiler *compiler,
    const char *structural_tag_json,
    XGCompiledGrammar **out_compiled
);

/*
 * Opaque handle wrapping `xgrammar::GrammarMatcher`. Construct from an
 * `XGCompiledGrammar` with `xg_matcher_new`; destroy with
 * `xg_matcher_free`. A matcher tracks per-session grammar state and
 * advances as tokens are committed.
 */
typedef struct XGMatcher XGMatcher;

/*
 * Return the required bitmask length, in int32 words, for the given
 * vocab size. Matches `xgrammar::GetBitmaskSize`:
 * `(vocab_size + 31) / 32`. Callers size their bitmask buffer with
 * this before calling `xg_matcher_fill_next_token_bitmask`.
 */
int32_t xg_bitmask_size(int32_t vocab_size);

/*
 * Construct an `XGMatcher` from a compiled grammar. The compiled
 * grammar must outlive the matcher (xgrammar uses shared ownership
 * internally, but the C handle remains the caller's to free). Stop
 * token overrides and rollback limits use xgrammar defaults (inherit
 * from tokenizer; unlimited rollback). On success `*out_matcher` is
 * set and `XG_OK` returned; on failure `*out_matcher` is untouched.
 */
XGStatus xg_matcher_new(
    XGCompiledGrammar *compiled,
    XGMatcher **out_matcher
);

/*
 * Release a handle returned by `xg_matcher_new`. Safe to call with a
 * NULL pointer.
 */
void xg_matcher_free(XGMatcher *matcher);

/*
 * Fill `bitmask` with the set of acceptable next tokens at the
 * matcher's current state. The bitmask is LSB-first per int32 word:
 * bit `i` of word `w` corresponds to token `w * 32 + i`.
 *
 * `bitmask` must point to at least `bitmask_words` int32 words, and
 * `bitmask_words` must equal `xg_bitmask_size(vocab_size)`. If not,
 * `XG_ERR_INTERNAL` is returned and the buffer is left untouched.
 *
 * `out_needs_apply` (optional, may be NULL) receives 1 if the mask
 * excludes at least one token (application is required) and 0 if
 * every token is acceptable (the mask can be skipped).
 */
XGStatus xg_matcher_fill_next_token_bitmask(
    XGMatcher *matcher,
    int32_t *bitmask,
    size_t bitmask_words,
    int32_t vocab_size,
    int32_t *out_needs_apply
);

/*
 * Commit a token to the matcher, advancing its state so that the
 * next `xg_matcher_fill_next_token_bitmask` reflects what is
 * acceptable after `token_id`.
 *
 * Returns:
 *   XG_OK              -- token accepted; matcher state advanced.
 *   XG_ERR_INVALID_ARG -- token rejected by the grammar (bit for
 *                         `token_id` was clear in the last bitmask).
 *                         Matcher state is unchanged.
 *   XG_ERR_INTERNAL    -- matcher is NULL, or xgrammar threw an
 *                         unexpected exception (e.g. matcher already
 *                         terminated). `xg_last_error_message`
 *                         returns the `what()` text.
 */
XGStatus xg_matcher_accept_token(XGMatcher *matcher, int32_t token_id);

/*
 * Roll back the most recently accepted `num_tokens` tokens, restoring
 * the matcher to the state it held before those commits. Accepts a
 * zero argument as a no-op.
 *
 * Mirrors `xgrammar::GrammarMatcher::Rollback(num_tokens)`. xgrammar
 * tracks a bounded rollback history sized by the `max_rollback_tokens`
 * construction argument (currently inherited as unlimited at
 * compile_grammar time); rolling back more than the history supports
 * throws an xgrammar internal error which surfaces here as
 * XG_ERR_INTERNAL with `xg_last_error_message()` populated.
 *
 * Return codes:
 *   XG_OK              -- matcher state rewound `num_tokens` steps.
 *   XG_ERR_INTERNAL    -- matcher is NULL, `num_tokens` is negative,
 *                         or xgrammar threw (history exceeded, etc.).
 */
XGStatus xg_matcher_rollback(XGMatcher *matcher, int32_t num_tokens);

/*
 * Query whether the matcher has consumed a stop token and terminated.
 * `*out_is_terminated` is set to 1 when terminated, 0 otherwise. The
 * pointer must be non-NULL; NULL returns XG_ERR_INTERNAL without
 * touching the matcher.
 *
 * This mirrors `xgrammar::GrammarMatcher::IsTerminated()`. It does not
 * include the weaker "root rule completed" state -- a grammar that has
 * reached a complete parse but has not yet accepted the configured stop
 * token is not considered terminated here. (xgrammar's `IsCompleted()`
 * covers that weaker state; it is not exposed here.)
 */
XGStatus xg_matcher_is_terminated(XGMatcher *matcher, int32_t *out_is_terminated);

/*
 * Return the jump-forward string at the matcher's current state —
 * xgrammar's `GrammarMatcher::FindJumpForwardString()`. This is the
 * longest string of characters the grammar currently forces next; the
 * caller tokenizes it through its own tokenizer and advances the
 * matcher token-by-token with `xg_matcher_accept_token`.
 *
 * On success:
 *   - `*out_ptr` points to a thread-local UTF-8 byte buffer owned by
 *     the shim. The pointer remains valid until the next call to
 *     `xg_matcher_find_jump_forward_string` on the same thread.
 *   - `*out_length` is the byte length of the string (excluding any
 *     NUL terminator). Zero means "no jump-forward available".
 * On failure, `*out_ptr` is left untouched and `*out_length` set to 0.
 *
 * Does not change matcher state. Safe to call idempotently.
 *
 * Note on encoding: xgrammar builds the jump-forward string from the
 * grammar's forced prefix, which for JSON-Schema grammars is ASCII
 * structural text. For byte-fallback tokenizers driving non-UTF-8
 * grammars (e.g. raw-bytes EBNF productions), the caller must handle
 * non-UTF-8 bytes itself; the JSON-Schema happy path assumes ASCII/UTF-8.
 */
XGStatus xg_matcher_find_jump_forward_string(
    XGMatcher *matcher,
    const char **out_ptr,
    size_t *out_length
);

/*
 * Deep-copy the matcher's per-session state into a new matcher, which
 * shares the compiled grammar and tokenizer with the original. Mirrors
 * `xgrammar::GrammarMatcher::Fork()`: commits on one matcher do not
 * affect the other, but the underlying compiled grammar is
 * shared — freeing `matcher` after forking does not invalidate the
 * fork, because xgrammar holds the compiled grammar through a
 * `shared_ptr` internally.
 *
 * The returned matcher is owned by the caller and must be released
 * with `xg_matcher_free`. The parent matcher remains valid and is
 * unchanged by this call.
 *
 * On failure `*out_matcher` is left untouched and a negative status
 * is returned.
 */
XGStatus xg_matcher_fork(
    XGMatcher *matcher,
    XGMatcher **out_matcher
);

#ifdef __cplusplus
}
#endif

#endif /* CXGRAMMAR_XGRAMMAR_C_H */
