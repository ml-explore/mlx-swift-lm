// shim.cc -- extern "C" interface between Swift and the vendored xgrammar
// C++ source under xgrammar/. Covers TokenizerInfo construction,
// discriminated error statuses, the Grammar::FromJSONSchema wrapper, the
// tokenizer-aware GrammarCompiler path, and GrammarMatcher.
//
// Warning-treatment policy. The CXGrammar SPM target globally suppresses a
// curated set of warnings (`-Wno-unused-parameter`, `-Wno-shadow`,
// `-Wno-sign-compare`, `-Wno-unused-but-set-variable`,
// `-Wno-deprecated-declarations`) because unmodified upstream triggers them.
// Those suppressions must not mask defects in our own shim code. The pragma
// block directly after the #includes re-enables and promotes the first four
// to errors for everything that follows in this translation unit. The
// `deprecated-declarations` path is left as a warning -- it can surface from
// Apple SDK detritus included transitively and is not a correctness signal
// for shim code either way.

#include "xgrammar_c.h"

#include <dlpack/dlpack.h>
#include <xgrammar/compiler.h>
#include <xgrammar/exception.h>
#include <xgrammar/grammar.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

// Keep shim code held to a stricter bar than vendored upstream. See the
// file-level comment above for why these four are promoted.
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wunused-parameter"
#pragma clang diagnostic error "-Wshadow"
#pragma clang diagnostic error "-Wsign-compare"
#pragma clang diagnostic error "-Wunused-but-set-variable"

namespace {
// The pinned upstream commit sha, kept in sync with
// Sources/CXGrammar/xgrammar/VERSION by scripts/sync-xgrammar-source.sh.
constexpr const char kXGrammarVersion[] = "d476a48dcd8fa3b5afeddbe850e73bb3b1dcf505";

xgrammar::VocabType MapVocabType(XGVocabType type) {
    switch (type) {
        case XG_VOCAB_TYPE_RAW:
            return xgrammar::VocabType::RAW;
        case XG_VOCAB_TYPE_BYTE_FALLBACK:
            return xgrammar::VocabType::BYTE_FALLBACK;
        case XG_VOCAB_TYPE_BYTE_LEVEL:
            return xgrammar::VocabType::BYTE_LEVEL;
    }
    return xgrammar::VocabType::RAW;
}

// Thread-local error message buffer surfaced via xg_last_error_message.
// Each WithExceptionBoundary path that caught an xgrammar exception
// overwrites this with the exception's what(); successful paths clear it
// so stale messages don't leak across calls on the same thread.
thread_local std::string g_last_error_message;

// Thread-local jump-forward string buffer. xgrammar returns the forced
// suffix as a std::string by value; the shim stashes it here so the
// extern "C" layer can hand Swift a stable pointer without either
// allocating caller-visible memory or forcing a two-phase query.
// Overwritten on every xg_matcher_find_jump_forward_string call; the
// caller must consume the previous value before the next call on the
// same thread.
thread_local std::string g_jump_forward_buffer;

void ClearLastErrorMessage() { g_last_error_message.clear(); }

void SetLastErrorMessage(const char *what_message) {
    if (what_message == nullptr) {
        g_last_error_message.clear();
    } else {
        g_last_error_message.assign(what_message);
    }
}

// Every xgrammar call can throw. `extern "C"` functions must catch
// everything before returning to Swift -- an uncaught C++ exception
// unwinding through the Swift ABI is undefined behavior on every Apple
// triple we ship. This helper is the single boundary: every shim
// function routes its xgrammar interaction through WithExceptionBoundary
// so there is exactly one catch clause the reviewer has to audit.
//
// Typed xgrammar exceptions with a dedicated XG_ERR_* status are listed
// in kExceptionMappings (single source of truth -- add a new exception
// type = one line). Anything else deriving from std::exception
// (including `LogFatalError`, which xgrammar's XGRAMMAR_CHECK macros
// throw for schema validation failures) maps to the calling function's
// `default_error`, documenting that function's "error domain". The
// bottom-most catch-all clears the buffer and returns XG_ERR_INTERNAL;
// it should only fire for non-std::exception throws, which xgrammar is
// not expected to produce.
struct ExceptionMapping {
    const std::type_info *type;
    XGStatus status;
};

const ExceptionMapping kExceptionMappings[] = {
    {&typeid(xgrammar::InvalidJSONSchemaError), XG_ERR_INVALID_JSON_SCHEMA},
    {&typeid(xgrammar::InvalidStructuralTagError), XG_ERR_INVALID_STRUCTURAL_TAG},
    {&typeid(xgrammar::InvalidJSONError), XG_ERR_INVALID_JSON},
};

XGStatus MapException(const std::exception &e, XGStatus default_error) {
    SetLastErrorMessage(e.what());
    const std::type_info &actual = typeid(e);
    for (const auto &mapping : kExceptionMappings) {
        if (actual == *mapping.type) return mapping.status;
    }
    return default_error;
}

template <typename F>
XGStatus WithExceptionBoundary(XGStatus default_error, F &&body) noexcept {
    try {
        ClearLastErrorMessage();
        return std::forward<F>(body)();
    } catch (const std::exception &e) {
        return MapException(e, default_error);
    } catch (...) {
        ClearLastErrorMessage();
        return XG_ERR_INTERNAL;
    }
}

// Shared scaffolding for every shim function whose contract is
// "consume a schema source string, hand back a heap-allocated opaque
// wrapper, treat any failure as a JSON-schema error". Both
// xg_grammar_from_json_schema (no tokenizer) and xg_compile_json_schema
// (tokenizer-aware) share this shape, and the regex / structural-tag /
// ebnf compile paths follow it too with a different error domain plugged
// in via default_error.
//
// Factory returns an xgrammar value (Grammar / CompiledGrammar / ...)
// by value; `XGWrapper` is the matching opaque struct from this file
// (XGGrammar / XGCompiledGrammar / ...). The factory receives a
// fully-formed std::string so it can pass it into xgrammar by
// const-ref. We delay the std::string construction until inside the
// boundary because it can throw on allocation failure.
template <typename XGWrapper, typename Factory>
XGStatus CompileSchemaInto(
    const char *schema_json,
    XGWrapper **out_wrapper,
    XGStatus default_error,
    Factory &&factory
) {
    if (out_wrapper == nullptr) return XG_ERR_INTERNAL;
    if (schema_json == nullptr) return XG_ERR_INTERNAL;

    return WithExceptionBoundary(default_error, [&]() -> XGStatus {
        *out_wrapper = new XGWrapper{factory(std::string(schema_json))};
        return XG_OK;
    });
}

// Build a DLTensor view over a caller-owned int32 bitmask buffer in
// the exact shape xgrammar's matcher APIs expect: 1-D, CPU, compact,
// dtype from xgrammar::GetBitmaskDLType(). The returned tensor aliases
// both `data` and `shape_storage`; both must outlive every xgrammar
// call that reads or writes through the tensor.
DLTensor MakeBitmaskTensor(int32_t *data, int64_t *shape_storage) {
    DLTensor tensor{};
    tensor.data = data;
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = xgrammar::GetBitmaskDLType();
    tensor.shape = shape_storage;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    return tensor;
}

// Unified rejection handling for xgrammar matcher operations that
// return bool (true = accepted; false = rejected by grammar). Every
// such operation -- AcceptToken today, AcceptString / BatchAcceptToken
// / similar paths added later -- maps the bool the same way, so
// the mapping lives in exactly one place. Callers that also need to
// handle exceptions wrap the call in WithExceptionBoundary; this
// helper is orthogonal.
XGStatus StatusFromAcceptance(bool accepted) {
    return accepted ? XG_OK : XG_ERR_INVALID_ARG;
}
}  // namespace

struct XGTokenizerInfo {
    xgrammar::TokenizerInfo inner;
};

struct XGGrammar {
    xgrammar::Grammar inner;
};

struct XGGrammarCompiler {
    xgrammar::GrammarCompiler inner;
};

struct XGCompiledGrammar {
    xgrammar::CompiledGrammar inner;
};

struct XGMatcher {
    xgrammar::GrammarMatcher inner;
};

extern "C" {

const char *xg_version(void) { return kXGrammarVersion; }

const char *xg_last_error_message(void) {
    if (g_last_error_message.empty()) return nullptr;
    return g_last_error_message.c_str();
}

XGStatus xg_tokenizer_info_new(
    const char *const *vocab,
    size_t vocab_count,
    XGVocabType vocab_type,
    const int32_t *stop_token_ids,
    size_t stop_token_ids_count,
    XGTokenizerInfo **out_info
) {
    // Fast-fail nullptr arg checks stay outside the boundary -- they
    // never throw and keeping them here makes the boundary body a pure
    // xgrammar interaction.
    if (out_info == nullptr) return XG_ERR_INTERNAL;
    if (vocab == nullptr && vocab_count != 0) return XG_ERR_INTERNAL;
    if (stop_token_ids == nullptr && stop_token_ids_count != 0) return XG_ERR_INTERNAL;

    return WithExceptionBoundary(XG_ERR_INTERNAL, [&]() -> XGStatus {
        std::vector<std::string> encoded_vocab;
        encoded_vocab.reserve(vocab_count);
        for (size_t i = 0; i < vocab_count; ++i) {
            const char *entry = vocab[i];
            if (entry == nullptr) {
                return XG_ERR_INTERNAL;
            }
            encoded_vocab.emplace_back(entry);
        }

        std::optional<std::vector<int32_t>> stop_tokens;
        if (stop_token_ids_count > 0) {
            stop_tokens = std::vector<int32_t>(
                stop_token_ids, stop_token_ids + stop_token_ids_count
            );
        }

        xgrammar::TokenizerInfo info(
            encoded_vocab,
            MapVocabType(vocab_type),
            /*vocab_size=*/std::nullopt,
            stop_tokens,
            /*add_prefix_space=*/false
        );

        *out_info = new XGTokenizerInfo{std::move(info)};
        return XG_OK;
    });
}

void xg_tokenizer_info_free(XGTokenizerInfo *info) {
    // `delete nullptr` is well-defined, but guarding makes the intent
    // obvious and documents the null-safety contract in the header.
    if (info == nullptr) return;
    delete info;
}

XGStatus xg_grammar_from_json_schema(
    const char *schema_json,
    XGGrammar **out_grammar
) {
    return CompileSchemaInto(
        schema_json,
        out_grammar,
        XG_ERR_INVALID_JSON_SCHEMA,
        [](const std::string &s) { return xgrammar::Grammar::FromJSONSchema(s); }
    );
}

void xg_grammar_free(XGGrammar *grammar) {
    if (grammar == nullptr) return;
    delete grammar;
}

XGStatus xg_grammar_compiler_new(
    XGTokenizerInfo *tokenizer_info,
    XGGrammarCompiler **out_compiler
) {
    if (out_compiler == nullptr) return XG_ERR_INTERNAL;
    if (tokenizer_info == nullptr) return XG_ERR_INTERNAL;

    return WithExceptionBoundary(XG_ERR_INTERNAL, [&]() -> XGStatus {
        xgrammar::GrammarCompiler compiler(tokenizer_info->inner);
        *out_compiler = new XGGrammarCompiler{std::move(compiler)};
        return XG_OK;
    });
}

void xg_grammar_compiler_free(XGGrammarCompiler *compiler) {
    if (compiler == nullptr) return;
    delete compiler;
}

XGStatus xg_compile_json_schema(
    XGGrammarCompiler *compiler,
    const char *schema_json,
    XGCompiledGrammar **out_compiled
) {
    if (compiler == nullptr) return XG_ERR_INTERNAL;
    return CompileSchemaInto(
        schema_json,
        out_compiled,
        XG_ERR_INVALID_JSON_SCHEMA,
        [&](const std::string &s) { return compiler->inner.CompileJSONSchema(s); }
    );
}

void xg_compiled_grammar_free(XGCompiledGrammar *compiled) {
    if (compiled == nullptr) return;
    delete compiled;
}

XGStatus xg_compile_grammar_from_ebnf(
    XGGrammarCompiler *compiler,
    const char *ebnf_text,
    const char *root_rule_name,
    XGCompiledGrammar **out_compiled
) {
    if (compiler == nullptr) return XG_ERR_INTERNAL;
    if (ebnf_text == nullptr) return XG_ERR_INTERNAL;
    if (out_compiled == nullptr) return XG_ERR_INTERNAL;

    return WithExceptionBoundary(XG_ERR_INTERNAL, [&]() -> XGStatus {
        std::string ebnf(ebnf_text);
        // Default to xgrammar's built-in "root" if the caller does not
        // override. An empty string is treated as "no override" so Swift
        // callers that pass `nil` via a zero-length C string see the
        // same defaulted behavior as `nullptr`.
        std::string root = (root_rule_name != nullptr && *root_rule_name != '\0')
            ? std::string(root_rule_name)
            : std::string("root");
        xgrammar::Grammar grammar = xgrammar::Grammar::FromEBNF(ebnf, root);
        *out_compiled = new XGCompiledGrammar{compiler->inner.CompileGrammar(grammar)};
        return XG_OK;
    });
}

XGStatus xg_compile_structural_tag(
    XGGrammarCompiler *compiler,
    const char *structural_tag_json,
    XGCompiledGrammar **out_compiled
) {
    if (compiler == nullptr) return XG_ERR_INTERNAL;
    if (structural_tag_json == nullptr) return XG_ERR_INTERNAL;
    if (out_compiled == nullptr) return XG_ERR_INTERNAL;

    return WithExceptionBoundary(XG_ERR_INTERNAL, [&]() -> XGStatus {
        auto result = xgrammar::Grammar::FromStructuralTag(
            std::string(structural_tag_json)
        );
        // FromStructuralTag returns a discriminated union rather than
        // throwing on parse failure. The error arm is itself a
        // `std::variant` over three exception types (InvalidJSONError,
        // InvalidJSONSchemaError, InvalidStructuralTagError); visit it
        // so we pick the right discriminated status for each case,
        // matching how `kExceptionMappings` routes the same types when
        // they throw from the JSON-schema compile path.
        if (std::holds_alternative<xgrammar::StructuralTagError>(result)) {
            const auto &error_variant = std::get<xgrammar::StructuralTagError>(result);
            return std::visit(
                [](const auto &err) -> XGStatus {
                    SetLastErrorMessage(err.what());
                    using E = std::decay_t<decltype(err)>;
                    if constexpr (std::is_same_v<E, xgrammar::InvalidJSONError>) {
                        return XG_ERR_INVALID_JSON;
                    } else if constexpr (std::is_same_v<E, xgrammar::InvalidJSONSchemaError>) {
                        return XG_ERR_INVALID_JSON_SCHEMA;
                    } else {
                        return XG_ERR_INVALID_STRUCTURAL_TAG;
                    }
                },
                error_variant
            );
        }
        xgrammar::Grammar grammar = std::move(std::get<xgrammar::Grammar>(result));
        *out_compiled = new XGCompiledGrammar{compiler->inner.CompileGrammar(grammar)};
        return XG_OK;
    });
}

int32_t xg_bitmask_size(int32_t vocab_size) {
    return xgrammar::GetBitmaskSize(vocab_size);
}

XGStatus xg_matcher_new(
    XGCompiledGrammar *compiled,
    XGMatcher **out_matcher
) {
    if (out_matcher == nullptr) return XG_ERR_INTERNAL;
    if (compiled == nullptr) return XG_ERR_INTERNAL;

    return WithExceptionBoundary(XG_ERR_INTERNAL, [&]() -> XGStatus {
        xgrammar::GrammarMatcher matcher(compiled->inner);
        *out_matcher = new XGMatcher{std::move(matcher)};
        return XG_OK;
    });
}

void xg_matcher_free(XGMatcher *matcher) {
    if (matcher == nullptr) return;
    delete matcher;
}

XGStatus xg_matcher_fill_next_token_bitmask(
    XGMatcher *matcher,
    int32_t *bitmask,
    size_t bitmask_words,
    int32_t vocab_size,
    int32_t *out_needs_apply
) {
    if (matcher == nullptr) return XG_ERR_INTERNAL;
    if (bitmask == nullptr) return XG_ERR_INTERNAL;
    if (vocab_size < 0) return XG_ERR_INTERNAL;

    const int32_t expected_words = xgrammar::GetBitmaskSize(vocab_size);
    if (expected_words < 0) return XG_ERR_INTERNAL;
    if (bitmask_words != static_cast<size_t>(expected_words)) {
        return XG_ERR_INTERNAL;
    }

    return WithExceptionBoundary(XG_ERR_INTERNAL, [&]() -> XGStatus {
        int64_t shape = static_cast<int64_t>(bitmask_words);
        DLTensor tensor = MakeBitmaskTensor(bitmask, &shape);

        bool needs_apply = matcher->inner.FillNextTokenBitmask(&tensor);
        if (out_needs_apply != nullptr) {
            *out_needs_apply = needs_apply ? 1 : 0;
        }
        return XG_OK;
    });
}

XGStatus xg_matcher_accept_token(XGMatcher *matcher, int32_t token_id) {
    if (matcher == nullptr) return XG_ERR_INTERNAL;

    return WithExceptionBoundary(XG_ERR_INTERNAL, [&]() -> XGStatus {
        return StatusFromAcceptance(matcher->inner.AcceptToken(token_id));
    });
}

XGStatus xg_matcher_rollback(XGMatcher *matcher, int32_t num_tokens) {
    if (matcher == nullptr) return XG_ERR_INTERNAL;
    if (num_tokens < 0) return XG_ERR_INTERNAL;

    return WithExceptionBoundary(XG_ERR_INTERNAL, [&]() -> XGStatus {
        matcher->inner.Rollback(static_cast<int>(num_tokens));
        return XG_OK;
    });
}

XGStatus xg_matcher_is_terminated(XGMatcher *matcher, int32_t *out_is_terminated) {
    if (matcher == nullptr) return XG_ERR_INTERNAL;
    if (out_is_terminated == nullptr) return XG_ERR_INTERNAL;

    return WithExceptionBoundary(XG_ERR_INTERNAL, [&]() -> XGStatus {
        *out_is_terminated = matcher->inner.IsTerminated() ? 1 : 0;
        return XG_OK;
    });
}

XGStatus xg_matcher_find_jump_forward_string(
    XGMatcher *matcher,
    const char **out_ptr,
    size_t *out_length
) {
    if (matcher == nullptr) return XG_ERR_INTERNAL;
    if (out_ptr == nullptr) return XG_ERR_INTERNAL;
    if (out_length == nullptr) return XG_ERR_INTERNAL;

    return WithExceptionBoundary(XG_ERR_INTERNAL, [&]() -> XGStatus {
        g_jump_forward_buffer = matcher->inner.FindJumpForwardString();
        *out_ptr = g_jump_forward_buffer.data();
        *out_length = g_jump_forward_buffer.size();
        return XG_OK;
    });
}

XGStatus xg_matcher_fork(XGMatcher *matcher, XGMatcher **out_matcher) {
    if (matcher == nullptr) return XG_ERR_INTERNAL;
    if (out_matcher == nullptr) return XG_ERR_INTERNAL;
    // GrammarMatcher::Fork() was introduced in xgrammar v0.1.34.
    // This build is pinned to v0.1.30 which does not have it.
    SetLastErrorMessage("xg_matcher_fork: Fork() not available in xgrammar v0.1.30");
    return XG_ERR_INTERNAL;
}

}  // extern "C"

#pragma clang diagnostic pop
