// grammar_functor_wrapper.cc — Unity wrapper for xgrammar/cpp/grammar_functor.cc.
//
// Provides out-of-class definitions for GrammarFSMHasherImpl's static const
// int16_t members. Clang ODR-uses these constants when they are passed to
// variadic function templates (HashCombine) and to std::set::insert, emitting
// relocations against the external symbol. Without out-of-class definitions
// the test-target link fails with "symbol(s) not found" even though the values
// are initialised in-class. (C++17 makes static constexpr members implicitly
// inline, but static const members without constexpr are not inline and still
// require an out-of-class definition when ODR-used.)
//
// The file is compiled in place of grammar_functor.cc (which is listed in the
// CXGrammar target's exclude list) so the translation unit is compiled exactly
// once.

#include "xgrammar/cpp/grammar_functor.cc"  // NOLINT(build/include)

namespace xgrammar {
const int16_t GrammarFSMHasherImpl::kSelfRecursionFlag;
const int16_t GrammarFSMHasherImpl::kSimpleCycleFlag;
const int16_t GrammarFSMHasherImpl::kUnKnownFlag;
}  // namespace xgrammar
