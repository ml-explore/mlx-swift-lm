import MLXCXGrammar
import Testing

@Suite
struct VersionTests {

    /// Verifies that kXGrammarVersion in shim.cc matches the upstream
    /// release tag recorded in Libraries/MLXCXGrammar/xgrammar/VERSION. The
    /// vendored snapshot is pinned to the legible tag (v0.1.30) rather
    /// than a bare commit SHA.
    @Test
    func testVersionMatchesVendoredTag() throws {
        let shimVersion = String(cString: xg_version())
        #expect(shimVersion == "v0.1.30")
    }
}
