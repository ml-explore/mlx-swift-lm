import CXGrammar
import Testing

@Suite
struct VersionTests {

    /// Verifies that kXGrammarVersion in shim.cc matches the commit SHA
    /// recorded in Sources/CXGrammar/xgrammar/VERSION, keeping the C
    /// layer honest about which upstream snapshot is vendored.
    @Test
    func testVersionMatchesVendoredSHA() throws {
        let shimVersion = String(cString: xg_version())
        #expect(shimVersion == "d476a48dcd8fa3b5afeddbe850e73bb3b1dcf505")
    }
}
