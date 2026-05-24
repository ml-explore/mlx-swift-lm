import Foundation
import MLXLMServer

do {
    switch try MLXServerCLI.parse() {
    case .help:
        print(MLXServerCLI.help)
    case .listRoutes:
        let data = try JSONEncoder.openAIServer.encode(MLXServerRoute.manifest)
        print(String(decoding: data, as: UTF8.self))
    case .run(let configuration):
        try await MLXServer.run(configuration: configuration)
    }
} catch {
    FileHandle.standardError.write(Data("mlx-server: \(error.localizedDescription)\n".utf8))
    Foundation.exit(1)
}
