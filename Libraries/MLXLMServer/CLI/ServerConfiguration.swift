// Copyright © 2026 Apple Inc.

import Foundation

public struct MLXServerConfiguration: Sendable, Equatable {
    public var model: String
    public var revision: String
    public var host: String
    public var port: Int
    public var modelType: String?
    public var toolCallParser: String?
    public var reasoningParser: ReasoningParserFormat?
    public var embeddingModel: String?

    public init(
        model: String,
        revision: String = "main",
        host: String = "127.0.0.1",
        port: Int = 8080,
        modelType: String? = nil,
        toolCallParser: String? = nil,
        reasoningParser: ReasoningParserFormat? = nil,
        embeddingModel: String? = nil
    ) {
        self.model = model
        self.revision = revision
        self.host = host
        self.port = port
        self.modelType = modelType
        self.toolCallParser = toolCallParser
        self.reasoningParser = reasoningParser
        self.embeddingModel = embeddingModel
    }
}

public enum MLXServerCLICommand: Sendable, Equatable {
    case run(MLXServerConfiguration)
    case help
    case listRoutes
}

public enum MLXServerCLIError: Error, LocalizedError, Equatable {
    case missingValue(String)
    case invalidPort(String)
    case unknownOption(String)

    public var errorDescription: String? {
        switch self {
        case .missingValue(let option):
            return "Missing value for \(option)"
        case .invalidPort(let value):
            return "Invalid port '\(value)'"
        case .unknownOption(let option):
            return "Unknown option '\(option)'"
        }
    }
}

public enum MLXServerCLI {
    public static func parse(
        arguments: [String] = CommandLine.arguments,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> MLXServerCLICommand {
        var model = environment["MLX_SERVER_MODEL"] ?? "mlx-community/Qwen3-0.6B-4bit"
        var revision = environment["MLX_SERVER_REVISION"] ?? "main"
        var host = environment["MLX_SERVER_HOST"] ?? "127.0.0.1"
        var port = Int(environment["MLX_SERVER_PORT"] ?? "") ?? 8080
        var modelType = environment["MLX_SERVER_MODEL_TYPE"]
        var toolCallParser = environment["MLX_SERVER_TOOL_CALL_PARSER"]
        var embeddingModel = environment["MLX_SERVER_EMBEDDING_MODEL"]
        var reasoningParser: ReasoningParserFormat?
        if let raw = environment["MLX_SERVER_REASONING_PARSER"] {
            reasoningParser = try decodeReasoningParser(raw)
        }

        var index = 1
        while index < arguments.count {
            let option = arguments[index]
            switch option {
            case "-h", "--help":
                return .help
            case "--list-routes":
                return .listRoutes
            case "--model", "-m":
                model = try value(after: option, arguments: arguments, index: &index)
            case "--revision":
                revision = try value(after: option, arguments: arguments, index: &index)
            case "--host":
                host = try value(after: option, arguments: arguments, index: &index)
            case "--port":
                let raw = try value(after: option, arguments: arguments, index: &index)
                guard let parsed = Int(raw) else {
                    throw MLXServerCLIError.invalidPort(raw)
                }
                port = parsed
            case "--model-type":
                modelType = try value(after: option, arguments: arguments, index: &index)
            case "--tool-call-parser":
                toolCallParser = try value(after: option, arguments: arguments, index: &index)
            case "--reasoning-parser":
                reasoningParser = try decodeReasoningParser(
                    try value(after: option, arguments: arguments, index: &index)
                )
            case "--embedding-model":
                embeddingModel = try value(after: option, arguments: arguments, index: &index)
            default:
                if option.hasPrefix("-") {
                    throw MLXServerCLIError.unknownOption(option)
                }
                model = option
            }
            index += 1
        }

        return .run(
            .init(
                model: model,
                revision: revision,
                host: host,
                port: port,
                modelType: modelType,
                toolCallParser: toolCallParser,
                reasoningParser: reasoningParser,
                embeddingModel: embeddingModel
            )
        )
    }

    public static let help = """
        Usage: mlx-server [options] [model-id]

        Options:
          -m, --model <id>                 Hugging Face model id to load
              --revision <revision>        Model revision (default: main)
              --host <host>                Bind host (default: 127.0.0.1)
              --port <port>                Bind port (default: 8080)
              --model-type <type>          Hint for automatic tool-call parser selection
              --tool-call-parser <parser>  auto, json, lfm2, xml_function, glm4, gemma, gemma4, kimi_k2, minimax_m2, mistral, llama3_json
              --reasoning-parser <parser>  none, deepseek_r1, qwen3, harmony
              --embedding-model <id>       Optional embedding model id for /v1/embeddings
              --list-routes                Print the server route manifest
          -h, --help                       Print this help

        Environment:
          MLX_SERVER_MODEL, MLX_SERVER_REVISION, MLX_SERVER_HOST, MLX_SERVER_PORT,
          MLX_SERVER_MODEL_TYPE, MLX_SERVER_TOOL_CALL_PARSER, MLX_SERVER_REASONING_PARSER,
          MLX_SERVER_EMBEDDING_MODEL
        """

    private static func value(
        after option: String,
        arguments: [String],
        index: inout Int
    ) throws -> String {
        let valueIndex = index + 1
        guard valueIndex < arguments.count else {
            throw MLXServerCLIError.missingValue(option)
        }
        index = valueIndex
        return arguments[valueIndex]
    }

    private static func decodeReasoningParser(_ raw: String) throws -> ReasoningParserFormat {
        let data = try JSONEncoder().encode(raw)
        return try JSONDecoder().decode(ReasoningParserFormat.self, from: data)
    }
}
