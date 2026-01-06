// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

public func loadTokenizer(configuration: ModelConfiguration, hub: HubApi) async throws -> Tokenizer
{
    switch configuration.id {
    case .id(let id, let revision):
        return try await AutoTokenizer.from(
            pretrained: configuration.tokenizerId ?? id,
            hubApi: hub,
            revision: revision
        )
    case .directory(let directory):
        return try await AutoTokenizer.from(modelFolder: directory, hubApi: hub)
    }
}
