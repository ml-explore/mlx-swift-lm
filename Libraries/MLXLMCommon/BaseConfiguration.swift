// Copyright © 2025 Apple Inc.

import Foundation
import MLX

/// The fundamental configuration for any MLX-based model.
///
/// `BaseConfiguration` provides the metadata necessary to identify the model architecture
/// (`modelType`) and describes the quantization parameters used to compress the model's weights.
/// It is designed to be decoded directly from a model repository's `config.json`.
///
/// Typically used the ``GenericModelFactory`` implementations during load.
public struct BaseConfiguration: Codable, Sendable {

    /// The architecture identifier (e.g., "bert", "roberta", "xlm-roberta").
    public let modelType: String

    /// Configuration parameters for weight quantization.
    ///
    /// MLX uses group-wise quantization to reduce memory footprint. This struct
    /// defines how weights are grouped and the precision (bits) used for each group.
    public struct Quantization: Codable, Sendable, Equatable {

        /// Initializes a new quantization configuration.
        /// - Parameters:
        ///   - groupSize: The number of weights that share the same scale and bias.
        ///   - bits: The bit-depth of the quantized weights (e.g., 4 or 8).
        public init(groupSize: Int, bits: Int) {
            self.groupSize = groupSize
            self.bits = bits
        }

        /// Initializes a new quantization configuration with an explicit mode.
        /// - Parameters:
        ///   - groupSize: The number of weights that share the same scale.
        ///   - bits: The bit-depth of the quantized weights.
        ///   - mode: The quantization method, e.g. `.mxfp8`.
        public init(groupSize: Int, bits: Int, mode: QuantizationMode) {
            self.groupSize = groupSize
            self.bits = bits
            self._mode = mode
        }

        /// The size of the quantization group.
        public let groupSize: Int

        /// The number of bits per weight.
        public let bits: Int

        /// Internal storage for the quantization mode.
        private var _mode: QuantizationMode? = nil

        /// The quantization method to use (defaults to `.affine`).
        ///
        /// Affine quantization (asymmetric) uses both a scale and a zero-point
        /// to map floating point values to integers.
        public var mode: QuantizationMode { _mode ?? .affine }

        /// Converts the configuration into a tuple format compatible with `MLX.quantize`.
        public var asTuple: (Int, Int, QuantizationMode) { (groupSize, bits, mode) }

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
            case _mode = "mode"
        }
    }

    /// handling instructions for ``PerLayerQuantization``
    public enum QuantizationOption: Sendable {
        /// Do not quantize this specific layer (keep it in high precision).
        case skip
        /// Quantize this layer using the provided parameters.
        case quantize(Quantization)
    }

    /// A container for per-layer ``Quantization`` settings.
    ///
    /// This allows for "Mixed-Precision" or "Heterogeneous" quantization, where
    /// sensitive layers (like the embedding head) can be kept at higher precision
    /// while the rest of the model is compressed.
    public struct PerLayerQuantization: Sendable {
        /// The default quantization for any layer not explicitly named in `perLayerQuantization`.
        public var quantization: Quantization? = nil

        /// A dictionary mapping layer paths (e.g., "model.embed_tokens") to their quantization options.
        public var perLayerQuantization: [String: QuantizationOption]

        public init(
            quantization: BaseConfiguration.Quantization? = nil,
            perLayerQuantization: [String: BaseConfiguration.QuantizationOption]
        ) {
            self.quantization = quantization
            self.perLayerQuantization = perLayerQuantization
        }

        /// Resolves the quantization parameters for a specific layer.
        /// - Parameter layer: The path/name of the layer.
        /// - Returns: The `Quantization` settings to apply, or `nil` if the layer should be skipped.
        public func quantization(layer: String) -> Quantization? {
            if let perLayer = perLayerQuantization[layer] {
                switch perLayer {
                case .skip:
                    return nil
                case .quantize(let quantization):
                    return quantization
                }
            } else {
                return quantization
            }
        }
    }

    /// An internal container designed to handle the mixed JSON structure found in `config.json`.
    ///
    /// ```
    /// "quantization": {
    ///     "group_size": 64,
    ///     "bits": 4,
    ///     "model.embed_tokens": {
    ///         "group_size": 32,
    ///         "bits": 4
    ///     },
    ///     "model.layers.0.self_attn.q_norm": false,
    /// ```
    ///
    /// Quantization configs in MLX often interleave global keys (like `bits`) with
    /// specific layer keys (like `model.layers.0...`). This container uses manual
    /// decoding to separate these interleaved values.
    struct QuantizationContainer: Codable, Sendable {
        var quantization: Quantization
        var perLayerQuantization: PerLayerQuantization

        init(quantization: Quantization) {
            self.quantization = quantization
            self.perLayerQuantization = PerLayerQuantization(
                quantization: quantization, perLayerQuantization: [:])
        }

        /// A custom CodingKey used to iterate over arbitrary layer names in JSON.
        internal struct _DictionaryCodingKey: CodingKey {
            internal let stringValue: String
            internal let intValue: Int?

            internal init(stringValue: String) {
                self.stringValue = stringValue
                self.intValue = Int(stringValue)
            }

            internal init(intValue: Int) {
                self.stringValue = "\(intValue)"
                self.intValue = intValue
            }
        }

        init(from decoder: any Decoder) throws {
            // handle the embedded Quantization
            self.quantization = try Quantization(from: decoder)

            // and the interleaved per-layer values
            var perLayerQuantization = [String: QuantizationOption]()
            let container = try decoder.container(keyedBy: _DictionaryCodingKey.self)
            for key in container.allKeys {
                switch key.stringValue {
                case Quantization.CodingKeys.groupSize.rawValue: continue
                case Quantization.CodingKeys.bits.rawValue: continue
                case Quantization.CodingKeys._mode.rawValue: continue

                // additional keys that are not layer instructions, see
                // mlx-community/bitnet-b1.58-2B-4T-4bit
                case "quant_method", "linear_class", "quantization_mode": continue

                default:
                    // If the value is a boolean 'false', we treat it as .skip
                    if let f = try? container.decode(Bool.self, forKey: key) {
                        if !f {
                            perLayerQuantization[key.stringValue] = .skip
                        }
                    } else {
                        // Otherwise, try to decode a specific Quantization object for this layer
                        perLayerQuantization[key.stringValue] = .quantize(
                            try container.decode(Quantization.self, forKey: key))
                    }
                }
            }
            self.perLayerQuantization = PerLayerQuantization(
                quantization: quantization, perLayerQuantization: perLayerQuantization)
        }

        func encode(to encoder: any Encoder) throws {
            try quantization.encode(to: encoder)

            var container = encoder.container(keyedBy: _DictionaryCodingKey.self)
            for (key, value) in perLayerQuantization.perLayerQuantization {
                switch value {
                case .skip:
                    try container.encode(false, forKey: .init(stringValue: key))
                case .quantize(let q):
                    try container.encode(q, forKey: .init(stringValue: key))
                }
            }
        }
    }

    private struct TextConfiguration: Codable, Sendable {
        var eosTokenIds: IntOrIntArray?

        enum CodingKeys: String, CodingKey {
            case eosTokenIds = "eos_token_id"
        }
    }

    /// An upstream `quantization_config` whose weights MLX loads packed.
    ///
    /// Mapped as mlx-lm's `load_model` does. Other formats map to `nil`: their models
    /// dequantize in `sanitize`, or the loader cannot read them.
    struct UpstreamQuantizationConfig: Decodable {
        var quantMethod: String?
        var scaleFmt: String?
        var weightBlockSize: [Int]?
        var format: String?

        enum CodingKeys: String, CodingKey {
            case quantMethod = "quant_method"
            case scaleFmt = "scale_fmt"
            case weightBlockSize = "weight_block_size"
            case format
        }

        var quantization: Quantization? {
            switch quantMethod {
            case "mxfp4":
                return .init(groupSize: 32, bits: 4, mode: .mxfp4)
            case "mxfp8":
                return .init(groupSize: 32, bits: 8, mode: .mxfp8)
            case "fp8" where scaleFmt == "ue8m0" && weightBlockSize == [32, 32]:
                // Power-of-two scales over 32x32 blocks are mxfp8, so the weights stay packed.
                return .init(groupSize: 32, bits: 8, mode: .mxfp8)
            case "compressed-tensors":
                switch format {
                case "nvfp4-pack-quantized": return .init(groupSize: 16, bits: 4, mode: .nvfp4)
                case "mxfp4-pack-quantized": return .init(groupSize: 32, bits: 4, mode: .mxfp4)
                case "pack-quantized": return .init(groupSize: 32, bits: 4, mode: .affine)
                default: return nil
                }
            default:
                return nil
            }
        }
    }

    private enum UpstreamKeys: String, CodingKey {
        case quantizationConfig = "quantization_config"
        case textConfig = "text_config"
    }

    /// Internal storage for quantization details extracted from `config.json`.
    var quantizationContainer: QuantizationContainer?

    /// Text-model metadata nested by composite model configurations.
    private var textConfiguration: TextConfiguration?

    /// EOS token IDs from config.json. Can be a single Int or an array of Ints.
    public var eosTokenIds: IntOrIntArray?

    /// EOS token IDs declared at either the model root or in `text_config`.
    public var effectiveEOSTokenIds: Set<Int> {
        Set(eosTokenIds?.values ?? textConfiguration?.eosTokenIds?.values ?? [])
    }

    /// The default quantization settings.
    @available(*, deprecated, message: "Please use perLayerQuantization instead")
    public var quantization: Quantization? {
        quantizationContainer?.quantization
    }

    /// The per-layer quantization settings, including the default fallback.
    public var perLayerQuantization: PerLayerQuantization? {
        quantizationContainer?.perLayerQuantization
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantizationContainer = "quantization"
        case textConfiguration = "text_config"
        case eosTokenIds = "eos_token_id"
    }

    /// Without a `quantization` entry, the default comes from `quantization_config`, at the top
    /// level or in `text_config`.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        quantizationContainer = try container.decodeIfPresent(
            QuantizationContainer.self, forKey: .quantizationContainer)
        textConfiguration = try container.decodeIfPresent(
            TextConfiguration.self, forKey: .textConfiguration)
        eosTokenIds = try container.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenIds)

        if quantizationContainer == nil {
            let upstream = try decoder.container(keyedBy: UpstreamKeys.self)
            let config =
                (try? upstream.decodeIfPresent(
                    UpstreamQuantizationConfig.self, forKey: .quantizationConfig))
                ?? (try? upstream.nestedContainer(keyedBy: UpstreamKeys.self, forKey: .textConfig)
                    .decodeIfPresent(UpstreamQuantizationConfig.self, forKey: .quantizationConfig))
            if let quantization = config?.quantization {
                quantizationContainer = QuantizationContainer(quantization: quantization)
            }
        }
    }
}
