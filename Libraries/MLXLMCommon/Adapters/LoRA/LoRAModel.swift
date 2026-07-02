//
//  LoRAModel.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 03.06.2025.
//

import Foundation
import MLX
import MLXNN

public protocol LoRAModel {

    /// Return the layers to apply LoRA adapters to.
    ///
    /// Typically, this includes all transformer layers.
    /// Must be defined explicitly since we can't unify it across all models.
    var loraLayers: [Module] { get }

    /// Default layer keys to apply LoRA adapters to.
    ///
    /// Used when not specified in `adapter_config.json`.
    /// Otherwise, keys from the config are applied.
    var loraDefaultKeys: [String] { get }
}

extension LoRAModel {

    /// By default we apply LoRA to all Linear layers.
    /// This is aligned with `mlx-lm` Python logic.
    public var loraDefaultKeys: [String] {
        let namedModules = loraLayers.flatMap { $0.namedModules() }
        let linearKeys = namedModules.compactMap { key, module in
            if module is Linear {
                return key
            } else {
                return nil
            }
        }
        let unique = Set(linearKeys)
        return Array(unique)
    }
}

/// A protocol representing a module that includes a LoRA adapter and can be converted
/// back to its original, unadapted form.
public protocol LoRALayer: Module {

    /// Returns a version of the module with the LoRA adapter permanently fused in.
    func fused() -> Module

    /// Returns the original module, without the LoRA adapter applied.
    func reverted() -> Module

    /// When false the layer behaves as the underlying base layer (no LoRA
    /// term added). Used by callers like `linearSpecGenerate` that need to
    /// toggle the adapter between draft and verify phases without unloading
    /// it.
    ///
    /// A default implementation is provided so existing `LoRALayer`
    /// conformers continue to compile unchanged — they appear as
    /// non-toggleable (always-on) layers, and `setLoRAEnabled(_:)` is a
    /// no-op for them. The four built-in implementations (`LoRALinear`,
    /// `QLoRALinear`, `DoRALinear`, `QDoRALinear`) override this with a
    /// stored property to provide the real toggle.
    var loraEnabled: Bool { get set }
}

extension LoRALayer {
    /// Default no-op toggle. Returns true (LoRA always applied) and ignores
    /// writes. Concrete classes that want a real toggle override this with a
    /// stored property — see `LoRALinear` etc.
    public var loraEnabled: Bool {
        get { true }
        set { /* no-op: this conformer doesn't support runtime toggling */  }
    }
}

extension Module {
    /// Walk all submodules and set `loraEnabled` on every `LoRALayer` found.
    ///
    /// This is the generic, always-correct path. For hot loops that toggle
    /// the adapter many times per generation (e.g. speculative decoding),
    /// cache the `[LoRALayer]` list once and call `loraEnabled = enabled`
    /// directly on each — see `NemotronLabsDiffusionModel.setLoRAEnabledFast`.
    public func setLoRAEnabled(_ enabled: Bool) {
        for (_, module) in self.namedModules() {
            if let layer = module as? LoRALayer {
                layer.loraEnabled = enabled
            }
        }
    }
}

/// Default implementation of `reverted()` for `Linear` layers, including support for quantized layers.
extension LoRALayer where Self: Linear {
    public func reverted() -> Module {
        if let quantized = self as? QuantizedLinear {
            return QuantizedLinear(
                weight: quantized.weight, bias: quantized.bias,
                scales: quantized.scales, biases: quantized.biases,
                groupSize: quantized.groupSize, bits: quantized.bits
            )
        } else {
            return Linear(weight: weight, bias: bias)
        }
    }
}

/// Extension for `QuantizedLinear` to provide helper properties.
extension QuantizedLinear {

    /// Computes the dequantized weight matrix using the stored quantization parameters.
    var dequantizedWeight: MLXArray {
        dequantized(
            weight,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}
