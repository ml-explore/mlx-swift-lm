//
//  MLXArray+Helper.swift
//  mlx-swift-lm
//
//  Created by Christoph Rohde on 24.01.26.
//

import MLX

extension MLXArray {
    
    public static func arange(_ size: Int) -> MLXArray {
        return MLXArray(Array(0 ..< size))
    }
    
}
