# Stand-in for the `fast_hadamard_transform` package DeepSeek-V4's reference `model.py` imports
# in `rotate_activation`. Identity: no rotation. In the reference the rotation is applied to both
# the indexer queries and the indexer keys right before FP4 quantization; without that
# quantization it cancels in their dot products, up to rounding.


def hadamard_transform(x, scale=1.0):
    return x.clone()
