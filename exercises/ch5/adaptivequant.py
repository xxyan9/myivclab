import numpy as np
import matplotlib.pyplot as plt


def adaptive_quantize(dct_patches, q_map, quantizer):
    """
    dct_patches: ndarray of shape [H', W', C, 8, 8]
    q_map: ndarray of shape [H', W', C]
    quantizer: a PatchQuant instance (will be updated in-place)
    """
    # Reshape quantization table to [[H', W', C, 1, 1] for broadcasting
    scale_map = q_map[..., np.newaxis, np.newaxis]  # shape: [H', W', C, 1, 1]

    scaled_dct = dct_patches / scale_map

    # Set the quantizer with the current scale=1.0 so that quantization_table is used as is for all blocks.
    quantizer.quantization_scale = 1.0
    quantized = quantizer.quantize(scaled_dct)

    return quantized



def adaptive_dequantize(quantized_patches, q_map, quantizer):
    """
    quantized_patches: ndarray of shape [H', W', C, 8, 8]
    q_map: ndarray of shape [H', W', C]
    quantizer: a PatchQuant instance (will be updated in-place)
    """
    quantizer.quantization_scale = 1.0
    dequantized = quantizer.dequantize(quantized_patches)

    scale_map = q_map[..., np.newaxis, np.newaxis]  # [H', W', C, 1, 1]
    restored = dequantized * scale_map

    return restored


def set_q_map(dct_patches, base_q):
    H, W, C, _, _ = dct_patches.shape
    q_map = np.zeros((H, W, C))
    energies = np.zeros_like(q_map)

    for i in range(H):
        for j in range(W):
            for c in range(C):
                patch = dct_patches[i, j, c]
                energy = np.sum(patch ** 2) - patch[0, 0] ** 2
                energies[i, j, c] = energy

    max_energy = np.max(energies) + 1e-8
    norm_energies = (energies / max_energy) ** 0.5

    q_map = np.clip(base_q * (1 - norm_energies), 0.2, 3)
    return q_map

