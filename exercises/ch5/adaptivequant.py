import numpy as np
import cv2


# def compute_gradient_blockwise(img_y, blk_size=8):
#     """
#     Compute the mean Sobel gradient for every block.
#     :param img_y: single channel Y
#     :return: grad_map，shape=(H//blk, W//blk)
#     """
#     img_y = img_y.astype(np.float32) / 255.0
#     sobel_x = cv2.Sobel(img_y, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_y = cv2.Sobel(img_y, cv2.CV_64F, 0, 1, ksize=3)
#     magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
#
#     H, W = img_y.shape
#     h_blk, w_blk = H // blk_size, W // blk_size
#     grad_map = np.zeros((h_blk, w_blk))
#
#     for i in range(h_blk):
#         for j in range(w_blk):
#             block = magnitude[i*blk_size:(i+1)*blk_size, j*blk_size:(j+1)*blk_size]
#             grad_map[i, j] = np.mean(block)
#
#     norm_grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)
#     norm_grad_map = norm_grad_map ** 2
#
#     return norm_grad_map

import numpy as np
import cv2

def compute_importance_map(img_y, blk_size=8, alpha=0.7, beta=1.0):
    """
    结合 Sobel 梯度与局部方差，生成每个 block 的 importance map。
    :param img_y: 单通道亮度图像 (0-255)
    :param blk_size: block 大小
    :param alpha: 梯度与方差的加权比，alpha 越大表示更依赖边缘信息
    :param beta: 非线性调节指数，beta > 1 增强响应差异
    :return: importance_map (归一化到 [0, 1])
    """
    img_y = img_y.astype(np.float32) / 255.0

    # Sobel 梯度幅度
    sobel_x = cv2.Sobel(img_y, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_y, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)

    # 局部方差计算
    mean = cv2.blur(img_y, (blk_size, blk_size))
    mean_sq = cv2.blur(img_y**2, (blk_size, blk_size))
    local_variance = mean_sq - mean**2
    local_variance = np.clip(local_variance, 0, None)

    # 按 block 计算平均梯度和方差
    H, W = img_y.shape
    h_blk, w_blk = H // blk_size, W // blk_size
    grad_block = np.zeros((h_blk, w_blk))
    var_block = np.zeros((h_blk, w_blk))

    for i in range(h_blk):
        for j in range(w_blk):
            y0, y1 = i * blk_size, (i + 1) * blk_size
            x0, x1 = j * blk_size, (j + 1) * blk_size
            grad_block[i, j] = np.mean(gradient[y0:y1, x0:x1])
            var_block[i, j] = np.mean(local_variance[y0:y1, x0:x1])

    # 标准化并融合
    grad_norm = (grad_block - grad_block.min()) / (grad_block.max() - grad_block.min() + 1e-8)
    var_norm = (var_block - var_block.min()) / (var_block.max() - var_block.min() + 1e-8)
    importance_map = alpha * grad_norm + (1 - alpha) * var_norm

    # 非线性增强
    importance_map = importance_map ** beta

    # 最终归一化
    importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
    return importance_map



def adaptive_quantize(dct_patches, img_shape, q_map, quantizer):
    """
    Apply adaptive quantization to DCT patches based on gradient information.
    Args:
        dct_patches: Input DCT patches with shape [num_patches, patch_size, patch_size]
        img_shape: Original image shape (H, W, C) for determining block positions
    Returns:
        Quantized patches with same shape as input
    """
    quantized_patches = np.zeros_like(dct_patches)
    num_channels = img_shape[2]  # Typically 3 for YCbCr
    blocks_per_row = img_shape[1] // 8

    for i in range(dct_patches.shape[0]):
        # Determine channel index (0=Y, 1=Cb, 2=Cr)
        channel_idx = i % num_channels
        blk_idx = i // num_channels

        # Calculate block position in the image
        h_idx = blk_idx // blocks_per_row
        w_idx = blk_idx % blocks_per_row

        if channel_idx == 0:  # Y channel
            # Get adaptive quantization scale for this block
            quantizer.quantization_scale = q_map[h_idx, w_idx]

        quantized_patches[i] = quantizer.quantize(dct_patches[i:i + 1])[0]

    return quantized_patches


def adaptive_dequantize(zz_patches, img_shape, q_map, quantizer):
    """
    Apply adaptive dequantization to zigzag-scanned patches.
    Args:
        zz_patches: Input patches after zigzag scan with shape [num_patches, patch_size*patch_size]
        img_shape: Original image shape (H, W, C) for determining block positions

    Returns:
        Dequantized patches with same shape as input
    """
    dequantized_patches = np.zeros_like(zz_patches)
    num_channels = img_shape[2]  # Typically 3 for YCbCr
    blocks_per_row = img_shape[1] // 8

    for i in range(zz_patches.shape[0]):
        # Determine channel index (0=Y, 1=Cb, 2=Cr)
        channel_idx = i % num_channels
        blk_idx = i // num_channels

        # Calculate block position in the image
        h_idx = blk_idx // blocks_per_row
        w_idx = blk_idx % blocks_per_row

        if channel_idx == 0:  # Y channel
            # Get adaptive quantization scale for this block
            quantizer.quantization_scale = q_map[h_idx, w_idx]

        dequantized_patches[i] = quantizer.dequantize(zz_patches[i:i + 1])[0]

    return dequantized_patches



