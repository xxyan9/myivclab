import numpy as np
from ivclab.utils import calc_mse

# Initial the mapping from index to mv
index_matrix = np.arange(1, 82).reshape((9, 9))
index_to_mv = {index_matrix[i, j]: (i - 4, j - 4) for i in range(9) for j in range(9)}

def mv_index_to_offset(index):
    return index_to_mv[index]

def block_mode_decision(
        curr_ycbcr, ref_ycbcr, motion_comp, motion_huffman,
        q_scale, intra_codec, residual_codec, base_lambda=8):
    H, W, C = curr_ycbcr.shape
    recon_ycbcr = np.zeros_like(curr_ycbcr)
    total_bits = 0
    skip_count = inter_count = intra_count = 0
    if q_scale < 0.3:
        lambda_rd = base_lambda * (q_scale ** 2)
    elif q_scale < 0.9:
        lambda_rd = base_lambda * (q_scale ** 0.7)
    else:
        lambda_rd = base_lambda * (q_scale ** 1.5)

    # mv_list = list()

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block_current = curr_ycbcr[i:i+8, j:j+8, :]
            block_ref = ref_ycbcr[i:i+8, j:j+8, :]

            # Mode 0: Skip (copy from reference)
            pred_skip = block_ref
            dist0 = calc_mse(block_current, pred_skip)

            skip_threshold = 5  # 可调参数，建议1~3
            if dist0 < skip_threshold:
                recon_ycbcr[i:i + 8, j:j + 8, :] = block_ref
                total_bits += 0  # skip 无需编码
                skip_count += 1
                continue  # Skip 优先，直接跳过其它模式

            # Mode 1: Inter + Residual
            mv, mv_index = motion_comp.compute_block_mv(ref_ycbcr[..., 0], block_current[..., 0], (i, j))

            mv_dy, mv_dx = mv_index_to_offset(mv_index)

            pred_block = motion_comp.reconstruct_block_with_mv(ref_ycbcr, (i, j), (mv_dy, mv_dx))

            residual1 = block_current - pred_block

            residual_stream, residual_bitsize = residual_codec.intra_encode(
                residual1, return_bpp=True, is_source_rgb=False, is_block_residual=True)

            recon_residual = residual_codec.intra_decode(
                residual_stream, residual1.shape, is_block_residual=True)

            mv_bits, _ = motion_huffman.encode([mv_index])
            mv_decoded = motion_huffman.decode(mv_bits, message_length=1)
            dy, dx = mv_index_to_offset(mv_decoded[0])

            mv_bitsize = 8

            # mv_list.append(mv_bitsize)

            pred_block_decoded = motion_comp.reconstruct_block_with_mv(ref_ycbcr, (i, j), (dy, dx))

            recon1 = pred_block_decoded + recon_residual

            dist1 = calc_mse(block_current, recon1)
            J1 = dist1 + lambda_rd * (mv_bitsize + residual_bitsize)

            # Mode 2: Intra only
            stream2, bits2 = intra_codec.intra_encode(
                block_current, return_bpp=True, is_source_rgb=False)
            recon2 = intra_codec.intra_decode(stream2, block_current.shape)
            dist2 = calc_mse(block_current, recon2)
            # J2 = dist2 + lambda_rd * bits2
            if q_scale > 0.6:
                J2 = dist2 + lambda_rd * (bits2 * 5)
            else:
                J2 = dist2 + lambda_rd * bits2

            # J_all = [J0, J1, J2]
            J_all = [np.inf, J1, J2]  # Mode 0 被 Skip 提前跳过，这里设为 inf
            best_mode = np.argmin(J_all)

            if best_mode == 1:
                recon_ycbcr[i:i+8, j:j+8, :] = recon1
                total_bits += mv_bitsize + residual_bitsize
                inter_count += 1
            else:
                recon_ycbcr[i:i+8, j:j+8, :] = recon2
                total_bits += bits2
                intra_count += 1
    print(f"Skip: {skip_count}, Inter: {inter_count}, Intra: {intra_count}")
    return recon_ycbcr, total_bits


