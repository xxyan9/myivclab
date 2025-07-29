import numpy as np
import math
from ivclab.utils import calc_mse


def inter_mode_decision(curr_ycbcr, ref_ycbcr, motion_comp, intra_codec, residual_codec, q_scale, base_lambda=0.1):
    H, W, C = curr_ycbcr.shape
    recon_ycbcr = np.zeros_like(curr_ycbcr)
    total_bits = 0
    skip_count = inter_count = intra_count = 0
    # lambda_rd = base_lambda * q_scale
    lambda_rd = base_lambda * (q_scale ** 0.7)
    # lambda_rd = base_lambda * math.log(1 + q_scale)

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block_current = curr_ycbcr[i:i+8, j:j+8, :]
            block_ref = ref_ycbcr[i:i+8, j:j+8, :]

            # Mode 0: Skip (copy from reference)
            pred_skip = block_ref
            dist0 = calc_mse(block_current, pred_skip)
            bits0 = 6
            J0 = dist0 + lambda_rd * bits0

            # Mode 1: Inter + Residual
            mv, pred_block_y, mv_index = motion_comp.compute_block_mv(
                ref_ycbcr[..., 0], block_current[..., 0], (i, j))
            pred_block = np.zeros_like(block_current)
            pred_block[..., 0] = pred_block_y
            pred_block[..., 1:] = ref_ycbcr[i:i+8, j:j+8, 1:]
            residual1 = block_current - pred_block
            residual_stream, residual_bits = residual_codec.intra_encode(
                residual1, return_bpp=True, is_source_rgb=False, is_block_residual=True)
            recon_residual = residual_codec.intra_decode(residual_stream, residual1.shape, is_block_residual=True)
            recon_block = pred_block + recon_residual
            mv_bits = 8  # simplification
            dist1 = calc_mse(block_current, recon_block)
            J1 = dist1 + lambda_rd * (mv_bits + residual_bits)

            # Mode 2: Intra only
            stream2, bits2 = intra_codec.intra_encode(
                block_current, return_bpp=True, is_source_rgb=False)
            recon_intra = intra_codec.intra_decode(stream2, block_current.shape)
            dist2 = calc_mse(block_current, recon_intra)
            J2 = dist2 + lambda_rd * bits2

            J_all = [J0, J1, J2]
            best_mode = np.argmin(J_all)

            if best_mode == 0:
                recon_ycbcr[i:i+8, j:j+8, :] = pred_skip
                total_bits += bits0
                skip_count += 1
            elif best_mode == 1:
                recon_ycbcr[i:i+8, j:j+8, :] = pred_block + residual1
                total_bits += mv_bits + residual_bits
                inter_count += 1
            else:
                recon_ycbcr[i:i+8, j:j+8, :] = recon_intra
                total_bits += bits2
                intra_count += 1

    # print(f"Skip={skip_count}, Inter={inter_count}, Intra={intra_count}")
    return recon_ycbcr, total_bits
