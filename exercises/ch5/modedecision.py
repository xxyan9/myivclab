import numpy as np
from ivclab.utils import calc_mse

# Initial the mapping from index to mv
index_matrix = np.arange(1, 82).reshape((9, 9))
index_to_mv = {index_matrix[i, j]: (i - 4, j - 4) for i in range(9) for j in range(9)}


def mv_index_to_offset(index):
    return index_to_mv[index]


def block_mode_decision(
        curr_ycbcr, ref_ycbcr, motion_comp, motion_huffman,
        q_scale, intra_codec, residual_codec):
    H, W, C = curr_ycbcr.shape
    recon_ycbcr = np.zeros_like(curr_ycbcr)
    total_bits = 0
    skip_count = inter_count = intra_count = 0
    if q_scale < 0.1:
        lambda_rd = 0.006
    elif q_scale < 0.3:
        lambda_rd = 0.023
    elif q_scale < 0.6:
        lambda_rd = 0.039
    elif q_scale < 1:
        lambda_rd = 0.045
    else:
        lambda_rd = 0.057 * (q_scale ** 2.3)

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block_current = curr_ycbcr[i:i+8, j:j+8, :]
            block_ref = ref_ycbcr[i:i+8, j:j+8, :]

            # Mode 0: Skip (copy from reference)
            dist0 = calc_mse(block_current, block_ref)
            J0 = dist0

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

            mv_bitsize = len(mv_bits)

            pred_block_decoded = motion_comp.reconstruct_block_with_mv(ref_ycbcr, (i, j), (dy, dx))
            recon1 = pred_block_decoded + recon_residual

            dist1 = calc_mse(block_current, recon1)
            J1 = dist1 + lambda_rd * (mv_bitsize + residual_bitsize)

            # Mode 2: Intra only
            stream2, bits2 = intra_codec.intra_encode(
                block_current, return_bpp=True, is_source_rgb=False)
            recon2 = intra_codec.intra_decode(stream2, block_current.shape)

            dist2 = calc_mse(block_current, recon2)
            J2 = dist2 + lambda_rd * bits2

            J_all = [J0, J1, J2]
            best_mode = np.argmin(J_all)

            if best_mode == 0:
                recon_ycbcr[i:i+8, j:j+8, :] = block_ref
                total_bits += 0  # No need to encode
                skip_count += 1

            elif best_mode == 1:
                recon_ycbcr[i:i+8, j:j+8, :] = recon1
                total_bits += mv_bitsize + residual_bitsize
                inter_count += 1

            else:
                recon_ycbcr[i:i+8, j:j+8, :] = recon2
                total_bits += bits2
                intra_count += 1

    return recon_ycbcr, total_bits


if __name__ == "__main__":
    from ivclab.utils import imread, calc_psnr
    from ivclab.video import VideoCodec
    from ivclab.signal import rgb2ycbcr, ycbcr2rgb
    import matplotlib.pyplot as plt

    lena_small = imread('../data/lena_small.tif')

    images = []
    for i in range(20, 40 + 1):
        images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

    all_bpps = list()
    all_psnrs = list()

    for q_scale in [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]:
        video_codec = VideoCodec(quantization_scale=q_scale)
        video_codec.intra_codec.train_huffman_from_image(lena_small)
        video_codec.use_mode_decision = True

        image_bpps = list()
        image_psnrs = list()

        for frame_num, image in enumerate(images):
            reconstructed_img, bitstream, bitsize = video_codec.encode_decode(image, frame_num=frame_num)
            reconstructed_img = ycbcr2rgb(reconstructed_img)
            bpp = bitsize / (image.size / 3)
            psnr = calc_psnr(image, reconstructed_img)
            image_bpps.append(bpp)
            image_psnrs.append(psnr)
            # print(f"frame: {frame_num}, PSNR: {psnr:.2f} dB, bpp: {bpp:.2f}")

        bpp = np.mean(image_bpps)
        psnr = np.mean(image_psnrs)
        all_bpps.append(bpp)
        all_psnrs.append(psnr)
        print(f"Q Scale: {q_scale}, PSNR: {psnr:.2f} dB, bpp: {bpp:.2f}")
        print('-' * 12)

    ch5_mdecision_bpps = np.array(all_bpps)
    ch5_mdecision_psnrs = np.array(all_psnrs)
    np.save('../data/ch5_mdecision_bpps.npy', ch5_mdecision_bpps)
    np.save('../data/ch5_mdecision_psnrs.npy', ch5_mdecision_psnrs)

    ch4_bpps = np.load('../data/ch4_bpps.npy')
    ch4_psnrs = np.load('../data/ch4_psnrs.npy')
    ch5_mdecision_bpps = np.load('../data/ch5_mdecision_bpps.npy')
    ch5_mdecision_psnrs = np.load('../data/ch5_mdecision_psnrs.npy')

    plt.figure()
    plt.xlabel('Bitrate (bpp)')
    plt.ylabel('PSNR [dB]')
    plt.title('Rate-Distortion Curve')
    plt.plot(ch4_bpps, ch4_psnrs, linestyle='--', marker='o', color='black', label='Video Codec Solution')
    plt.plot(ch5_mdecision_bpps, ch5_mdecision_psnrs, linestyle='--', marker='*', color='orange',
             label='Video Opt: Block mode decision')
    plt.legend()
    plt.show()



