import numpy as np


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


if __name__ == "__main__":
    from ivclab.utils import imread, calc_psnr
    from ivclab.image import IntraCodec
    from ivclab.signal import rgb2ycbcr, ycbcr2rgb
    import matplotlib.pyplot as plt

    lena_small = imread('../data/lena_small.tif')

    images = []
    for i in range(20, 40 + 1):
        images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

    all_bpps = list()
    all_psnrs = list()

    for q_scale in [0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10]:
        intra_codec = IntraCodec(quantization_scale=q_scale)
        intra_codec.train_huffman_from_image(lena_small)

        if q_scale in [0.3, 0.7, 1.0, 1.5]:
            intra_codec.use_adaptive_quant = True

        image_bpps = list()
        image_psnrs = list()

        for frame_num, image in enumerate(images):
            message, bitrate = intra_codec.intra_encode(image, return_bpp=True, is_source_rgb=True)
            reconstructed_img = intra_codec.intra_decode(message, image.shape)
            reconstructed_img = ycbcr2rgb(reconstructed_img)
            bpp = bitrate / (image.size / 3)
            psnr = calc_psnr(image, reconstructed_img)
            image_bpps.append(bpp)
            image_psnrs.append(psnr)

        bpp = np.mean(image_bpps)
        psnr = np.mean(image_psnrs)
        all_bpps.append(bpp)
        all_psnrs.append(psnr)
        print(f"Q Scale: {q_scale}, PSNR: {psnr:.2f} dB, bpp: {bpp:.2f}")
        print('-' * 12)

    ch5_aquant_bpps = np.array(all_bpps)
    ch5_aquant_psnrs = np.array(all_psnrs)
    np.save('../data/ch5_aquant_bpps.npy', ch5_aquant_bpps)
    np.save('../data/ch5_aquant_psnrs.npy', ch5_aquant_psnrs)

    ch3_bpps = np.load('../data/ch3_bpps.npy')
    ch3_psnrs = np.load('../data/ch3_psnrs.npy')
    ch5_aquant_bpps = np.load('../data/ch5_aquant_bpps.npy')
    ch5_aquant_psnrs = np.load('../data/ch5_aquant_psnrs.npy')

    plt.figure()
    plt.xlabel('Bitrate (bpp)')
    plt.ylabel('PSNR [dB]')
    plt.title('Rate-Distortion Curve')
    plt.plot(ch3_bpps, ch3_psnrs, linestyle='--', marker='*', color='black', label='Image Codec Solution')
    plt.plot(ch5_aquant_bpps, ch5_aquant_psnrs, linestyle='--', marker='*', color='green',
             label='Intra Opt: Adaptive quantize')
    plt.legend()
    plt.show()
