import numpy as np

# Alpha and Beta tables (from H.264 standard, QP index 0~51)
alpha_table = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22, 25, 28,
    32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255
]
beta_table = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18
]

def deblock(img_ycbcr: np.ndarray, index: int) -> np.ndarray:
    """
    Apply post-deblocking filter to YCbCr image.
    :param img_ycbcr: Input image in shape (H, W, 3), float or uint8
    :param index: quantization index (0~51), mapped from q_scale
    :return: deblocked image (same shape as input)
    """
    img = img_ycbcr.astype(np.float32).copy()
    H, W, C = img.shape
    blk = 8

    alpha = alpha_table[index]
    beta = beta_table[index]

    def filter_edge(p, q):
        diff1 = abs(p[0] - q[0])
        diff2 = abs(p[2] - p[0])
        diff3 = abs(q[2] - q[0])
        diff4 = abs(p[1] - p[0])
        diff5 = abs(q[1] - q[0])

        if diff1 < alpha and diff4 < beta and diff5 < beta:
            if diff2 < beta and diff3 < beta:
                # Strong deblocking
                p[0] = (p[2] + 2*p[1] + 2*p[0] + 2*q[0] + q[1] + 4) / 8
                p[1] = (p[2] + p[1] + p[0] + q[0] + 2) / 4
                p[2] = (2*p[3] + 3*p[2] + p[1] + p[0] + q[0] + 4) / 8
                q[0] = (q[2] + 2*q[1] + 2*q[0] + 2*p[0] + p[1] + 4) / 8
                q[1] = (q[2] + q[1] + q[0] + p[0] + 2) / 4
                q[2] = (2*q[3] + 3*q[2] + q[1] + q[0] + p[0] + 4) / 8
            else:
                # Weak deblocking
                p[0] = (2*p[1] + p[0] + q[1] + 2) / 4
                q[0] = (2*q[1] + q[0] + p[1] + 2) / 4

    # Vertical edge filtering
    for c in range(C):
        for row in range(blk, H - blk, blk):
            for col in range(0, W, blk):
                upper = img[row-blk:row, col:col+blk, c]
                lower = img[row:row+blk, col:col+blk, c]
                for k in range(blk):
                    p = upper[-4:, k].copy()
                    q = lower[:4, k].copy()
                    filter_edge(p, q)
                    upper[-3:, k] = p[1:]
                    lower[:3, k] = q[:3]

    # Horizontal edge filtering
    for c in range(C):
        for col in range(blk, W - blk, blk):
            for row in range(0, H, blk):
                left = img[row:row+blk, col-blk:col, c]
                right = img[row:row+blk, col:col+blk, c]
                for k in range(blk):
                    p = left[k, -4:].copy()
                    q = right[k, :4].copy()
                    filter_edge(p, q)
                    left[k, -3:] = p[1:]
                    right[k, :3] = q[:3]

    return np.clip(img_ycbcr, 0, 255)

if __name__ == "__main__":
    import numpy as np

    from ivclab import ycbcr2rgb
    from ivclab.image import IntraCodec
    from ivclab.utils import imread, calc_psnr
    import matplotlib.pyplot as plt

    lena = imread(f'../data/lena.tif')
    lena_small = imread(f'../data/lena_small.tif')
    H, W, C = lena.shape
    all_PSNRs = list()
    all_bpps = list()
    indexDeblocking = [1, 1, 1, 1, 1, 28, 29, 35, 39, 45]
    # YOUR CODE STARTS HERE
    for scale_idx, q in enumerate([0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10]):
        index = indexDeblocking[scale_idx]
        intracodec = IntraCodec(quantization_scale=q)

        intracodec.train_huffman_from_image(lena_small)

        symbols, bitsize = intracodec.intra_encode(lena, return_bpp=True)
        reconstructed_img = intracodec.intra_decode(symbols, lena.shape)
        print("Deblock input stats:",
              "min =", reconstructed_img.min(),
              "max =", reconstructed_img.max(),
              "dtype =", reconstructed_img.dtype)
        reconstructed_img = deblock(reconstructed_img, index)
        diff = np.mean(np.abs(reconstructed_img - ycbcr2rgb(lena)))
        print(f"Mean change after deblock: {diff}")
        reconstructed_img = ycbcr2rgb(reconstructed_img)

        psnr = calc_psnr(lena, reconstructed_img)
        bpp = bitsize / (H * W)

        all_PSNRs.append(psnr)
        all_bpps.append(bpp)
    # YOUR CODE ENDS HERE

    all_bpps = np.array(all_bpps)
    all_PSNRs = np.array(all_PSNRs)


    print(all_bpps, all_PSNRs)
    plt.plot(all_bpps, all_PSNRs, marker='o')
    plt.show()
