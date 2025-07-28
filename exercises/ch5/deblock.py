import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.image import IntraCodec
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.video import MotionCompensator
from ivclab.utils import calc_psnr
import matplotlib.pyplot as plt


class DeblockingFilter:
    """
    H.264-style deblocking filter for post-processing
    """

    def __init__(self, block_size=8):
        self.block_size = block_size

        # Alpha and Beta tables (from H.264 standard, QP index 0~51)
        self.alpha_table = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22, 25, 28,
            32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255
        ])

        self.beta_table = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
            9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18
        ])

    def quantization_scale_to_qp(self, q_scale):
        """
        将量化尺度映射到QP索引
        """
        # 根据你的量化范围[0.07, 4.5]映射到[0, 51]
        qp = int(np.clip(np.log2(q_scale) * 6 + 26, 0, 51))
        return qp

    def deblock(self, img_ycbcr: np.ndarray, index: int) -> np.ndarray:
        """
        Apply post-deblocking filter to YCbCr image.
        :param img_ycbcr: Input image in shape (H, W, 3), float or uint8
        :param index: quantization index (0~51), mapped from q_scale
        :return: deblocked image (same shape as input)
        """
        img = img_ycbcr.astype(np.float32).copy()
        H, W, C = img.shape
        blk = self.block_size

        alpha = self.alpha_table[index]  # Threshold for luminance difference of the boundary
        beta = self.beta_table[index]  # Threshold for difference within the boundary

        def filter_edge(p, q):
            diff1 = abs(p[0] - q[0])
            diff2 = abs(p[2] - p[0])
            diff3 = abs(q[2] - q[0])
            diff4 = abs(p[1] - p[0])
            diff5 = abs(q[1] - q[0])

            if diff1 < alpha and diff4 < beta and diff5 < beta:
                if diff2 < beta and diff3 < beta:
                    # Strong deblocking
                    p[0] = (p[2] + 2 * p[1] + 2 * p[0] + 2 * q[0] + q[1] + 4) / 8
                    p[1] = (p[2] + p[1] + p[0] + q[0] + 2) / 4
                    p[2] = (2 * p[3] + 3 * p[2] + p[1] + p[0] + q[0] + 4) / 8
                    q[0] = (q[2] + 2 * q[1] + 2 * q[0] + 2 * p[0] + p[1] + 4) / 8
                    q[1] = (q[2] + q[1] + q[0] + p[0] + 2) / 4
                    q[2] = (2 * q[3] + 3 * q[2] + q[1] + q[0] + p[0] + 4) / 8
                else:
                    # Weak deblocking (only smooth boundary pixels)
                    p[0] = (2 * p[1] + p[0] + q[1] + 2) / 4
                    q[0] = (2 * q[1] + q[0] + p[1] + 2) / 4

        # Vertical edge filtering
        for c in range(C):
            for row in range(blk, H - blk, blk):
                for col in range(0, W, blk):
                    upper = img[row - blk:row, col:col + blk, c]
                    lower = img[row:row + blk, col:col + blk, c]
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
                    left = img[row:row + blk, col - blk:col, c]
                    right = img[row:row + blk, col:col + blk, c]
                    for k in range(blk):
                        p = left[k, -4:].copy()
                        q = right[k, :4].copy()
                        filter_edge(p, q)
                        left[k, -3:] = p[1:]
                        right[k, :3] = q[:3]

        return img


class IntraCodec:
    """
    Enhanced Intra Codec with Deblocking Filter
    """

    def __init__(self,
                 quantization_scale=1.0,
                 bounds=(-1000, 4000),
                 end_of_block=4000,
                 block_shape=(8, 8),
                 use_deblocking=True  # 新增参数
                 ):

        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape
        self.use_deblocking = use_deblocking  # 是否使用去块滤波

        # 原始IntraCodec
        from ivclab.image import IntraCodec as OriginalIntraCodec
        self.intra_codec = OriginalIntraCodec(quantization_scale=quantization_scale, bounds=bounds,
                                              end_of_block=end_of_block, block_shape=block_shape)

        # 去块滤波器
        if use_deblocking:
            self.deblocking_filter = DeblockingFilter(block_size=block_shape[0])

    def train_huffman_from_image(self, training_img, is_source_rgb=True):
        """
        Finds the symbols representing the image, extracts the
        probability distribution of them and trains the huffman coder with it.

        training_img: np.array of shape [H, W, C]

        returns:
            Nothing
        """
        return self.intra_codec.train_huffman_from_image(training_img, is_source_rgb=is_source_rgb)

    def intra_encode(self, img: np.array, return_bpp=False, is_source_rgb=True):
        """
        Encodes an image to a bitstream and return it by converting it to
        symbols and compressing them with the Huffman coder.

        img: np.array of shape [H, W, C]

        returns:
            bitstream: List of integers produced by the Huffman coder
        """
        return self.intra_codec.intra_encode(img, return_bpp=return_bpp, is_source_rgb=is_source_rgb)

    def intra_decode(self, bitstream, original_shape):
        """
        Decodes an image from a bitstream by decoding it with the Huffman
        coder and reconstructing it from the symbols.

        bitstream: List of integers produced by the Huffman coder
        original_shape: List of 3 values that contain H, W, and C

        returns:
            reconstructed_img: np.array of shape [H, W, C]
        """
        # 解码
        reconstructed_img = self.intra_codec.intra_decode(bitstream, original_shape)

        # 可选地应用去块滤波
        if self.use_deblocking:
            qp_index = self.deblocking_filter.quantization_scale_to_qp(self.quantization_scale)
            reconstructed_img = self.deblocking_filter.deblock(reconstructed_img, qp_index)

        return reconstructed_img
if __name__ == "__main__":
    import numpy as np
    from ivclab.utils import imread, calc_psnr
    import matplotlib.pyplot as plt
    from ivclab.signal import rgb2ycbcr, ycbcr2rgb

    '''
    lena_small = imread('../data/lena_small.tif')

    images = []
    for i in range(20, 40 + 1):
        images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

    all_bpps = list()
    all_PSNRs = list()

    for q_scale in [0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10]:
        # for q_scale in [1.0]:
        intracodec = IntraCodec(quantization_scale=q_scale, use_deblocking=False)
        intracodec.train_huffman_from_image(lena_small)
        image_psnrs = list()
        image_bpps = list()
        for i in range(len(images)):
            img = images[i]
            H, W, C = img.shape
            message, bitrate = intracodec.intra_encode(img, return_bpp=True, is_source_rgb=True)
            reconstructed_img = intracodec.intra_decode(message, img.shape)
            reconstructed_img = ycbcr2rgb(reconstructed_img)
            psnr = calc_psnr(img, reconstructed_img)
            bpp = bitrate / (H * W)
            image_psnrs.append(psnr)
            image_bpps.append(bpp)

        psnr = np.mean(image_psnrs)
        bpp = np.mean(image_bpps)
        all_bpps.append(bpp)
        all_PSNRs.append(psnr)
        print(f"Q-Scale {q_scale}, PSNR: {psnr:.2f} dB, bpp: {bpp:.2f}")

    print('-' * 12)
    ch3_bpps = np.array(all_bpps)
    ch3_psnrs = np.array(all_PSNRs)

    images = []
    for i in range(20, 40 + 1):
        images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

    all_bpps = list()
    all_PSNRs = list()

    for q_scale in [0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10]:
        # for q_scale in [1.0]:
        intracodec = IntraCodec(quantization_scale=q_scale, use_deblocking=True)
        intracodec.train_huffman_from_image(lena_small)
        image_psnrs = list()
        image_bpps = list()
        for i in range(len(images)):
            img = images[i]
            H, W, C = img.shape
            message, bitrate = intracodec.intra_encode(img, return_bpp=True, is_source_rgb=True)
            reconstructed_img = intracodec.intra_decode(message, img.shape)
            reconstructed_img = ycbcr2rgb(reconstructed_img)
            psnr = calc_psnr(img, reconstructed_img)
            bpp = bitrate / (H * W)
            image_psnrs.append(psnr)
            image_bpps.append(bpp)

        psnr = np.mean(image_psnrs)
        bpp = np.mean(image_bpps)
        all_bpps.append(bpp)
        all_PSNRs.append(psnr)
        print(f"Q-Scale {q_scale}, PSNR: {psnr:.2f} dB, bpp: {bpp:.2f}")

    print('-' * 12)
    ch5_deblock_bpps = np.array(all_bpps)
    ch5_deblock_psnrs = np.array(all_PSNRs)

    np.save('../data/ch5_deblock_bpps.npy', ch5_deblock_bpps)
    np.save('../data/ch5_deblock_psnrs.npy', ch5_deblock_psnrs)
'''
    ch3_bpps = np.load('../data/ch3_bpps.npy')
    ch3_psnrs = np.load('../data/ch3_psnrs.npy')
    ch5_deblock_bpps = np.load('../data/ch5_deblock_bpps.npy')
    ch5_deblock_psnrs = np.load('../data/ch5_deblock_psnrs.npy')

    plt.figure()
    plt.xlabel('Bitrate (bpp)')
    plt.ylabel('PSNR [dB]')
    plt.title('Rate-Distortion Curve')
    plt.plot(ch3_bpps, ch3_psnrs, linestyle='--', marker='.', color='orange', label='Image Codec Solution')
    plt.plot(ch5_deblock_bpps, ch5_deblock_psnrs, linestyle='--', marker='x', color='purple', label='Image Opt Deblock')
    plt.legend()
    plt.show()
