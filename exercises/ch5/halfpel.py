import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.image import IntraCodec
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.video import MotionCompensator
from ivclab.utils import calc_psnr
import matplotlib.pyplot as plt


class HalfPelMotionCompensator:
    """
    Half-pixel motion compensator for both encoding and decoding
    """

    def __init__(self, search_range=4):
        self.search_range = search_range
        self.integer_mc = MotionCompensator(search_range=search_range)

    def get_half_pel_block(self, image, ref_x, ref_y, block_h, block_w):
        """
        按需计算半像素块 (替代create_half_pel_image)
        根据参考位置计算8x8块的半像素值
        """
        h, w = image.shape

        # 检查边界
        if (ref_x < 0 or ref_y < 0 or
                ref_x + block_w > w or ref_y + block_h > h):
            return np.zeros((block_h, block_w), dtype=np.float32)

        # 获取整数和小数部分
        int_x, int_y = int(ref_x), int(ref_y)
        frac_x, frac_y = ref_x - int_x, ref_y - int_y

        # 如果是整数位置，直接返回
        if frac_x == 0 and frac_y == 0:
            return image[int_y:int_y + block_h, int_x:int_x + block_w].astype(np.float32)

        # 需要插值的情况
        ref_block = np.zeros((block_h, block_w), dtype=np.float32)

        for y in range(block_h):
            for x in range(block_w):
                pixel_x = ref_x + x
                pixel_y = ref_y + y
                ref_block[y, x] = self.bilinear_interpolation(image, pixel_x, pixel_y)

        return ref_block

    def bilinear_interpolation(self, image, x, y):
        """
        双线性插值 (解码端使用)
        用于根据分数运动向量重建像素值
        """
        h, w = image.shape

        # 边界检查 - 更严格的边界处理
        if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
            # 边界外使用边界像素值
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            return image[int(y), int(x)]

        # 获取整数部分
        x1, y1 = int(x), int(y)
        x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)

        # 获取小数部分
        fx, fy = x - x1, y - y1

        # 双线性插值
        result = (1 - fx) * (1 - fy) * image[y1, x1] + \
                 fx * (1 - fy) * image[y1, x2] + \
                 (1 - fx) * fy * image[y2, x1] + \
                 fx * fy * image[y2, x2]

        return result

    def block_matching_hierarchical(self, current_block, reference_image, loc_x, loc_y):
        """
        分层搜索：先整数后半像素 (基于MATLAB版本的思路)
        """
        block_h, block_w = current_block.shape
        h, w = reference_image.shape

        # Step 1: 整数像素搜索 (±4范围)
        min_ssd = float('inf')
        best_int_x, best_int_y = loc_x, loc_y

        for ref_x in range(loc_x - self.search_range, loc_x + self.search_range + 1):
            if ref_x < 0 or ref_x > w - block_w:  # 边界检查
                continue
            for ref_y in range(loc_y - self.search_range, loc_y + self.search_range + 1):
                if ref_y < 0 or ref_y > h - block_h:  # 边界检查
                    continue

                # 当前参考块
                ref_block = reference_image[ref_y:ref_y + block_h, ref_x:ref_x + block_w]

                # 计算SSD (与MATLAB保持一致)
                diff = (current_block.astype(np.float32) - ref_block.astype(np.float32)) ** 2
                ssd = np.sum(diff)

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_int_x, best_int_y = ref_x, ref_y

        # Step 2: 半像素搜索 (在最佳整数位置周围搜索8个半像素位置)
        # 定义8个半像素搜索位置 (包括MATLAB的4个 + 4个边中点)
        half_pel_patterns = [
            (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5),  # MATLAB的4个角点
            (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5)  # 4个边中点
        ]

        final_best_x, final_best_y = best_int_x, best_int_y
        final_min_ssd = min_ssd

        for dx, dy in half_pel_patterns:
            new_ref_x = best_int_x + dx
            new_ref_y = best_int_y + dy

            # 边界检查
            if (new_ref_x < 0 or new_ref_y < 0 or
                    new_ref_x + block_w > w or new_ref_y + block_h > h):
                continue

            # 获取半像素参考块
            new_ref_block = self.get_half_pel_block(reference_image, new_ref_x, new_ref_y, block_h, block_w)

            # 计算SSD
            diff = (current_block.astype(np.float32) - new_ref_block) ** 2
            ssd = np.sum(diff)

            if ssd < final_min_ssd:
                final_min_ssd = ssd
                final_best_x, final_best_y = new_ref_x, new_ref_y

        # 计算相对运动向量
        mv_x = final_best_x - loc_x
        mv_y = final_best_y - loc_y

        return (mv_x, mv_y), final_min_ssd

    def compute_motion_vector_half_pel(self, reference_frame, current_frame):
        """
        计算半像素精度的运动向量 (使用分层搜索)
        """
        h, w = current_frame.shape
        block_h, block_w = 8, 8  # 使用8x8块

        # 计算块的数量
        num_blocks_y = h // block_h
        num_blocks_x = w // block_w

        motion_vectors = np.zeros((num_blocks_y, num_blocks_x, 2), dtype=np.float32)

        for by in range(num_blocks_y):
            for bx in range(num_blocks_x):
                # 获取当前块的位置
                loc_x = bx * block_w  # 块的左上角x坐标
                loc_y = by * block_h  # 块的左上角y坐标

                # 获取当前块
                current_block = current_frame[loc_y:loc_y + block_h, loc_x:loc_x + block_w]

                # 进行分层半像素块匹配
                mv, ssd = self.block_matching_hierarchical(current_block, reference_frame, loc_x, loc_y)
                motion_vectors[by, bx] = mv

        return motion_vectors

    def reconstruct_with_motion_vector_half_pel(self, reference_frame, motion_vectors):
        """
        使用半像素运动向量重建帧 (解码端使用)
        """
        if len(reference_frame.shape) == 3:
            h, w, c = reference_frame.shape
            reconstructed = np.zeros_like(reference_frame)

            for channel in range(c):
                reconstructed[:, :, channel] = self.reconstruct_with_motion_vector_half_pel(
                    reference_frame[:, :, channel], motion_vectors)
            return reconstructed

        h, w = reference_frame.shape
        block_h, block_w = 8, 8
        reconstructed = np.zeros((h, w), dtype=np.float32)

        num_blocks_y = h // block_h
        num_blocks_x = w // block_w

        for by in range(num_blocks_y):
            for bx in range(num_blocks_x):
                # 获取运动向量
                mv_x, mv_y = motion_vectors[by, bx]

                # 计算块位置
                loc_x = bx * block_w
                loc_y = by * block_h

                # 计算参考块位置
                ref_x = loc_x + mv_x
                ref_y = loc_y + mv_y

                # 使用按需插值获取参考块
                ref_block = self.get_half_pel_block(reference_frame, ref_x, ref_y, block_h, block_w)
                reconstructed[loc_y:loc_y + block_h, loc_x:loc_x + block_w] = ref_block

        return reconstructed


class VideoCodec:

    def __init__(self,
                 quantization_scale=1.0,
                 bounds=(-1000, 4000),
                 end_of_block=4000,
                 block_shape=(8, 8),
                 search_range=4,
                 use_half_pel=True  # 新增参数
                 ):

        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape
        self.search_range = search_range
        self.use_half_pel = use_half_pel  # 是否使用半像素运动估计

        self.intra_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds, end_of_block=end_of_block,
                                      block_shape=block_shape)
        self.residual_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds,
                                         end_of_block=end_of_block, block_shape=block_shape)

        # 根据是否使用半像素选择运动补偿器
        if use_half_pel:
            self.motion_comp = HalfPelMotionCompensator(search_range=search_range)
        else:
            self.motion_comp = MotionCompensator(search_range=search_range)

        self.motion_huffman = HuffmanCoder(lower_bound=0)
        self.decoder_recon = None

    def encode_decode(self, frame, frame_num=0):
        if frame_num == 0:
            # ---Intra mode---
            bitstream, residual_bitsize = self.intra_codec.intra_encode(frame, return_bpp=True)
            self.decoder_recon = self.intra_codec.intra_decode(bitstream, frame.shape)
            # Convert to RGB
            self.decoder_recon = ycbcr2rgb(self.decoder_recon)
            motion_bitsize = 0  # No motion in intra frame
        else:
            # ---Inter mode---
            # Perform color transform
            curr_ycbcr = rgb2ycbcr(frame)
            ref_ycbcr = rgb2ycbcr(self.decoder_recon)

            # Motion vector computation (支持半像素)
            if self.use_half_pel:
                motion_vector = self.motion_comp.compute_motion_vector_half_pel(
                    ref_ycbcr[..., 0], curr_ycbcr[..., 0])
            else:
                motion_vector = self.motion_comp.compute_motion_vector(
                    ref_ycbcr[..., 0], curr_ycbcr[..., 0])

            # Perform motion compensation
            if self.use_half_pel:
                recon_pred_frame_ycbcr = self.motion_comp.reconstruct_with_motion_vector_half_pel(
                    ref_ycbcr, motion_vector)
            else:
                recon_pred_frame_ycbcr = self.motion_comp.reconstruct_with_motion_vector(
                    ref_ycbcr, motion_vector)

            # Compute residual
            residual = curr_ycbcr - recon_pred_frame_ycbcr

            # 对于半像素运动向量，需要量化编码
            if self.use_half_pel:
                # 将半像素运动向量量化为整数表示 (乘以2)
                mv_quantized = (motion_vector * 2).astype(np.int32)
                mv_flat = mv_quantized.flatten()

                # 调整符号范围以适应半像素
                if frame_num == 1:
                    symbol_range = np.arange(-self.search_range * 4, self.search_range * 4 + 1)
                    mv_pmf = stats_marg(mv_flat, symbol_range)
                    self.motion_huffman = HuffmanCoder(lower_bound=-self.search_range * 4)
                    self.motion_huffman.train(mv_pmf)
                    self.residual_codec.train_huffman_from_image(residual, is_source_rgb=False)
            else:
                # 原来的整数运动向量处理
                mv_flat = motion_vector.flatten()
                if frame_num == 1:
                    symbol_range = np.arange(0, 81 + 2)
                    mv_pmf = stats_marg(mv_flat, symbol_range)
                    self.motion_huffman.train(mv_pmf)
                    self.residual_codec.train_huffman_from_image(residual, is_source_rgb=False)

            motion_stream, motion_bitsize = self.motion_huffman.encode(mv_flat)

            # Residual encode
            residual_stream, residual_bitsize = self.residual_codec.intra_encode(
                residual, return_bpp=True, is_source_rgb=False)

            # Decoding motion vector
            mv_decoded = self.motion_huffman.decode(motion_stream, motion_vector.size)

            if self.use_half_pel:
                # 半像素运动向量反量化 (除以2)
                mv_decoded = (mv_decoded.reshape(motion_vector.shape) / 2.0).astype(np.float32)
            else:
                mv_decoded = mv_decoded.reshape(motion_vector.shape)

            # Motion compensation (decoder)
            if self.use_half_pel:
                recon_pred_frame_ycbcr_decoded = self.motion_comp.reconstruct_with_motion_vector_half_pel(
                    ref_ycbcr, mv_decoded)
            else:
                recon_pred_frame_ycbcr_decoded = self.motion_comp.reconstruct_with_motion_vector(
                    ref_ycbcr, mv_decoded)

            # Residual decoding
            residual_recon = self.residual_codec.intra_decode(residual_stream, frame.shape)
            recon_frame_ycbcr = recon_pred_frame_ycbcr_decoded + residual_recon

            # Convert to RGB
            self.decoder_recon = ycbcr2rgb(recon_frame_ycbcr).copy()

            bitstream = (motion_stream, residual_stream)

        bitsize = residual_bitsize + motion_bitsize
        return self.decoder_recon, bitstream, bitsize

if __name__ == "__main__":
    from ivclab.image import IntraCodec
    import numpy as np
    from ivclab.utils import imread, calc_psnr
    import matplotlib.pyplot as plt
    from ivclab.signal import rgb2ycbcr, ycbcr2rgb
    from exercises.ch5.deblock import deblock

    lena_small = imread('../data/lena_small.tif')

    # 2. Chapter 4: Video Coding
    images = []
    for i in range(20, 40 + 1):
        images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

    all_bpps = list()
    all_psnrs = list()

    for q_scale in [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]:
        video_codec = VideoCodec(quantization_scale=q_scale, use_half_pel=False)
        video_codec.intra_codec.train_huffman_from_image(lena_small, is_source_rgb=True)
        bpps = list()
        psnrs = list()
        for frame_num, image in enumerate(images):
            reconstructed_image, bitstream, bitsize = video_codec.encode_decode(image, frame_num=frame_num)
            bpp = bitsize / (image.size / 3)
            psnr = calc_psnr(image, reconstructed_image)
            print(f"Frame:{frame_num} PSNR: {psnr:.2f} dB bpp: {bpp:.2f}")
            bpps.append(bpp)
            psnrs.append(psnr)

        all_bpps.append(np.mean(bpps))
        all_psnrs.append(np.mean(psnrs))
        print(f"Q-Scale {q_scale}, Average PSNR: {np.mean(psnrs):.2f} dB Average bpp: {np.mean(bpps):.2f}")
        print('-' * 12)

    ch4_bpps = np.array(all_bpps)
    ch4_psnrs = np.array(all_psnrs)

    np.save('../data/ch4_bpps.npy', ch4_bpps)
    np.save('../data/ch4_psnrs.npy', ch4_psnrs)

    images = []
    for i in range(20, 40 + 1):
        images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

    all_bpps = list()
    all_PSNRs = list()
    for q_scale in [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]:
        video_codec = VideoCodec(quantization_scale=q_scale, use_half_pel=True)
        video_codec.intra_codec.train_huffman_from_image(lena_small, is_source_rgb=True)
        bpps = list()
        psnrs = list()
        for frame_num, image in enumerate(images):
            reconstructed_image, bitstream, bitsize = video_codec.encode_decode(image, frame_num=frame_num)
            bpp = bitsize / (image.size / 3)
            psnr = calc_psnr(image, reconstructed_image)
            print(f"Frame:{frame_num} PSNR: {psnr:.2f} dB bpp: {bpp:.2f}")
            bpps.append(bpp)
            psnrs.append(psnr)

        all_bpps.append(np.mean(bpps))
        all_PSNRs.append(np.mean(psnrs))
        print(f"Q-Scale {q_scale}, Average PSNR: {np.mean(psnrs):.2f} dB Average bpp: {np.mean(bpps):.2f}")
    ch5_halfpel_bpps = np.array(all_bpps)
    ch5_halfpel_psnrs = np.array(all_PSNRs)

    np.save('../data/ch5_halfpel_bpps.npy', ch5_halfpel_bpps)
    np.save('../data/ch5_halfpel_psnrs.npy', ch5_halfpel_psnrs)
    ch4_bpps = np.load('../data/ch4_bpps.npy')
    ch4_psnrs = np.load('../data/ch4_psnrs.npy')
    ch5_halfpel_bpps = np.load('../data/ch5_halfpel_bpps.npy')
    ch5_halfpel_psnrs = np.load('../data/ch5_halfpel_psnrs.npy')

    plt.figure()
    plt.xlabel('Bitrate (bpp)')
    plt.ylabel('PSNR [dB]')
    plt.title('Rate-Distortion Curve')
    plt.plot(ch4_bpps, ch4_psnrs, linestyle='--', marker='s', color='blue', label='Video Codec Solution')
    plt.plot(ch5_halfpel_bpps, ch5_halfpel_psnrs, linestyle='--', marker='s', color='yellow', label='Video Opt: Halfpel')
    plt.legend()
    plt.show()