import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.image import IntraCodec
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.video import MotionCompensator
from ivclab.utils import calc_psnr
import matplotlib.pyplot as plt
import time


class ThreeStepMotionEstimator:
    """
    Three-Step Search Motion Estimation
    """

    def __init__(self, search_range=16):
        self.search_range = search_range

    def calculate_sad(self, current_block, reference_block):
        """
        Calculate Sum of Absolute Differences
        """
        if current_block.shape != reference_block.shape:
            return float('inf')
        return np.sum(np.abs(current_block.astype(np.float32) - reference_block.astype(np.float32)))

    def get_reference_block(self, reference_image, center_x, center_y, mv_x, mv_y, block_size):
        """
        Get reference block based on motion vector
        """
        h, w = reference_image.shape
        ref_x = center_x + mv_x
        ref_y = center_y + mv_y

        # Boundary check
        if (ref_x < 0 or ref_y < 0 or
                ref_x + block_size > w or ref_y + block_size > h):
            return np.zeros((block_size, block_size), dtype=np.float32)

        return reference_image[ref_y:ref_y + block_size, ref_x:ref_x + block_size]

    def three_step_search(self, current_block, reference_image, center_x, center_y):
        """
        Three-Step Search Algorithm
        """
        block_size = current_block.shape[0]
        min_sad = float('inf')
        best_mv = (0, 0)

        # Initial step size (half of search range)
        step_size = self.search_range // 2
        if step_size == 0:
            step_size = 1

        # Current center
        curr_x, curr_y = 0, 0

        # Three-step search
        for step in range(3):
            candidates = []

            if step == 0:
                # First step: 9 points in large step
                candidates = [
                    (curr_x - step_size, curr_y - step_size),
                    (curr_x, curr_y - step_size),
                    (curr_x + step_size, curr_y - step_size),
                    (curr_x - step_size, curr_y),
                    (curr_x, curr_y),
                    (curr_x + step_size, curr_y),
                    (curr_x - step_size, curr_y + step_size),
                    (curr_x, curr_y + step_size),
                    (curr_x + step_size, curr_y + step_size)
                ]
            else:
                # Subsequent steps: 8 neighbors
                candidates = [
                    (curr_x - step_size, curr_y - step_size),
                    (curr_x, curr_y - step_size),
                    (curr_x + step_size, curr_y - step_size),
                    (curr_x - step_size, curr_y),
                    (curr_x + step_size, curr_y),
                    (curr_x - step_size, curr_y + step_size),
                    (curr_x, curr_y + step_size),
                    (curr_x + step_size, curr_y + step_size)
                ]

            # Find the best candidate in this step
            step_min_sad = min_sad
            for dx, dy in candidates:
                # Check if within search range
                if abs(dx) > self.search_range or abs(dy) > self.search_range:
                    continue

                ref_block = self.get_reference_block(reference_image, center_x, center_y,
                                                     dx, dy, block_size)
                sad = self.calculate_sad(current_block, ref_block)

                if sad < step_min_sad:
                    step_min_sad = sad
                    curr_x, curr_y = dx, dy

            min_sad = step_min_sad
            best_mv = (curr_x, curr_y)

            # Reduce step size for next iteration
            step_size = max(1, step_size // 2)

        return best_mv, min_sad

    def compute_motion_vectors(self, reference_frame, current_frame):
        """
        Compute motion vectors using Three-Step Search
        """
        h, w = current_frame.shape
        block_size = 8  # 8x8 blocks

        num_blocks_y = h // block_size
        num_blocks_x = w // block_size

        motion_vectors = np.zeros((num_blocks_y, num_blocks_x, 2), dtype=np.int32)

        for by in range(num_blocks_y):
            for bx in range(num_blocks_x):
                # Get current block
                y_start = by * block_size
                x_start = bx * block_size
                current_block = current_frame[y_start:y_start + block_size,
                                x_start:x_start + block_size]

                # Block center for search
                center_x = x_start
                center_y = y_start

                # Three-Step Search
                mv, sad = self.three_step_search(current_block, reference_frame,
                                                 center_x, center_y)
                motion_vectors[by, bx] = mv

        return motion_vectors

    def reconstruct_with_motion_vectors(self, reference_frame, motion_vectors):
        """
        Motion compensation reconstruction
        """
        if len(reference_frame.shape) == 3:
            h, w, c = reference_frame.shape
            reconstructed = np.zeros_like(reference_frame)

            for channel in range(c):
                reconstructed[:, :, channel] = self.reconstruct_with_motion_vectors(
                    reference_frame[:, :, channel], motion_vectors)
            return reconstructed

        h, w = reference_frame.shape
        block_size = 8
        reconstructed = np.zeros((h, w), dtype=np.float32)

        num_blocks_y = h // block_size
        num_blocks_x = w // block_size

        for by in range(num_blocks_y):
            for bx in range(num_blocks_x):
                mv_x, mv_y = motion_vectors[by, bx]

                y_start = by * block_size
                x_start = bx * block_size

                # Calculate reference block position
                ref_x = x_start + mv_x
                ref_y = y_start + mv_y

                # Boundary check and copy
                if (ref_x >= 0 and ref_y >= 0 and
                        ref_x + block_size <= w and ref_y + block_size <= h):
                    ref_block = reference_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                    reconstructed[y_start:y_start + block_size, x_start:x_start + block_size] = ref_block

        return reconstructed


class VideoCodec:
    """
    Video Codec with Three-Step Search Motion Estimation
    """

    def __init__(self,
                 quantization_scale=1.0,
                 bounds=(-1000, 4000),
                 end_of_block=4000,
                 block_shape=(8, 8),
                 search_range=16,
                 use_three_step=True  # Enable/disable Three-Step Search
                 ):

        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape
        self.search_range = search_range
        self.use_three_step = use_three_step

        self.intra_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds,
                                      end_of_block=end_of_block, block_shape=block_shape)
        self.residual_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds,
                                         end_of_block=end_of_block, block_shape=block_shape)

        # Motion estimator
        if use_three_step:
            self.motion_comp = ThreeStepMotionEstimator(search_range=search_range)
        else:
            self.motion_comp = MotionCompensator(search_range=search_range)

        self.motion_huffman = HuffmanCoder(lower_bound=0)
        self.decoder_recon = None

    def encode_decode(self, frame, frame_num=0):
        if frame_num == 0:
            # I-frame
            bitstream, residual_bitsize = self.intra_codec.intra_encode(frame, return_bpp=True)
            self.decoder_recon = self.intra_codec.intra_decode(bitstream, frame.shape)
            self.decoder_recon = ycbcr2rgb(self.decoder_recon)
            motion_bitsize = 0
        else:
            # P-frame
            curr_ycbcr = rgb2ycbcr(frame)
            ref_ycbcr = rgb2ycbcr(self.decoder_recon)

            # Motion vector computation
            if self.use_three_step:
                motion_vector = self.motion_comp.compute_motion_vectors(
                    ref_ycbcr[..., 0], curr_ycbcr[..., 0])
            else:
                motion_vector = self.motion_comp.compute_motion_vector(
                    ref_ycbcr[..., 0], curr_ycbcr[..., 0])

            # Motion compensation
            if self.use_three_step:
                recon_pred_frame_ycbcr = self.motion_comp.reconstruct_with_motion_vectors(
                    ref_ycbcr, motion_vector)
            else:
                recon_pred_frame_ycbcr = self.motion_comp.reconstruct_with_motion_vector(
                    ref_ycbcr, motion_vector)

            # Compute residual
            residual = curr_ycbcr - recon_pred_frame_ycbcr

            # Train codecs on the first P-frame
            if frame_num == 1:
                if self.use_three_step:
                    # Use uniform distribution for Three-Step Search
                    mv_min = -self.search_range
                    mv_max = self.search_range
                    symbol_range = np.arange(mv_min, mv_max + 1)
                    uniform_pmf = np.ones(len(symbol_range)) / len(symbol_range)
                    self.motion_huffman = HuffmanCoder(lower_bound=mv_min)
                    self.motion_huffman.train(uniform_pmf)
                else:
                    # Use original encoding for MotionCompensator
                    mv_flat = motion_vector.flatten()
                    symbol_range = np.arange(0, 81 + 2)  # Original range
                    mv_pmf = stats_marg(mv_flat, symbol_range)
                    self.motion_huffman = HuffmanCoder(lower_bound=0)
                    self.motion_huffman.train(mv_pmf)

                # Train residual codec
                self.residual_codec.train_huffman_from_image(residual, is_source_rgb=False)

            # Motion vector encoding
            mv_flat = motion_vector.flatten()
            motion_stream, motion_bitsize = self.motion_huffman.encode(mv_flat)

            # Residual encode
            residual_stream, residual_bitsize = self.residual_codec.intra_encode(
                residual, return_bpp=True, is_source_rgb=False)

            # Decoding
            mv_decoded = self.motion_huffman.decode(motion_stream, motion_vector.size)
            mv_decoded = mv_decoded.reshape(motion_vector.shape)

            # Motion compensation (decoder)
            if self.use_three_step:
                recon_pred_frame_ycbcr_decoded = self.motion_comp.reconstruct_with_motion_vectors(
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
    import numpy as np
    from ivclab.utils import imread, calc_psnr
    import matplotlib.pyplot as plt
    from ivclab.signal import rgb2ycbcr, ycbcr2rgb

    lena_small = imread('../data/lena_small.tif')

    images = []
    for i in range(20, 40 + 1):
        images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

    all_bpps = list()
    all_psnrs = list()

    for q_scale in [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]:
        start_time = time.time()
        video_codec = VideoCodec(quantization_scale=q_scale, use_three_step = False)
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
    full_search_time = time.time() - start_time

    images = []
    for i in range(20, 40 + 1):
        images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

    all_bpps = list()
    all_psnrs = list()

    for q_scale in [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]:
        start_time = time.time()
        video_codec = VideoCodec(quantization_scale=q_scale, use_three_step = True)
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

    ch5_fast_bpps = np.array(all_bpps)
    ch5_fast_psnrs = np.array(all_psnrs)
    three_step_time = time.time() - start_time

    np.save('../data/ch5_fast_bpps.npy', ch5_fast_bpps)
    np.save('../data/ch5_fast_psnrs.npy', ch5_fast_psnrs)

    ch4_bpps = np.load('../data/ch4_bpps.npy')
    ch4_psnrs = np.load('../data/ch4_psnrs.npy')
    ch5_fast_bpps = np.load('../data/ch5_fast_bpps.npy')
    ch5_fast_psnrs = np.load('../data/ch5_fast_psnrs.npy')


    plt.figure()
    plt.xlabel('Bitrate (bpp)')
    plt.ylabel('PSNR [dB]')
    plt.title('Rate-Distortion Curve')
    plt.plot(ch4_bpps, ch4_psnrs, linestyle='--', marker='s', color='blue', label='Video Codec Solution')
    plt.plot(ch5_fast_bpps, ch5_fast_psnrs, linestyle='--', marker='s', color='yellow',label='Video Opt: Fast Motion')
    plt.legend()
    plt.show()