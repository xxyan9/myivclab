import numpy as np
import matplotlib.pyplot as plt
from ivclab.signal import ycbcr2rgb

class MotionCompensator:

    def __init__(self, search_range=4):
        self.search_range = search_range

    def compute_motion_vector(self, ref_image, image):
        index_matrix = np.arange(1, 82).reshape((9, 9))

        H, W = ref_image.shape
        motion_vectors_indices = np.zeros((H // 8, W // 8), dtype=int)

        last_i = H - 8
        last_j = W - 8

        for i in range(0, H, 8):
            for j in range(0, W, 8):
                current_block = image[i:i + 8, j:j + 8]
                sse_min = float('inf')

                m_start = -4
                n_start = -4
                m_end = 4
                n_end = 4

                if i == 0:
                    m_start = 0
                if j == 0:
                    n_start = 0
                if i == last_i:
                    m_end = 0
                if j == last_j:
                    n_end = 0

                for m in range(m_start, m_end + 1):
                    for n in range(n_start, n_end + 1):
                        ref_block = ref_image[i + m:i + m + 8, j + n:j + n + 8]
                        sse_now = np.sum((ref_block - current_block) ** 2)

                        if sse_now < sse_min:
                            sse_min = sse_now
                            motion_vectors_indices[i // 8, j // 8] = index_matrix[m + 4, n + 4]
        return motion_vectors_indices

    def reconstruct_with_motion_vector(self, ref_image, motion_vectors):
        h, w = motion_vectors.shape
        if ref_image.ndim == 3:
            image = np.zeros((h * 8, w * 8, ref_image.shape[2]), dtype=ref_image.dtype)
        else:
            image = np.zeros((h * 8, w * 8), dtype=ref_image.dtype)

        for i in range(h):
            for j in range(w):
                # Calculate motion vector indices
                idx_i = (motion_vectors[i, j] - 1) // 9 - 4  # in [-4, 4]
                idx_j = (motion_vectors[i, j] - 1) - 9 * (idx_i + 4) - 4  # in [-4, 4]

                # Calculate block positions
                ref_i_start = i * 8 + idx_i
                ref_i_end = (i + 1) * 8 + idx_i
                ref_j_start = j * 8 + idx_j
                ref_j_end = (j + 1) * 8 + idx_j

                # Copy the block from reference image to reconstructed image
                image[i * 8: (i + 1) * 8, j * 8: (j + 1) * 8] = ref_image[ref_i_start: ref_i_end,
                                                                       ref_j_start: ref_j_end]
        # YOUR CODE ENDS HERE
        return image