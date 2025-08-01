import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.image import IntraCodec
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.video import MotionCompensator
from exercises.ch5.modedecision import block_mode_decision
from ivclab.utils import calc_psnr
import matplotlib.pyplot as plt
from exercises.ch5 import deblock


class VideoCodec:

    def __init__(self,
                 quantization_scale=1.0,
                 bounds=(-1000, 4000),
                 end_of_block=4000,
                 block_shape=(8, 8),
                 search_range=4,
                 use_mode_decision=False
                 ):

        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape
        self.search_range = search_range
        self.use_mode_decision = use_mode_decision

        self.intra_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds, end_of_block=end_of_block, block_shape=block_shape)
        self.residual_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds, end_of_block=end_of_block, block_shape=block_shape)
        self.motion_comp = MotionCompensator(search_range=search_range)
        # self.motion_huffman = HuffmanCoder(lower_bound=-((2*search_range + 1)**2 - 1)//2)
        self.motion_huffman = HuffmanCoder(lower_bound=0)
        self.decoder_recon = None

    def encode_decode(self, frame, frame_num=0):
        if frame_num == 0:
        # YOUR CODE STARTS HERE
            # ---Intra mode---
            bitstream, residual_bitsize = self.intra_codec.intra_encode(frame, return_bpp=True)
            self.decoder_recon = self.intra_codec.intra_decode(bitstream, frame.shape)
            motion_bitsize = 0  # No motion in intra frame
            bitsize = residual_bitsize + motion_bitsize
        else:
            # ---Inter mode---
            # Perform color transform
            curr_ycbcr = rgb2ycbcr(frame)
            ref_ycbcr = self.decoder_recon

            # Motion vector computation
            motion_vector = self.motion_comp.compute_motion_vector(ref_ycbcr[..., 0], curr_ycbcr[..., 0])

            # Perform motion compensation
            recon_pred_frame_ycbcr = self.motion_comp.reconstruct_with_motion_vector(ref_ycbcr, motion_vector)

            # Compute residual
            residual = curr_ycbcr - recon_pred_frame_ycbcr

            # Huffman encode motion vector
            mv_flat = motion_vector.flatten()

            if frame_num == 1:
                symbol_range = np.arange(0, 81 + 2)
                mv_pmf = stats_marg(mv_flat, symbol_range)
                self.motion_huffman.train(mv_pmf)
                if self.use_mode_decision:
                    self.residual_codec.train_huffman_from_image(residual, is_source_rgb=False, is_block_residual=True)
                else:
                    self.residual_codec.train_huffman_from_image(residual, is_source_rgb=False)

            # Block mode decision
            if self.use_mode_decision:
                recon_ycbcr, bitsize = block_mode_decision(
                    curr_ycbcr, ref_ycbcr, motion_comp=self.motion_comp,
                    motion_huffman=self.motion_huffman, q_scale=self.quantization_scale,
                    intra_codec=self.intra_codec, residual_codec=self.residual_codec
                )
                self.decoder_recon = recon_ycbcr
                bitstream = None
            else:
                motion_stream, motion_bitsize = self.motion_huffman.encode(mv_flat)

                # Residual encode
                residual_stream, residual_bitsize = self.residual_codec.intra_encode(
                    residual, return_bpp=True, is_source_rgb=False)

                # Decoding motion vector
                mv_decoded = self.motion_huffman.decode(motion_stream, motion_vector.size)
                mv_decoded = mv_decoded.reshape(motion_vector.shape)

                # Motion compensation (decoder)
                recon_pred_frame_ycbcr_docoded = self.motion_comp.reconstruct_with_motion_vector(ref_ycbcr, mv_decoded)

                # Residual decoding
                residual_recon = self.residual_codec.intra_decode(residual_stream, frame.shape)
                self.decoder_recon = recon_pred_frame_ycbcr_docoded + residual_recon

                bitstream = (motion_stream, residual_stream)
                # YOUR CODE ENDS HERE
                bitsize = residual_bitsize + motion_bitsize
        return self.decoder_recon, bitstream, bitsize
