from ivclab.utils import imread
from ivclab.entropy import stats_marg, calc_entropy, HuffmanCoder
from ivclab.image import three_pixels_predictor, inverse_three_pixels_predictor
import numpy as np
from ivclab. utils import calc_psnr
import matplotlib.pyplot as plt

# For this exercise, you need to implement three_pixels_predictor and
# _predict_from_neighbors functions in ivclab.image.predictive file.
# You can run ch2 tests to make sure they are implemented correctly

lena_img = imread(f'../data/lena.tif')
residual_image_Y, residual_image_CbCr = three_pixels_predictor(lena_img, subsample_color_channels=False)
merged_residuals = np.concatenate([residual_image_Y.ravel(), residual_image_CbCr.ravel()])
pmf = stats_marg(merged_residuals, np.arange(-255, 257))
entropy = calc_entropy(pmf)

print(f"Three pixels predictive coding entropy of lena.tif: H={entropy:.2f} bits/pixel")


# 2.2 Huffman Coding
lena_small_img = imread(f'../data/lena_small.tif')
residual_small_Y, residual_small_CbCr = three_pixels_predictor(lena_small_img, subsample_color_channels=False)
merged_small_residuals = np.concatenate([residual_small_Y.ravel(), residual_small_CbCr.ravel()])
pmf_small = stats_marg(merged_small_residuals, np.arange(-255, 257))

huffman = HuffmanCoder()
huffman.train(pmf_small)

# Shift residuals of lena to make all values non-negative (range: 0 ~ 510)
residuals = merged_residuals.astype(np.int32) + 255  # now in range [0, 510]

compressed, bitrate = huffman.encode(residuals)
print(f"Total bitrate: {bitrate}")

decoded_message = huffman.decode(compressed, message_length=len(residuals))
decoded_residuals = decoded_message - 255  # shift back to original [-255, 255]

# Get the code length of every codewords
code_lens = huffman.get_codeword_lengths()
print(f"Number of codewords: {len(code_lens)}")
print(f"Max. codeword length: {np.max(code_lens)}")
print(f"Min. codeword length: {np.min(code_lens)}")

# Calculate the bit rate
original_bits = lena_img.size * 24
compression_ratio = original_bits / bitrate
bitrate_pixel = bitrate / lena_img.size
print(f"Bitrate: {bitrate_pixel:.2f} bits/pixel")
print(f"Compression Ratio: {compression_ratio:.2f}")

# Calculate PSNR
Y_size = residual_image_Y.size
Y_recon = decoded_residuals[:Y_size].reshape(residual_image_Y.shape)
CbCr_recon = decoded_residuals[Y_size:].reshape(residual_image_CbCr.shape)
reconstructed_img = inverse_three_pixels_predictor(Y_recon, CbCr_recon, subsample_color_channels=False)
psnr_val = calc_psnr(lena_img.astype(np.float32), reconstructed_img.astype(np.float32))
print(f"Overall image PSNR: {psnr_val:.2f} dB")