import numpy as np
from scipy.fft import dct, idct
import matplotlib.pyplot as plt
from ivclab.utils import imread, imshow
from ivclab.signal import rgb2ycbcr


def apply_dct_and_reconstruct(img_path, percent_to_remove):
    lena = imread(f'../data/lena_gray.tif')
    if len(lena.shape) == 3:
        lena = rgb2ycbcr(lena)

    # Apply
    dct_coeffs = dct(dct(lena.T, norm='ortho').T, norm='ortho')

    # Find the top coefficients
    abs_coeffs = np.abs(dct_coeffs)
    threshold = np.percentile(abs_coeffs, 100 - percent_to_remove)
    mask = abs_coeffs < threshold

    # Reset to zero and reconstruct the image
    modified_dct = dct_coeffs * mask
    reconstructed = idct(idct(modified_dct.T, norm='ortho').T, norm='ortho')

    return lena, reconstructed, percent_to_remove


original, recon_1, _ = apply_dct_and_reconstruct('lena_gray.tif', 1)
_, recon_5, _ = apply_dct_and_reconstruct('lena_gray.tif', 5)
_, recon_10, _ = apply_dct_and_reconstruct('lena_gray.tif', 10)

# Plot
plt.figure(figsize=(8, 8))
plt.subplot(221), plt.imshow(original, cmap='gray'), plt.title('Original')
plt.subplot(222), plt.imshow(recon_1, cmap='gray'), plt.title('Top 1% removed')
plt.subplot(223), plt.imshow(recon_5, cmap='gray'), plt.title('Top 5% removed')
plt.subplot(224), plt.imshow(recon_10, cmap='gray'), plt.title('Top 10% removed')
plt.show()