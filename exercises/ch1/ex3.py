from ivclab.utils import imread, imshow, calc_psnr
from ivclab.signal import FilterPipeline, lowpass_filter, downsample, upsample, rgb2gray
import numpy as np
import matplotlib.pyplot as plt

image = imread('../data/satpic1.bmp')

kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
n_kernel = kernel/np.sum(kernel)  # Normalize the kernel

# E1-2.b
fft_kernel = np.fft.fft2(n_kernel)  # Compute the 2D FFT of the kernel
fft_shifted = np.fft.fftshift(fft_kernel)   # Shift the zero-frequency component to the center
magnitude_spectrum = np.abs(fft_shifted)    # Compute the magnitude spectrum (in dB)

# Plot the frequency response
plt.figure()
plt.matshow(magnitude_spectrum, cmap='viridis', fignum=1)
plt.colorbar(label='Magnitude (dB)')
plt.title('Frequency Response (log scale)')


# E1-2.c
filtered = lowpass_filter(image, n_kernel)
difference = np.abs(image.astype(np.float32) - filtered.astype(np.float32))
n_difference = (difference / difference.max())  # Normalize the difference to [0, 1]

# Plot images
fig1, axs = plt.subplots(1, 3)
imshow(axs[0], image, title='Original Image')
imshow(axs[1], filtered, title='Filtered Image')
imshow(axs[2], n_difference, title='Difference')


# E1-2.d
downsampled = downsample(filtered)  # Down sampling to [H/2, W/2, C]
upsampled = upsample(downsampled)   # Up sampling to [H, W, C] (zero-padding)

# Plot images
fig2, axs = plt.subplots(1, 3)
imshow(axs[0], filtered, title='Filtered Image')
imshow(axs[1], downsampled, title='Downsampled Image')
imshow(axs[2], upsampled, title='Upsampled Image')


# E1-2.e
pipeline = FilterPipeline(kernel=kernel)

recon_image_not_pre = pipeline.filter_img(image, False)
recon_image_pre = pipeline.filter_img(image, True)

recon_not_pre = recon_image_not_pre.astype(np.float32)
recon_pre = recon_image_not_pre.astype(np.float32)

psnr_not_pre = calc_psnr(image, recon_image_not_pre)
psnr_pre = calc_psnr(image, recon_image_pre)


# Plot images
fig3, axs = plt.subplots(1, 2)
imshow(axs[0], recon_image_not_pre, title='Not Pre-filtered Image')
imshow(axs[1], recon_image_pre / 255.0, title='Pre-filtered Image')
plt.show()

print(f"Reconstructed image, not prefiltered, PSNR = {psnr_not_pre:.2f} dB")
print(f"Reconstructed image, prefiltered, PSNR = {psnr_pre:.2f} dB")
