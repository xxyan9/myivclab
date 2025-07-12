from ivclab.utils import imread, imshow, calc_psnr
import matplotlib.pyplot as plt

# read images
img_lena = imread('../data/lena.tif')
img_lena_compressed = imread('../data/lena_compressed.tif')
img_monarch = imread('../data/monarch.tif')
img_monarch_compressed = imread('../data/monarch_compressed.tif')

# YOUR CODE STARTS HERE

# Compute the PSNR values for Lena and Monarch compression
psnr_lena = calc_psnr(img_lena, img_lena_compressed)
psnr_monarch = calc_psnr(img_monarch, img_monarch_compressed)

# YOUR CODE ENDS HERE

# print metrics
print(f'PSNR of lena.tif is {psnr_lena:.3f} dB\n')
print(f'PSNR of monarch.tif is {psnr_monarch:.3f} dB\n')

# plot images
fig1, axs = plt.subplots(2, 2)
imshow(axs[0][0], img_lena, title='Original Lena Image')
imshow(axs[0][1], img_lena_compressed, title='Compressed Lena Image')
imshow(axs[1][0], img_monarch, title='Original Monarch Image')
imshow(axs[1][1], img_monarch_compressed, title='Compressed Monarch Image')
# plt.show()

# E1-1.f
bitrates = 8
fig2, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(bitrates, psnr_lena, 'o-')
ax1.set_xlabel('Bit Rate (bit/pixel)')
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('Rate-Distortion Point of Lena')

ax2.plot(bitrates, psnr_monarch, 'o-')
ax2.set_xlabel('Bit Rate (bit/pixel)')
ax2.set_ylabel('PSNR (dB)')
ax2.set_title('Rate-Distortion Point of Monarch')

plt.show()


