from ivclab.utils import imread, imshow, calc_mse, calc_psnr
from ivclab.signal import rgb2gray
import matplotlib.pyplot as plt

# read images
img_lena = imread('../data/lena.tif')
img_lena_gray = rgb2gray(img_lena)
img_smandril = imread('../data/smandril.tif')
img_smandril_gray = rgb2gray(img_smandril)
img_smandril_rec = imread('../data/smandril_rec.tif')

# plot images
fig, axs = plt.subplots(2,2)
imshow(axs[0][0], img_lena, title='Original Lena Image')
imshow(axs[0][1], img_lena_gray, title='Compressed Lena Image')
imshow(axs[1][0], img_smandril, title='Original Smandril Image')
imshow(axs[1][1], img_smandril_gray, title='Compressed Smandril Image')
plt.show()

# E1-1.d Calculate the MSE & PSNR of Smandril
mse = calc_mse(img_smandril, img_smandril_rec)
psnr = calc_psnr(img_smandril, img_smandril_rec)

print(f"MSE: {mse:.2f}")
print(f"PSNR: {psnr:.2f} dB")

# E1-1.e Plot bit/pixel v.s. PSNR
bitrates = 8
plt.plot(bitrates, psnr, 'o-')
plt.xlabel('Bit Rate (bit/pixel)')
plt.ylabel('PSNR (dB)')
plt.title('Rate-Distortion Point')
plt.show()

