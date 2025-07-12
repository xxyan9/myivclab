from ivclab.utils import imread, imshow, calc_psnr
from ivclab.image import yuv420compression
import numpy as np
import matplotlib.pyplot as plt

image = imread('../data/sail.tif')
# image = imread('../data/lena.tif')

recon_image = yuv420compression(image)

psnr_recon = calc_psnr(image, recon_image)

print(f"Reconstructed image, not prefiltered, PSNR = {psnr_recon:.2f} dB")

