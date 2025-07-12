from ivclab.utils import imread, calc_psnr
from ivclab.signal import FilterPipeline
import numpy as np

image = imread('data/satpic1.bmp')

lowpass_kernel = ...
pipeline = FilterPipeline(kernel=lowpass_kernel)

recon_image_not_pre = pipeline.filter_img(image, False)
recon_image_pre = pipeline.filter_img(image, True)

psnr_not_pre = calc_psnr(image, recon_image_not_pre)
psnr_pre = calc_psnr(image, recon_image_pre)

print(f"Reconstructed image, not prefiltered, PSNR = {psnr_not_pre:.2f} dB")
print(f"Reconstructed image, prefiltered, PSNR = {psnr_pre:.2f} dB")
