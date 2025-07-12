from ivclab.utils import imread
from ivclab.entropy import stats_marg, calc_entropy
from ivclab.image import single_pixel_predictor
import numpy as np

# For this exercise, you need to implement 
# single_pixel_predictor function in ivclab.image.predictive file.
# You can run ch2 tests to make sure they are implemented correctly

lena_img = imread(f'../data/lena.tif')
residual_image = single_pixel_predictor(lena_img)
pmf = stats_marg(residual_image, np.arange(-256, 256))
entropy = calc_entropy(pmf)

print(f"Single pixel predictive coding entropy of lena.tif: H={entropy:.2f} bits/pixel")
