from ivclab.utils import imread
from ivclab.entropy import stats_cond
import matplotlib.pyplot as plt
import numpy as np

# For this exercise, you need to implement 
# stats_cond function in ivclab.entropy.probability file.
# You can run ch2 tests to make sure they are implemented correctly

lena_img = imread(f'../data/lena.tif')
pixel_range = np.arange(256)
cond_entropy = stats_cond(lena_img, pixel_range)

print(f"Conditional entropy of lena.tif: H={cond_entropy:.2f} bits/pixel pair")
