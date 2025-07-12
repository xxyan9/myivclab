import numpy as np
from ivclab.image import IntraCodec
from ivclab.utils import imread, calc_psnr
import matplotlib.pyplot as plt

# Implement the IntraCodec and all the necessary modules
# For each given quantization scale in the handout:
# - Initialize a new IntraCodec
# - Use lena_small to train Huffman coder of IntraCodec.
# - Compress and decompress 'lena.tif'
# - Measure bitrate and PSNR on lena
# Plot all the measurements in a Rate Distortion plot

lena = imread(f'../data/lena.tif')
lena_small = imread(f'../data/lena_small.tif')
H, W, C = lena.shape
all_PSNRs = list()
all_bpps = list()

# YOUR CODE STARTS HERE
quantization_scales = [0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10]

for q in quantization_scales:
    intracodec = IntraCodec(quantization_scale=q, bounds=(-1000, 4000))
    intracodec.train_huffman_from_image(lena_small)
    symbols, bitsize = intracodec.intra_encode(lena, return_bpp=True)
    reconstructed_img = intracodec.intra_decode(symbols, lena.shape)

    psnr = calc_psnr(lena, reconstructed_img)
    bpp = bitsize / (lena.size / 3)

    all_PSNRs.append(psnr)
    all_bpps.append(bpp)
    plt.imshow(reconstructed_img.astype(np.uint8))
    print(reconstructed_img[0:3,0:3,1])
# YOUR CODE ENDS HERE

all_bpps = np.array(all_bpps)
all_PSNRs = np.array(all_PSNRs)

print(all_bpps, all_PSNRs)
plt.plot(all_bpps, all_PSNRs, marker='o')
plt.xlabel('bpp')
plt.ylabel('PSNR[dB]')

plt.show()
