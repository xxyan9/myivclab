from ivclab.video import VideoCodec
from ivclab.image import IntraCodec
import numpy as np
from ivclab.utils import imread, calc_psnr
import matplotlib.pyplot as plt
from ivclab.signal import rgb2ycbcr, ycbcr2rgb

lena_small = imread('../data/lena_small.tif')

images = []
for i in range(20, 40 + 1):
    images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

all_bpps = list()
all_psnrs = list()

for q_scale in [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]:
    video_codec = VideoCodec(quantization_scale=q_scale)
    video_codec.intra_codec.train_huffman_from_image(lena_small, is_source_rgb=True)
    bpps = list()
    psnrs = list()
    for frame_num, image in enumerate(images):
        reconstructed_image, bitstream, bitsize = video_codec.encode_decode(image, frame_num=frame_num)
        bpp = bitsize/(image.size/3)
        psnr = calc_psnr(image, reconstructed_image)
        print(f"Frame:{frame_num} PSNR: {psnr:.2f} dB bpp: {bpp:.2f}")
        bpps.append(bpp)
        psnrs.append(psnr)

    all_bpps.append(np.mean(bpps))
    all_psnrs.append(np.mean(psnrs))
    print(f"Q-Scale {q_scale}, Average PSNR: {np.mean(psnrs):.2f} dB Average bpp: {np.mean(bpps):.2f}")
    print('-'*12)

ch4_bpps = np.array(all_bpps)
ch4_psnrs = np.array(all_psnrs)

np.save('../data/ch4_bpps.npy', ch4_bpps)
np.save('../data/ch4_psnrs.npy', ch4_psnrs)

images = []
for i in range(20, 40 + 1):
    images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

all_bpps = list()
all_PSNRs = list()

for q_scale in [0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10]:
# for q_scale in [1.0]:
    intracodec = IntraCodec(quantization_scale=q_scale)
    intracodec.train_huffman_from_image(lena_small)
    image_psnrs = list()
    image_bpps = list()
    for i in range(len(images)):
        img = images[i]
        H, W, C = img.shape
        message, bitrate = intracodec.intra_encode(img, return_bpp=True, is_source_rgb=True)
        reconstructed_img = intracodec.intra_decode(message, img.shape)
        reconstructed_img = ycbcr2rgb(reconstructed_img)
        psnr = calc_psnr(img, reconstructed_img)
        bpp = bitrate / (H * W)
        image_psnrs.append(psnr)
        image_bpps.append(bpp)
        print(f"Frame:{i} PSNR: {psnr:.2f} dB bpp: {bpp:.2f}")
    psnr = np.mean(image_psnrs)
    bpp = np.mean(image_bpps)
    all_bpps.append(bpp)
    all_PSNRs.append(psnr)
    print(f"Q-Scale {q_scale}, PSNR: {psnr:.2f} dB, bpp: {bpp:.2f}")

ch3_bpps = np.array(all_bpps)
ch3_psnrs = np.array(all_PSNRs)

ch3_bpps = np.load('../data/ch3_bpps.npy')
ch3_psnrs = np.load('../data/ch3_psnrs.npy')
ch4_bpps = np.load('../data/ch4_bpps.npy')
ch4_psnrs = np.load('../data/ch4_psnrs.npy')

plt.figure()
plt.xlabel('Bitrate (bpp)')
plt.ylabel('PSNR [dB]')
plt.title('Rate-Distortion Curve')
plt.plot(ch4_bpps, ch4_psnrs, linestyle='--', marker='s', color='blue', label='Video Codec Solution')
plt.plot(ch3_bpps, ch3_psnrs, linestyle='--', marker='s', color='orange', label='Image Codec Solution')
plt.legend()
plt.show()