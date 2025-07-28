from ivclab.video import VideoCodec
from ivclab.image import IntraCodec
import numpy as np
from ivclab.utils import imread, calc_psnr
import matplotlib.pyplot as plt
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
import deblock

lena_small = imread('../data/lena_small.tif')

'''
# 1. chapter 3: Intra Coding
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

    psnr = np.mean(image_psnrs)
    bpp = np.mean(image_bpps)
    all_bpps.append(bpp)
    all_PSNRs.append(psnr)
    print(f"Q-Scale {q_scale}, PSNR: {psnr:.2f} dB, bpp: {bpp:.2f}")

print('-'*12)
ch3_bpps = np.array(all_bpps)
ch3_psnrs = np.array(all_PSNRs)

# 2. Chapter 4: Video Coding
images = []
for i in range(20, 40 + 1):
    images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

all_bpps = list()
all_psnrs = list()

for q_scale in [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]:
# for q_scale in [1]:
    video_codec = VideoCodec(quantization_scale=q_scale)
    video_codec.intra_codec.train_huffman_from_image(lena_small, is_source_rgb=True)
    bpps = list()
    psnrs = list()
    for frame_num, image in enumerate(images):
        reconstructed_image, bitstream, bitsize = video_codec.encode_decode(image, frame_num=frame_num)
        reconstructed_image = ycbcr2rgb(reconstructed_image)
        bpp = bitsize/(image.size/3)
        psnr = calc_psnr(image, reconstructed_image)
        bpps.append(bpp)
        psnrs.append(psnr)

    all_bpps.append(np.mean(bpps))
    all_psnrs.append(np.mean(psnrs))
    print(f"Q-Scale {q_scale}, Average PSNR: {np.mean(psnrs):.2f} dB Average bpp: {np.mean(bpps):.2f}")
    print('-'*12)

ch4_bpps = np.array(all_bpps)
ch4_psnrs = np.array(all_psnrs)


# 3. Post-deblocking Filter for Intracodec
images = []
for i in range(20, 40 + 1):
    images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

all_bpps = list()
all_psnrs = list()

for q_scale in [0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10]:
# for q_scale in [1.0]:
    intra_codec = IntraCodec(quantization_scale=q_scale)
    intra_codec.train_huffman_from_image(lena_small)

    bpps = list()
    psnrs = list()

    for frame_num, image in enumerate(images):
        message, bitsize = intra_codec.intra_encode(image, return_bpp=True, is_source_rgb=True)
        reconstructed_img = intra_codec.intra_decode(message, image.shape)

        # Apply post-deblocking filter
        qp_index = deblock.qscale_to_qp_index(q_scale)
        deblocked_img = deblock.deblock(reconstructed_img, qp_index)
        reconstructed_img = ycbcr2rgb(deblocked_img)

        bpp = bitsize / (image.size / 3)
        psnr = calc_psnr(image, reconstructed_img)
        bpps.append(bpp)
        psnrs.append(psnr)

    all_bpps.append(np.mean(bpps))
    all_psnrs.append(np.mean(psnrs))
    print(f"Q-Scale {q_scale}, PSNR: {psnr:.2f} dB, bpp: {bpp:.2f}")

print('-' * 12)
ch5_deblock_bpps = np.array(all_bpps)
ch5_deblock_psnrs = np.array(all_psnrs)
np.save('../data/ch5_deblock_bpps.npy', ch5_deblock_bpps)
np.save('../data/ch5_deblock_psnrs.npy', ch5_deblock_psnrs)


# 4. Adaptive Quantization
images = []
for i in range(20, 40 + 1):
    images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

all_bpps = list()
all_psnrs = list()

for q_scale in [0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10]:
    intra_codec = IntraCodec(quantization_scale=q_scale)
    intra_codec.train_huffman_from_image(lena_small)

    if q_scale in [0.3, 0.7, 1.0, 1.5]:
        intra_codec.use_adaptive_quant = True

    image_bpps = list()
    image_psnrs = list()

    for frame_num, image in enumerate(images):
        message, bitrate = intra_codec.intra_encode(image, return_bpp=True, is_source_rgb=True)
        reconstructed_img = intra_codec.intra_decode(message, image.shape)
        reconstructed_img = ycbcr2rgb(reconstructed_img)
        bpp = bitrate / (image.size / 3)
        psnr = calc_psnr(image, reconstructed_img)
        image_bpps.append(bpp)
        image_psnrs.append(psnr)

    bpp = np.mean(image_bpps)
    psnr = np.mean(image_psnrs)
    all_bpps.append(bpp)
    all_psnrs.append(psnr)
    print(f"Q Scale: {q_scale}, PSNR: {psnr:.2f} dB, bpp: {bpp:.2f}")

print('-' * 12)
ch5_aquant_bpps = np.array(all_bpps)
ch5_aquant_psnrs = np.array(all_psnrs)
np.save('../data/ch5_aquant_bpps.npy', ch5_aquant_bpps)
np.save('../data/ch5_aquant_psnrs.npy', ch5_aquant_psnrs)
'''

# 5. Block Mode Decision
images = []
for i in range(20, 40 + 1):
    images.append(imread(f'../data/foreman20_40_RGB/foreman00{i}.bmp'))

all_bpps = list()
all_psnrs = list()

for q_scale in [0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10]:
    intra_codec = IntraCodec(quantization_scale=q_scale)
    intra_codec.train_huffman_from_image(lena_small)
    intra_codec.use_mode_decision = True

    image_bpps = list()
    image_psnrs = list()

    for frame_num, image in enumerate(images):
        message, bitrate = intra_codec.intra_encode(image, return_bpp=True, is_source_rgb=True)
        reconstructed_img = intra_codec.intra_decode(message, image.shape)
        reconstructed_img = ycbcr2rgb(reconstructed_img)
        bpp = bitrate / (image.size / 3)
        psnr = calc_psnr(image, reconstructed_img)
        image_bpps.append(bpp)
        image_psnrs.append(psnr)

    bpp = np.mean(image_bpps)
    psnr = np.mean(image_psnrs)
    all_bpps.append(bpp)
    all_psnrs.append(psnr)
    print(f"Q Scale: {q_scale}, PSNR: {psnr:.2f} dB, bpp: {bpp:.2f}")

print('-' * 12)
ch5_mdecision_bpps = np.array(all_bpps)
ch5_mdecision_psnrs = np.array(all_psnrs)
np.save('../data/ch5_aquant_bpps.npy', ch5_mdecision_bpps)
np.save('../data/ch5_aquant_psnrs.npy', ch5_mdecision_psnrs)

# 6. Half pel motion compensation
# 7. Quarter pel motion compensation

ch3_bpps = np.load('../data/ch3_bpps.npy')
ch3_psnrs = np.load('../data/ch3_psnrs.npy')
ch4_bpps = np.load('../data/ch4_bpps.npy')
ch4_psnrs = np.load('../data/ch4_psnrs.npy')
ch5_deblock_bpps = np.load('../data/ch5_deblock_bpps.npy')
ch5_deblock_psnrs = np.load('../data/ch5_deblock_psnrs.npy')
ch5_aquant_bpps = np.load('../data/ch5_aquant_bpps.npy')
ch5_aquant_psnrs = np.load('../data/ch5_aquant_psnrs.npy')
ch5_mdecision_bpps = np.load('../data/ch5_mdecision_bpps.npy')
ch5_mdecision_psnrs = np.load('../data/ch5_mdecision_psnrs.npy')
ch5_halfpel_bpps = np.load('../data/ch5_halfpel_bpps.npy')
ch5_halfpel_psnrs = np.load('../data/ch5_halfpel_psnrs.npy')
# ch5_quarterpel_bpps = np.load('../data/ch5_quarterpel_bpps.npy')
# ch5_quarterpel_psnrs = np.load('../data/ch5_quarterpel_psnrs.npy')

plt.figure()
plt.xlabel('Bitrate (bpp)')
plt.ylabel('PSNR [dB]')
plt.title('Rate-Distortion Curve')
plt.plot(ch3_bpps, ch3_psnrs, linestyle='--', marker='+', color='orange', label='Image Codec Solution')
plt.plot(ch4_bpps, ch4_psnrs, linestyle='--', marker='v', color='blue', label='Video Codec Solution')
# plt.plot(ch5_deblock_bpps, ch5_deblock_psnrs, linestyle='--', marker='x',
#          color='green', label='Intra Opt: Post-deblocking')
plt.plot(ch5_aquant_bpps, ch5_aquant_psnrs, linestyle='--', marker='.',
         color='purple', label='Intra Opt: Adaptive quantize')
plt.plot(ch5_mdecision_bpps, ch5_mdecision_psnrs, linestyle='--', marker='>',
         color='purple', label='Intra Opt: Block mode decision')
plt.plot(ch5_halfpel_bpps, ch5_halfpel_psnrs, linestyle='--', marker='^',
         color='red', label='Video Opt: Halfpel')
# plt.plot(ch5_quarterpel_bpps, ch5_quarterpel_psnrs, linestyle='--', marker='s',
#          color='yellow', label='Video Opt: Quarterfpel')
plt.legend()
plt.show()