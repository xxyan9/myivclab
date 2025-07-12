import numpy as np
from scipy.signal import decimate, resample
from ivclab.signal import rgb2gray, rgb2ycbcr, ycbcr2rgb
import matplotlib.pyplot as plt
from ivclab.utils import imshow

def yuv420compression(image: np.ndarray):
    """
    Steps:
    1. Convert an image from RGB to YCbCr
    2. Compress the image
        A. Pad the image with 4 pixels symmetric pixels on each side
        B. Downsample only Cb and Cr channels with prefiltering (use scipy.signal.decimate for it)
        C. Crop the image 2 pixels from each side to get rid of padding
    3. Apply rounding to Y, Cb and Cr channels
    4. Decompress the image
        A. Pad the image with 2 pixels symmetric pixels on each side
        B. Upsample Cb and Cr channels (use scipy.signal.resample for it)
        C. Crop the image 4 pixels from each side to get rid of padding
    5. Convert the YCbCr image back to RGB

    image: np.array of shape [H, W, C]

    returns 
        output_image: np.array of shape [H, W, C]
    """
    # Cast image to floating point
    image = image * 1.0

    # YOUR CODE STARTS HERE
    # Step 1: Convert from RGB to YCbCr
    ycbcr = rgb2ycbcr(image)
    Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]

    # Step 2.A: Pad Cb and Cr with 4-pixel symmetric padding
    Cb_padded = np.pad(Cb, pad_width=4, mode='symmetric')
    Cr_padded = np.pad(Cr, pad_width=4, mode='symmetric')

    # Step 2.B: Downsample Cb and Cr (vertically and horizontally)
    Cb_down = decimate(decimate(Cb_padded, q=2, axis=0, ftype='fir', zero_phase=True),
                       q=2, axis=1, ftype='fir', zero_phase=True)
    Cr_down = decimate(decimate(Cr_padded, q=2, axis=0, ftype='fir', zero_phase=True),
                       q=2, axis=1, ftype='fir', zero_phase=True)

    # Step 2.C: Crop back center to original downsampled size
    h, w = image.shape[:2]
    Cb_down = Cb_down[2:-2, 2:-2]
    Cr_down = Cr_down[2:-2, 2:-2]

    # Step 3: Round all channels
    Y = np.round(Y)
    Cb_down = np.round(Cb_down)
    Cr_down = np.round(Cr_down)

    # Step 4.A: Pad Cb_down and Cr_down with 2-pixel symmetric padding
    Cb_down_padded = np.pad(Cb_down, pad_width=2, mode='symmetric')
    Cr_down_padded = np.pad(Cr_down, pad_width=2, mode='symmetric')

    # Step 4.B: Upsample back to original size (vertically and horizontally)
    Cb_up = resample(resample(Cb_down_padded, h+8, axis=0), w+8, axis=1)
    Cr_up = resample(resample(Cr_down_padded, h+8, axis=0), w+8, axis=1)

    # Step 4.C: Crop back to exact shape
    Cb_up = Cb_up[4:-4, 4:-4]
    Cr_up = Cr_up[4:-4, 4:-4]

    # Step 5: Convert from YCbCr back to RGB
    ycbcr_reconstructed = np.stack((Y, Cb_up, Cr_up), axis=2)
    output = ycbcr2rgb(ycbcr_reconstructed)

    # Plot images
    image_plt = image.astype(np.uint8)
    ycbcr_plt = ycbcr.astype(np.uint8)
    output_plt = output.astype(np.uint8)
    fig, axs = plt.subplots(1, 3)
    imshow(axs[0], image_plt, title='Original Image')
    imshow(axs[1], ycbcr_plt, title='YCbCr Image')
    imshow(axs[2], output_plt, title='Reconstructed Image')
    plt.show()

    # YOUR CODE ENDS HERE

    # Y channel: full resolution
    Y_bits = h * w * 8

    # Cb and Cr: downsampled to (H/2, W/2)
    Cb_bits = (h // 2) * (w // 2) * 8
    Cr_bits = (h // 2) * (w // 2) * 8

    total_bits = Y_bits + Cb_bits + Cr_bits

    # Bit rate = total bits / total number of original pixels
    total_pixels = h * w
    bit_rate = total_bits / total_pixels
    print(f"bit rate={bit_rate:2f}")

    # Cast output to integer again
    output = np.round(output).astype(np.uint8)
    return output
