import numpy as np

def rgb2gray(image: np.array):
    """
    Computes the grayscale version of the image. 

    image: np.array of shape [H, W, C]

    returns 
        output_image: np.array of shape [H, W, 1]
    """
    output_image = np.mean(image, axis=-1, keepdims=True)
    return output_image

def rgb2ycbcr(image: np.array):
    """
    Converts an RGB image to its YCbCr version. 

    image: np.array of shape [H, W, 3]

    returns 
        output_image: np.array of shape [H, W, 3]
    """
    image = np.float32(image)
    output_image = np.zeros_like(image)

    # YOUR CODE STARTS HERE
    r = image[..., 0].astype(np.float32)
    g = image[..., 1].astype(np.float32)
    b = image[..., 2].astype(np.float32)
    
    # YCbCr conversion formula (ITU-R BT.601)
    output_image[..., 0] = 0.299 * r + 0.587 * g + 0.114 * b  # Y
    output_image[..., 1] = -0.169 * r - 0.331 * g + 0.5 * b  # Cb
    output_image[..., 2] = 0.5 * r - 0.419 * g - 0.081 * b  # Cr
    # YOUR CODE ENDS HERE

    return output_image

def ycbcr2rgb(image: np.array):
    """
    Converts an YCbCr image to its RGB version. 

    image: np.array of shape [H, W, 3]

    returns 
        output_image: np.array of shape [H, W, 3]
    """
    output_image = np.zeros_like(image)

    # YOUR CODE STARTS HERE
    y = image[..., 0].astype(np.float32)
    cb = image[..., 1].astype(np.float32)
    cr = image[..., 2].astype(np.float32)
    
    # RGB conversion formula (ITU-R BT.601)
    output_image[..., 0] = y + 1.402 * cr  # R
    output_image[..., 1] = y - 0.344 * cb - 0.714 * cr  # G
    output_image[..., 2] = y + 1.772 * cb  # B
    
    # Clip values to valid range [0, 255]
    output_image = np.clip(output_image, 0, 255)
    # YOUR CODE ENDS HERE

    return output_image.astype(np.uint8)