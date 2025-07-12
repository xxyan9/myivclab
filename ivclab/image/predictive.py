import numpy as np
from scipy.signal import decimate
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from scipy.ndimage import zoom

def single_pixel_predictor(image):
    """
    Creates a residual image after a single pixel predictor for overlapping 
    pixel pairs. The right pixel is predicted from the left pixel with the formula
    R_pred = L * a1 where a1=1. This function returns the residual R - R_pred. For
    the first pixels of each row who don't have a left neighbor, it copies the values
    from the original image instead of making a prediction

    image: np.array of shape [H, W, C]

    returns 
        residual_image: np.array of shape [H, W, C]
    """
    # Convert image to floating points
    image = image * 1.0

    a1 = 1.0

    # Create residual image
    residual_image = np.zeros_like(image)

    # YOUR CODE STARTS HERE
    # Copy the first column directly (no left neighbor)
    residual_image[:, 0, :] = image[:, 0, :]
    # For all other pixels, compute residual: actual - predicted
    residual_image[:, 1:, :] = image[:, 1:, :] - a1 * image[:, :-1, :]  # Subtract the value of left neighboring pixel
    # YOUR CODE ENDS HERE

    residual_image = np.round(np.clip(residual_image, -255, 255))

    return residual_image


def _predict_from_neighbors(original, coefficients):
    """
    Helper function for the three pixel predictor. Here is the main computation:

    prediction(current) = coefficients * reconstruction(previous)
    error(current) = round(original(current) - prediction(current))
    reconstruction(current) = prediction(current) + error(current)

    We need to create two arrays, reconstruction and residual_error. They are already
    initialized such that the top row and the leftmost column of the original image
    is copied to them.
    
    It applies this over all pixels from top-left to bottom-right in order.

    Hint: Start from the second index in "for loops" of both directions

    original: np.array of shape [H, W, C]
    reconstruction: np.array of shape [H, W, C]
    residual_error: np.array of shape [H, W, C]
    coefficients: list of 3 floating point numbers (see lab slides for what they represent)

    returns 
        residual_error: np.array of shape [H, W, C]
    """
    H, W, C = original.shape

    reconstruction = np.zeros_like(original)
    reconstruction[0, :, :] = original[0, :, :]
    reconstruction[:, 0, :] = original[:, 0, :]

    residual_error = np.copy(reconstruction)

    # YOUR CODE STARTS HERE
    for h in range(1, H):
        for w in range(1, W):
            pre_s1 = coefficients[0] * reconstruction[h, w - 1, :]  # Left pixel
            pre_s2 = coefficients[1] * reconstruction[h - 1, w - 1, :]  # Left-top pixel
            pre_s3 = coefficients[2] * reconstruction[h - 1, w, :]  # Top pixel

            prediction = pre_s1 + pre_s2 + pre_s3
            error = np.round(original[h, w, :] - prediction)
            reconstruction[h, w, :] = prediction + error
            residual_error[h, w, :] = error
    # YOUR CODE ENDS HERE

    return residual_error

def three_pixels_predictor(image, subsample_color_channels=False):
    """
    Creates a residual image after a three pixels predictor.

    1. Convert the input image to YCbCr color space
    2. If subsample_color_channels, then subsample the Cb and Cr channels
        by 2, similar to the yuv420codec (use scipy.signal.decimate)
    3. Apply three pixel prediction with the given coefficients for Y and CbCr channels.
        You must use _predict_from_neighbors helper function
    4. Return the residual error images

    image: np.array of shape [H, W, C]

    returns 
        residual_image_Y: np.array of shape [H, W, 1]
        residual_image_CbCr: np.array of shape [H, W, 2] (or [H // 2, W // 2, 2] if subsampled)
    """
    # Convert image to floating points
    image = image * 1.0

    coefficients_Y = [7/8, -4/8, 5/8]
    coefficients_CbCr = [3/8, -2/8, 7/8]

    # YOUR CODE STARTS HERE
    # Convert RGB to YCbCr
    image_ycbcr = rgb2ycbcr(image)

    # Separate channels
    Y = image_ycbcr[:, :, 0:1]
    Cb = image_ycbcr[:, :, 1:2]
    Cr = image_ycbcr[:, :, 2:3]

    if subsample_color_channels:
        Cb = decimate(decimate(Cb, q=2, axis=0, ftype='fir', zero_phase=True), q=2, axis=1, ftype='fir', zero_phase=True)
        Cr = decimate(decimate(Cr, q=2, axis=0, ftype='fir', zero_phase=True), q=2, axis=1, ftype='fir', zero_phase=True)

    CbCr = np.concatenate((Cb, Cr), axis=2)

    # Apply prediction
    residual_image_Y = _predict_from_neighbors(Y, coefficients_Y)
    residual_image_CbCr = _predict_from_neighbors(CbCr, coefficients_CbCr)
    # YOUR CODE ENDS HERE

    residual_image_Y = np.round(np.clip(residual_image_Y, -255, 255)).astype(np.int32)
    residual_image_CbCr = np.round(np.clip(residual_image_CbCr, -255, 255)).astype(np.int32)

    return residual_image_Y, residual_image_CbCr

def reconstruct_image(residual, coefficients):
    H, W, C = residual.shape

    reconstruction = np.zeros_like(residual)
    reconstruction[0, :, :] = residual[0, :, :]
    reconstruction[:, 0, :] = residual[:, 0, :]

    for h in range(1, H):
        for w in range(1, W):
            pre_s1 = coefficients[0] * reconstruction[h, w - 1, :]
            pre_s2 = coefficients[1] * reconstruction[h - 1, w - 1, :]
            pre_s3 = coefficients[2] * reconstruction[h - 1, w, :]

            prediction = pre_s1 + pre_s2 + pre_s3
            reconstruction[h, w, :] = prediction + residual[h, w, :]
    return reconstruction


def inverse_three_pixels_predictor(residual_Y, residual_CbCr, subsample_color_channels=True):
    # Coefficients: same as forward predictor
    coefficients_Y = [7 / 8, -4 / 8, 5 / 8]
    coefficients_CbCr = [3 / 8, -2 / 8, 7 / 8]

    if residual_Y.ndim == 2:
        residual_Y = residual_Y[:, :, np.newaxis]

    recons_Y = reconstruct_image(residual_Y.astype(np.float32), coefficients_Y)

    if subsample_color_channels:
        # Upsample CbCr from (H/2, W/2) to (H, W)
        CbCr_upsampled = zoom(residual_CbCr, (2, 2, 1), order=0)    # Nearest-neighbor
        CbCr_upsampled = CbCr_upsampled[:residual_Y.shape[0], :residual_Y.shape[1], :]  # Trim if needed
    else:
        CbCr_upsampled = residual_CbCr

    recons_CbCr = reconstruct_image(CbCr_upsampled.astype(np.float32), coefficients_CbCr)

    # Combine Y, Cb, Cr
    image_ycbcr = np.zeros((*recons_Y.shape[:2], 3), dtype=np.float32)
    image_ycbcr[..., 0] = recons_Y.squeeze()
    image_ycbcr[..., 1] = recons_CbCr[..., 0]
    image_ycbcr[..., 2] = recons_CbCr[..., 1]

    # Convert to RGB
    image_rgb = ycbcr2rgb(image_ycbcr)
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    return image_rgb

