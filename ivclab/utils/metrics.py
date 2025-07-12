import numpy as np

def calc_mse(orig: np.array, rec: np.array):
    """
    Computes the Mean Squared Error by taking the square of
    the difference between orig and rec, and averaging it
    over all the pixels.

    orig: np.array of shape [H, W, C]
    rec: np.array of shape [H, W, C]

    returns 
        mse: a scalar value
    """
    # YOUR CODE STARTS HERE

    orig = orig.astype(np.double)
    rec = rec.astype(np.double)
    H, W, C = np.shape(orig)

    # Compute squared difference and average over all pixels
    mse = np.sum(((orig - rec) ** 2)/(W*H*C))

    # YOUR CODE ENDS HERE
    return mse

def calc_psnr(orig: np.array, rec:np.array, maxval=255):
    """
    Computes the Peak Signal Noise Ratio by computing
    the MSE and using it in the formula from the lectures.

    > **_ Warning _**: Assumes the signals are in the 
    range [0, 255] by default

    orig: np.array of shape [H, W, C]
    rec: np.array of shape [H, W, C]

    returns 
        psnr: a scalar value
    """
    # YOUR CODE STARTS HERE

    # Compute MSE first
    mse = calc_mse(orig, rec)

    # Avoid division by zero (if MSE is 0, PSNR is infinite)
    if mse == 0:
        return float('inf')

    # Compute PSNR using the formula: 10 * log10(maxval^2 / MSE)
    psnr = 10 * np.log10((maxval ** 2) / mse)

    # YOUR CODE ENDS HERE
    return psnr