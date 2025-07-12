import numpy as np
from einops import rearrange
from numpy.lib.stride_tricks import sliding_window_view

def stats_joint(image, pixel_range):
    """
    Computes joint probability of non-overlapping horizontal pixel pairs
    of an image, similar to stats_marg function. However, this
    counts every instance of pixel pairs in a 2D table to
    find the frequencies. Then, it normalizes the values to
    convert them to probabilities. Return a 1D vector
    since this is how we represent pmf values.

    Hint: You can use np.histogram2d for counting quickly over pixel pairs

    image: np.array of shape [H, W, C]
    pixel_range: np.array of shape [B] where B is number of bins, e.g. pixel_range=np.arange(256)

    returns 
        pmf: np.array of shape [B^2], probability mass function of image pixel pairs over range
    """
    # A table to hold count of pixel pair occurences
    count_table = np.zeros((len(pixel_range), len(pixel_range)))

    # Get all non overlapping horizontal pixel pairs as an array of shape [N, 2]
    pixel_pairs = rearrange(image, 'h (w s) c -> (h w c) s', s=2)

    # YOUR CODE STARTS HERE
    # Count the number of occurrences of the pixel pair (i, j).
    for pair in pixel_pairs:
        i, j = pair
        count_table[i, j] += 1

    # Normalize to get joint PMF
    joint_pmf = count_table / count_table.sum()

    # Flatten to 1D
    pmf = joint_pmf.flatten()
    
    # YOUR CODE ENDS HERE
    return pmf

def stats_cond(image, pixel_range, eps=1e-8):
    """
    Computes conditional probability of overlapping horizontal pixel pairs
    of an image, similar to stats_joint function. The conditional probability
    is found by the formula SUM{ - p(x,y) * ( log2( p(x,y) ) - log2( p(x) ) ) }. To compute
    p(x), you can take the sum of normalized probabilities of p(x,y) over row axis.
    Make sure to add a small epsilon before computing the log probabilities. You can
    ignore the first pixels in every row since they don't have a left neighbor.

    Hint: You can use np.histogram2d for counting quickly over pixel pairs

    image: np.array of shape [H, W, C]
    pixel_range: np.array of shape [B] where B is number of bins, e.g. pixel_range=np.arange(256)

    returns 
        cond_entropy: a scalar value
    """
    # Get all overlapping horizontal pixel pairs as an array of shape [N, 2]
    pixel_pairs = rearrange(sliding_window_view(image, 2, axis=1), 'h w c s-> (h w c) s', s=2) 

    # YOUR CODE STARTS HERE
    left_pixels = pixel_pairs[:, 0].flatten()   # S3
    curr_pixels = pixel_pairs[:, 1].flatten()    # S4

    # Compute joint histogram p(x, y)
    joint_counts, _, _ = np.histogram2d(left_pixels, curr_pixels, bins=[pixel_range, pixel_range])
    joint_pmf = joint_counts / joint_counts.sum()

    # Compute marginal p(x) by summing over rows (axis=1)
    marginal_pmf = joint_pmf.sum(axis=1)

    # Compute conditional entropy
    joint_pmf = joint_pmf + eps
    marginal_pmf = marginal_pmf + eps
    cond_entropy = -np.sum(joint_pmf * (np.log2(joint_pmf) - np.log2(marginal_pmf[:, None])))
    
    # YOUR CODE ENDS HERE
    return cond_entropy
