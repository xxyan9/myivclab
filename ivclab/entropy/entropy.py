import numpy as np

def stats_marg(image, pixel_range):
    """
    Computes marginal probability of the pixels of an image
    by counting them with np.histogram and normalizing 
    using the total count. Do not forget to flatten the image
    first. You can pass the range as 'bins' argument to the 
    histogram function. The histogram function returns the counts
    and the bin edges.

    image: np.array of any shape, preferably [H, W, C]
    pixel_range: np.array of shape [B] where B is number of bins, e.g. pixel_range=np.arange(256)

    returns 
        pmf: np.array of shape [B], probability mass function of image pixels over range
    """
    # Convert image to floating point
    image = image * 1.0

    # YOUR CODE STARTS HERE
    counts, _ = np.histogram(image, bins=pixel_range)   # Calculate histogram (count occurrences in each bin)
    pmf = counts / image.size   # Normalize counts to get probabilities (PMF)

    # YOUR CODE ENDS HERE
    return pmf

def calc_entropy(pmf, eps=1e-8):
    """
    Computes entropy for the given probability mass function
    with the formula SUM{ - p(x) * log2(p(x))}.

    pmf: np.array of shape [B] containing the probabilities for bins

    returns 
        entropy: scalar value, computed entropy according to the above formula
    """
    # It's good practice to add small epsilon
    # to get rid of bins with zeroes before taking logarithm
    pmf = pmf + eps

    # YOUR CODE STARTS HERE
    entropy = -np.sum(pmf * np.log2(pmf))

    # YOUR CODE ENDS HERE
    return entropy

def min_code_length(target_pmf, common_pmf, eps=1e-8):
    """
    Computes minimum average codeword length for the
    target pmf given the common pmf using the formula
    formula SUM{ - p(x) * log2(q(x))} where p(x) is the
    target probability and q(x) comes from the common pmf

    target_pmf: np.array of shape [B] containing the probabilities for bins
    common_pmf: np.array of shape [B] containing the probabilities for bins

    returns 
        code_length: scalar value, computed entropy according to the above formula
    """
    # It's good practice to add small epsilon
    # to get rid of bins with zeroes before taking logarithm
    common_pmf = common_pmf + eps
    
    # YOUR CODE STARTS HERE
    target_pmf = np.asarray(target_pmf)
    common_pmf = np.asarray(common_pmf)

    code_length = -np.sum(target_pmf * np.log2(common_pmf))

    # YOUR CODE ENDS HERE
    return code_length