import numpy as np
from einops import rearrange

class ZeroRunCoder:

    def __init__(self, end_of_block=4000, block_size = 64):
        self.EOB = end_of_block
        self.block_size = block_size

    def encode(self, flat_patch_img: np.array):
        """
        This function gets a flattened patched image and produces a list of 
        symbols that applies a zero run encoding of the input where sequential
        blocks of zeroes (e.g. [... 0 0 0 0 0 ...]) are replaced with a marker zero
        and the number of additional zeroes (e.g. [... 0 4 ...]). The original sequence
        is processed in blocks of block_size and every encoding of a block ends with an
        end of block symbol. If all the original values are zero until the end of block,
        then no marker is necessary and we can put an EOB symbol directly.

        flat_patch_img: np.array of shape [H_patch, W_patch, C, Block_size]

        returns:
            encoded: List of symbols that represent the original elements
        
        """
        flat_img = rearrange(flat_patch_img, 'h w c p-> (h w c) p', p=self.block_size)

        # YOUR CODE STARTS HERE
        encoded = []

        for block in flat_img:
            zero_count = 0
            eob_written = False

            for i, value in enumerate(block):
                # The value from index i to the end are all 0
                if np.all(block[i:] == 0):
                    encoded.append(self.EOB)
                    eob_written = True
                    break

                if value == 0:
                    zero_count += 1
                else:
                    while zero_count > 0:
                        run_length = min(zero_count, 255)  # Max run length is 255
                        encoded.extend([0, run_length])
                        zero_count -= run_length
                    encoded.append(value)     # Add the non-zero value

            # Make sure every block ends with EOB
            if not eob_written:
                while zero_count > 0:
                    run_length = min(zero_count, 255)
                    encoded.extend([0, run_length])
                    zero_count -= run_length
                encoded.append(self.EOB)
        # YOUR CODE ENDS HERE

        encoded = np.array(encoded, dtype=np.int32)
        return encoded
    
    def decode(self, encoded, original_shape):
        """
        This function gets an encoding and the original shape to decode the elements 
        of the original array. It acts as the inverse function of the encoder.

        encoded: List of symbols that represent the original elements
        original_shape: List of 3 numbers that represent number of H_patch, W_patch and C

        returns:
            flat_patch_img: np.array of shape [H_patch, W_patch, C, Block_size]
        
        """
        
        # YOUR CODE STARTS HERE
        flat_img = []
        block = []
        i = 0

        while i < len(encoded):
            value = encoded[i]
            if value == self.EOB:
                while len(block) < self.block_size:
                    block.append(0)
                flat_img.append(block)
                block = []  # Set up the next block
                i += 1
            elif value == 0:
                run_length = encoded[i + 1]
                block.extend([0] * run_length)
                i += 2
            else:
                block.append(value)
                i += 1
        flat_img = np.array(flat_img, dtype=np.int32)
        # YOUR CODE ENDS HERE

        flat_patch_img = rearrange(
            flat_img,
            '(h w c) p -> h w c p', 
            h=original_shape[0], w=original_shape[1],
            c=original_shape[2], p=self.block_size)
        return flat_patch_img

            
            