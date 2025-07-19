import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.quantization import PatchQuant
from ivclab.utils import ZigZag, Patcher
from ivclab.signal import DiscreteCosineTransform
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from exercises.ch5 import adaptivequant
import matplotlib.pyplot as plt

class IntraCodec:

    def __init__(self, 
                 quantization_scale=1.0,
                 bounds=(-1000, 4000),
                 end_of_block=4000,
                 block_shape=(8, 8),
                 use_adaptive_quant=False,
                 min_q=0.5,
                 max_q=2
                 ):
        
        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape
        self.symbol_length = 0
        self.use_adaptive_quant = use_adaptive_quant
        self.min_q = min_q
        self.max_q = max_q
        self.q_map = None   # For adaptive quantization

        self.dct = DiscreteCosineTransform()
        self.quant = PatchQuant(quantization_scale=quantization_scale)
        self.zigzag = ZigZag()
        self.zerorun = ZeroRunCoder(end_of_block=end_of_block, block_size=block_shape[0] * block_shape[1])
        self.huffman = HuffmanCoder(lower_bound=bounds[0])
        self.patcher = Patcher()

    def image2symbols(self, img: np.array, is_source_rgb=True):
        """
        Computes the symbol representation of an image by applying rgb2ycbcr,
        DCT, Quantization, ZigZag and ZeroRunEncoding in order.

        img: np.array of shape [H, W, C]

        returns:
            symbols: List of integers
        """
        # YOUR CODE STARTS HERE
        if is_source_rgb:
            img = rgb2ycbcr(img)

        patches = self.patcher.patch(img)
        dct_patches = self.dct.transform(patches)

        if self.use_adaptive_quant:
            img_y = img[..., 0]
            # # Compute gradient strength per block
            # norm_grad_map = adaptivequant.compute_gradient_blockwise(img_y)
            # # Map gradient to local quantization scale
            # self.q_map = self.max_q - (self.max_q - self.min_q) * norm_grad_map

            importance_map = adaptivequant.compute_importance_map(img_y, blk_size=8, alpha=0.6, beta=1.5)
            self.q_map = self.max_q - (self.max_q - self.min_q) * importance_map
            # self.q_map = np.clip(self.q_map, self.min_q, self.max_q)
            quantized = adaptivequant.adaptive_quantize(dct_patches=dct_patches, img_shape=img.shape,
                q_map=self.q_map, quantizer=self.quant)
        else:
            quantized = self.quant.quantize(dct_patches)

        zz_scanned = self.zigzag.flatten(quantized)
        zr_encoded = self.zerorun.encode(zz_scanned)
        symbols = zr_encoded.flatten()
        # YOUR CODE ENDS HERE

        return symbols
    
    def symbols2image(self, symbols, original_shape):
        """
        Reconstructs the original image from the symbol representation
        by applying ZeroRunDecoding, Inverse ZigZag, Dequantization and 
        IDCT, ycbcr2rgb in order. The argument original_shape is required to compute 
        patch_shape, which is needed by ZeroRunDecoding to correctly 
        reshape the input image from blocks.

        symbols: List of integers
        original_shape: List of 3 elements that contains H, W and C
        
        returns:
            reconstructed_img: np.array of shape [H, W, C]
        """
        patch_shape = [original_shape[0] // 8, original_shape[1] // 8, original_shape[2]]

        # YOUR CODE STARTS HERE
        zr_decoded = self.zerorun.decode(symbols, patch_shape)
        zz_inverse = self.zigzag.unflatten(zr_decoded)

        if self.use_adaptive_quant:
            dequantized = adaptivequant.adaptive_dequantize(zz_patches=zz_inverse, img_shape=original_shape,
                                                        q_map=self.q_map, quantizer=self.quant)
        else:
            dequantized = self.quant.dequantize(zz_inverse)

        idct_patches = self.dct.inverse_transform(dequantized)
        reconstructed_img = self.patcher.unpatch(idct_patches)

        # # Convert back to RGB
        #reconstructed_img = ycbcr2rgb(reconstructed_img)
        # YOUR CODE ENDS HERE

        return reconstructed_img
    
    def train_huffman_from_image(self, training_img, is_source_rgb=True):
        """
        Finds the symbols representing the image, extracts the 
        probability distribution of them and trains the huffman coder with it.

        training_img: np.array of shape [H, W, C]

        returns:
            Nothing
        """
        # YOUR CODE STARTS HERE
        symbols = self.image2symbols(training_img, is_source_rgb)
        symbol_range = np.arange(self.bounds[0], self.bounds[1]+2)
        pmf = stats_marg(symbols, symbol_range)    # Calculate probability mass function
        self.huffman.train(pmf)
        # YOUR CODE ENDS HERE

    def intra_encode(self, img: np.array, return_bpp=False, is_source_rgb=True):
        """
        Encodes an image to a bitstream and return it by converting it to
        symbols and compressing them with the Huffman coder.

        img: np.array of shape [H, W, C]

        returns:
            bitstream: List of integers produced by the Huffman coder
        """
        # YOUR CODE STARTS HERE
        symbols = self.image2symbols(img, is_source_rgb)
        self.symbol_length = len(symbols)
        bitstream, bitrate = self.huffman.encode(symbols)

        if return_bpp:
            return bitstream, bitrate
        # YOUR CODE ENDS HERE
        return bitstream

    def intra_decode(self, bitstream, original_shape):
        """
        Decodes an image from a bitstream by decoding it with the Huffman
        coder and reconstructing it from the symbols.

        bitstream: List of integers produced by the Huffman coder
        original_shape: List of 3 values that contain H, W, and C

        returns:
            reconstructed_img: np.array of shape [H, W, C]

        """
        # YOUR CODE STARTS HERE
        symbols = self.huffman.decode(bitstream, self.symbol_length)
        reconstructed_img = self.symbols2image(symbols, original_shape)
        # YOUR CODE ENDS HERE

        return reconstructed_img
    
if __name__ == "__main__":
    from ivclab.utils import imread, calc_psnr
    import matplotlib.pyplot as plt

    lena = imread(f'../data/lena.tif')
    lena_small = imread(f'../data/lena_small.tif')
    intracodec = IntraCodec(quantization_scale=1)
    intracodec.train_huffman_from_image(lena_small)
    symbols, bitsize = intracodec.intra_encode(lena, return_bpp=True)
    reconstructed_img = intracodec.intra_decode(symbols, lena.shape)
    reconstructed_img = ycbcr2rgb(reconstructed_img)
    psnr = calc_psnr(lena, reconstructed_img)
    print(f"PSNR: {psnr:.4f} dB, bpp: {bitsize / (lena.size / 3)}")

    # Chapter 4-1.a
    foreman20 = imread(f'../data/foreman20_40_RGB/foreman0020.bmp')
    intracodec2 = IntraCodec(quantization_scale=1)
    intracodec2.train_huffman_from_image(lena_small)
    symbols2, bitsize2 = intracodec.intra_encode(foreman20, return_bpp=True)
    reconstructed_img2 = intracodec.intra_decode(symbols2, foreman20.shape)
    reconstructed_img2 = ycbcr2rgb(reconstructed_img2)
    psnr2 = calc_psnr(foreman20, reconstructed_img2)
    print(f"PSNR: {psnr2:.4f} dB, bpp: {bitsize2 / (foreman20.size / 3)}")




    