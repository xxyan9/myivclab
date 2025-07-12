import unittest
import numpy as np
from ivclab.utils import imread
from ivclab.utils import Patcher
from ivclab.signal import DiscreteCosineTransform
from ivclab.quantization import PatchQuant
from ivclab.utils import calc_psnr
from ivclab.utils.metrics import calc_mse

class TestDiscreteCosineTransform(unittest.TestCase):

    def setUp(self) -> None:
        self.orig_img = imread('../data/satpic1.bmp')
        self.patcher = Patcher(window_size = (8,8))
        self.dct = DiscreteCosineTransform(norm = 'ortho')
        return super().setUp()
    
    def test_forward_transform(self) -> None:
        patched_img = self.patcher.patch(self.orig_img)
        transformed = self.dct.transform(patched_img)
        self.assertAlmostEqual(np.mean(transformed**2), 10616, delta=100)

    def test_inverse_transform(self) -> None:
        patched_img = self.patcher.patch(self.orig_img)
        transformed = self.dct.transform(patched_img)
        reconstructed_patched = self.dct.inverse_transform(transformed)
        self.assertTrue(np.allclose(reconstructed_patched, patched_img))

class TestPatchQuant(unittest.TestCase):

    def setUp(self) -> None:
        self.orig_img = imread('../data/satpic1.bmp')
        self.patcher = Patcher(window_size = (8,8))
        self.quantizer = PatchQuant(quantization_scale=1.0)
        return super().setUp()
    
    def test_quantization(self) -> None:
        patched_img = self.patcher.patch(self.orig_img)
        quantized = self.quantizer.quantize(patched_img)
        self.assertAlmostEqual(np.mean(quantized**2), 7.5409901936848955, delta=0.1)

    def test_dequantization(self) -> None:
        patched_img = self.patcher.patch(self.orig_img)
        quantized = self.quantizer.quantize(patched_img)
        dequantized = self.quantizer.dequantize(quantized)
        reconstructed = self.patcher.unpatch(dequantized)
        self.assertAlmostEqual(calc_mse(self.orig_img, reconstructed), 348.2207400004069, delta=5)
    
if __name__ == '__main__':
    unittest.main()