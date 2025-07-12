import unittest
import numpy as np
from ivclab.utils import imread
from ivclab.entropy import stats_marg, calc_entropy, min_code_length 
from ivclab.entropy import stats_joint, stats_cond
from ivclab.image import single_pixel_predictor, three_pixels_predictor

class TestProbability(unittest.TestCase):

    def setUp(self) -> None:
        self.orig_img = imread('../exercises/data/satpic1.bmp')
        self.ref_img = imread('../exercises/data/lena.tif')
        return super().setUp()
    
    def test_correct_entropy(self) -> None:
        pmf = stats_marg(self.orig_img, np.arange(256))
        entropy = calc_entropy(pmf)
        self.assertAlmostEqual(entropy, 6.80779061643218, delta=0.2)

    def test_correct_code_length(self) -> None:
        target_pmf = stats_marg(self.orig_img, np.arange(256))
        common_pmf = stats_marg(self.ref_img, np.arange(256))
        code_length = min_code_length(target_pmf, common_pmf)
        self.assertAlmostEqual(code_length, 7.423096098407454, delta=0.2)

class TestEntropy(unittest.TestCase):

    def setUp(self) -> None:
        self.orig_img = imread('../exercises/data/satpic1.bmp')
        return super().setUp()
    
    def test_correct_joint_entropy(self) -> None:
        joint_pmf = stats_joint(self.orig_img, np.arange(256))
        joint_entropy = calc_entropy(joint_pmf)
        self.assertAlmostEqual(joint_entropy, 12.02494851967153, delta=0.2)

    def test_correct_cond_entropy(self) -> None:
        cond_entropy = stats_cond(self.orig_img, np.arange(256))
        self.assertAlmostEqual(cond_entropy, 5.22159752979922, delta=0.2)

class TestPredictive(unittest.TestCase):

    def setUp(self) -> None:
        self.orig_img = imread('../exercises/data/sail.tif')
        return super().setUp()
    
    def test_correct_single_pixel_predictor_coding(self) -> None:
        residual_image = single_pixel_predictor(self.orig_img)
        pmf = stats_marg(residual_image, np.arange(-255,255))
        entropy = calc_entropy(pmf)
        self.assertAlmostEqual(entropy, 5.67565776280646, delta=0.2)

    def test_correct_three_pixels_predictor_coding(self) -> None:
        residual_image_Y, residual_image_CbCr = three_pixels_predictor(self.orig_img, subsample_color_channels=False)
        merged_residuals = np.concatenate([residual_image_Y.ravel(), residual_image_CbCr.ravel()])
        pmf = stats_marg(merged_residuals, np.arange(-255,255))
        entropy = calc_entropy(pmf)
        self.assertAlmostEqual(entropy, 3.850937452840888, delta=0.2)
        
if __name__ == '__main__':
    unittest.main()