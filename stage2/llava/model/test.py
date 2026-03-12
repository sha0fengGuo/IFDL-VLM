import sys
import os
import torch
import numpy as np
from PIL import Image
import pytest
import tempfile
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

import unittest.mock as mock

# Add the LLaVA path for absolute imports
sys.path.insert(0, '/data/guoshaofeng/LLaVA')


class TestCLIPVisionTowerApplyMask:
    """Test suite for the _apply_mask_to_image method"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Mock args for CLIPVisionTower initialization
        self.mock_args = mock.MagicMock()
        self.mock_args.mm_vision_select_layer = -2
        self.mock_args.mm_vision_select_feature = 'patch'
        self.mock_args.enable_region_aware = True
        self.mock_args.region_weight = 0.5
        self.mock_args.unfreeze_mm_vision_tower = False
        
        # Create CLIPVisionTower instance with delay_load=True to avoid loading actual model
        self.vision_tower = CLIPVisionTower(
            vision_tower="openai/clip-vit-large-patch14-336",
            args=self.mock_args,
            delay_load=True
        )
        
        # Mock the vision_tower properties to avoid actual model loading
        self.vision_tower.is_loaded = True
        mock_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vision_tower._device = mock_device
        
    def create_test_image(self, height=336, width=336, dtype=torch.float32, device='cpu'):
        """Helper to create test image tensor"""
        # Create a random image tensor [3, H, W]
        image = torch.randn(3, height, width, dtype=dtype, device=torch.device(device))
        # Normalize to typical image range
        image = torch.clamp(image, -2.0, 2.0)
        return image
        
    def create_test_mask_tensor(self, height=336, width=336, dtype=torch.float32, device='cpu'):
        """Helper to create test mask as tensor"""
        # Create a mask with a rectangular region in the center
        mask = torch.zeros(height, width, dtype=dtype, device=torch.device(device))
        h_start, h_end = height//4, 3*height//4
        w_start, w_end = width//4, 3*width//4
        mask[h_start:h_end, w_start:w_end] = 1.0
        return mask
        
    def create_test_mask_numpy(self, height=336, width=336):
        """Helper to create test mask as numpy array"""
        mask = np.zeros((height, width), dtype=np.uint8)
        h_start, h_end = height//4, 3*height//4
        w_start, w_end = width//4, 3*width//4
        mask[h_start:h_end, w_start:w_end] = 255
        return mask
        
    def create_test_mask_pil(self, height=336, width=336):
        """Helper to create test mask as PIL Image"""
        mask_array = self.create_test_mask_numpy(height, width)
        return Image.fromarray(mask_array, mode='L')

    def test_basic_mask_application(self):
        """Test basic mask application with tensor inputs"""
        image = self.create_test_image()
        mask = self.create_test_mask_tensor()
        
        # Mock device property
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result = self.vision_tower._apply_mask_to_image(image, mask)
        
        # Verify output shape
        assert result.shape == image.shape, f"Expected shape {image.shape}, got {result.shape}"
        
        # Verify masked regions are zero where mask is 0
        mask_3d = mask.unsqueeze(0).expand_as(image)
        expected = image * mask_3d
        torch.testing.assert_close(result, expected)
        
    def test_data_type_consistency_float32(self):
        """Test data type consistency with float32"""
        image = self.create_test_image(dtype=torch.float32)
        mask = self.create_test_mask_tensor(dtype=torch.float32)
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result = self.vision_tower._apply_mask_to_image(image, mask)
            
        assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"
        
    def test_data_type_consistency_float16(self):
        """Test data type consistency with float16"""
        image = self.create_test_image(dtype=torch.float16)
        mask = self.create_test_mask_tensor(dtype=torch.float16)
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result = self.vision_tower._apply_mask_to_image(image, mask)
            
        assert result.dtype == torch.float16, f"Expected float16, got {result.dtype}"
        
    def test_data_type_consistency_bfloat16(self):
        """Test data type consistency with bfloat16 (key for training error)"""
        if not torch.cuda.is_available():
            pytest.skip("bfloat16 requires CUDA")
            
        device = torch.device('cuda')
        image = self.create_test_image(dtype=torch.bfloat16, device='cuda')
        mask = self.create_test_mask_tensor(dtype=torch.bfloat16, device='cuda')
        
        with mock.patch.object(self.vision_tower, 'device', device):
            result = self.vision_tower._apply_mask_to_image(image, mask)
            
        assert result.dtype == torch.bfloat16, f"Expected bfloat16, got {result.dtype}"
        
    def test_mask_type_conversion_numpy(self):
        """Test mask conversion from numpy array"""
        image = self.create_test_image()
        mask_np = self.create_test_mask_numpy()
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result = self.vision_tower._apply_mask_to_image(image, mask_np)
            
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        
    def test_mask_type_conversion_pil(self):
        """Test mask conversion from PIL Image"""
        image = self.create_test_image()
        mask_pil = self.create_test_mask_pil()
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result = self.vision_tower._apply_mask_to_image(image, mask_pil)
            
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        
    def test_mask_dimension_handling(self):
        """Test different mask dimensions"""
        image = self.create_test_image()
        
        # Test 2D mask [H, W]
        mask_2d = self.create_test_mask_tensor()
        
        # Test 3D mask [1, H, W]
        mask_3d = mask_2d.unsqueeze(0)
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result_2d = self.vision_tower._apply_mask_to_image(image, mask_2d)
            result_3d = self.vision_tower._apply_mask_to_image(image, mask_3d)
            
        # Both should produce same result
        torch.testing.assert_close(result_2d, result_3d)
        
    def test_mask_resizing(self):
        """Test mask resizing when dimensions don't match"""
        image = self.create_test_image(height=336, width=336)
        mask = self.create_test_mask_tensor(height=256, width=256)  # Different size
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result = self.vision_tower._apply_mask_to_image(image, mask)
            
        assert result.shape == image.shape
        
    def test_mask_thresholding_normalized(self):
        """Test mask thresholding for normalized values (0-1)"""
        image = self.create_test_image()
        # Create mask with values in 0-1 range
        mask = torch.rand(336, 336) * 0.8 + 0.1  # Values between 0.1 and 0.9
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result = self.vision_tower._apply_mask_to_image(image, mask)
            
        assert result.shape == image.shape
        
    def test_mask_thresholding_uint8(self):
        """Test mask thresholding for uint8 values (0-255)"""
        image = self.create_test_image()
        # Create mask with values in 0-255 range
        mask = (torch.rand(336, 336) * 200 + 28).to(torch.uint8)  # Values between 28 and 228
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result = self.vision_tower._apply_mask_to_image(image, mask)
            
        assert result.shape == image.shape
        
    def test_invalid_mask_dimensions(self):
        """Test error handling for invalid mask dimensions"""
        image = self.create_test_image()
        # Create invalid 4D mask
        invalid_mask = torch.rand(2, 2, 336, 336)
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            with pytest.raises(ValueError, match="Unexpected mask dimensions"):
                self.vision_tower._apply_mask_to_image(image, invalid_mask)
                
    def test_device_consistency_cuda(self):
        """Test device consistency when using CUDA"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = torch.device('cuda')
        image = self.create_test_image(device='cuda')
        mask = self.create_test_mask_tensor(device='cpu')  # Different device
        
        with mock.patch.object(self.vision_tower, 'device', device):
            result = self.vision_tower._apply_mask_to_image(image, mask)
            
        assert result.device == image.device
        
    def test_mask_statistics_output(self, capsys):
        """Test that mask statistics are printed correctly"""
        image = self.create_test_image()
        mask = self.create_test_mask_tensor()
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            self.vision_tower._apply_mask_to_image(image, mask)
            
        captured = capsys.readouterr()
        assert "mask统计" in captured.out
        assert "masked_image统计" in captured.out
        
    def test_zero_mask(self):
        """Test behavior with all-zero mask"""
        image = self.create_test_image()
        mask = torch.zeros(336, 336)
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result = self.vision_tower._apply_mask_to_image(image, mask)
            
        # Result should be all zeros
        expected = torch.zeros_like(image)
        torch.testing.assert_close(result, expected)
        
    def test_full_mask(self):
        """Test behavior with all-ones mask"""
        image = self.create_test_image()
        mask = torch.ones(336, 336)
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result = self.vision_tower._apply_mask_to_image(image, mask)
            
        # Result should equal original image
        torch.testing.assert_close(result, image)
        
    def test_mixed_dtype_handling(self):
        """Test handling when image and mask have different dtypes"""
        image = self.create_test_image(dtype=torch.float32)
        mask = self.create_test_mask_tensor(dtype=torch.float16)
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result = self.vision_tower._apply_mask_to_image(image, mask)
            
        # Result should maintain image dtype
        assert result.dtype == image.dtype
        
    def test_batch_consistency(self):
        """Test that multiple calls produce consistent results"""
        image = self.create_test_image()
        mask = self.create_test_mask_tensor()
        
        with mock.patch.object(self.vision_tower, 'device', torch.device('cpu')):
            result1 = self.vision_tower._apply_mask_to_image(image, mask)
            result2 = self.vision_tower._apply_mask_to_image(image, mask)
            
        torch.testing.assert_close(result1, result2)

if __name__ == "__main__":
    pytest.main([__file__])