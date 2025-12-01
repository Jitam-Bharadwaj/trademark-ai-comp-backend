import pytest
from PIL import Image
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.image_processor import ImagePreprocessor

class TestImagePreprocessor:
    """Test image preprocessing"""
    
    @pytest.fixture
    def preprocessor(self):
        return ImagePreprocessor(target_size=(224, 224))
    
    @pytest.fixture
    def sample_image(self):
        # Create a simple test image
        return Image.new('RGB', (100, 100), color='red')
    
    def test_load_image(self, preprocessor, tmp_path):
        """Test image loading"""
        # Create temp image
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(img_path)
        
        # Load
        loaded = preprocessor.load_image(img_path)
        assert loaded is not None
        assert loaded.mode == 'RGB'
    
    def test_resize_with_padding(self, preprocessor, sample_image):
        """Test resizing with padding"""
        resized = preprocessor.resize_with_padding(sample_image)
        assert resized.size == (224, 224)
    
    def test_preprocess_pil(self, preprocessor, sample_image):
        """Test full preprocessing pipeline"""
        tensor = preprocessor.preprocess_pil(sample_image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)