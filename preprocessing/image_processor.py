import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional
import torch
from torchvision import transforms
from pathlib import Path

class ImagePreprocessor:
    """Handles all image preprocessing operations"""
    
    def __init__(self, target_size: tuple = (224, 224), remove_bg: bool = False):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (width, height)
            remove_bg: Whether to remove background
        """
        self.target_size = target_size
        self.remove_bg = remove_bg
        
        # CLIP preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        if remove_bg:
            try:
                from rembg import remove
                self.bg_remover = remove
            except ImportError:
                self.remove_bg = False
                print("Warning: rembg not installed, background removal disabled")
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[Image.Image]:
        """
        Load image from path
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image or None if failed
        """
        try:
            img = Image.open(image_path)
            return img.convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background from image"""
        if not self.remove_bg:
            return image
        
        try:
            # Convert PIL to bytes
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Remove background
            output = self.bg_remover(img_byte_arr)
            
            # Convert back to PIL
            return Image.open(io.BytesIO(output)).convert('RGB')
        except Exception as e:
            print(f"Error removing background: {e}")
            return image
    
    def resize_with_padding(self, image: Image.Image) -> Image.Image:
        """Resize image while maintaining aspect ratio using padding"""
        # Get current size
        width, height = image.size
        target_width, target_height = self.target_size
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create new image with padding
        new_image = Image.new('RGB', self.target_size, (255, 255, 255))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancement (contrast, sharpness)"""
        from PIL import ImageEnhance
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance sharpness slightly
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def preprocess(self, image_path: Union[str, Path]) -> Optional[torch.Tensor]:
        """
        Complete preprocessing pipeline
        
        Args:
            image_path: Path to image
            
        Returns:
            Preprocessed tensor or None if failed
        """
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Remove background
        if self.remove_bg:
            image = self.remove_background(image)
        
        # Resize with padding
        image = self.resize_with_padding(image)
        
        # Enhance
        image = self.enhance_image(image)
        
        # Convert to tensor
        tensor = self.transform(image)
        
        return tensor
    
    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image directly"""
        # Convert to RGB
        image = image.convert('RGB')
        
        # Remove background
        if self.remove_bg:
            image = self.remove_background(image)
        
        # Resize with padding
        image = self.resize_with_padding(image)
        
        # Enhance
        image = self.enhance_image(image)
        
        # Convert to tensor
        tensor = self.transform(image)
        
        return tensor