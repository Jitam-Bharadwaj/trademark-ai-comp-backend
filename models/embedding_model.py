import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union
import numpy as np
from PIL import Image

class EmbeddingGenerator:
    """Generate embeddings using CLIP model"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the pretrained model
            device: Device to run model on (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.projection_dim
    
    @torch.no_grad()
    def generate_embedding(self, image: Union[Image.Image, torch.Tensor]) -> np.ndarray:
        """
        Generate embedding for single image
        
        Args:
            image: PIL Image or preprocessed tensor
            
        Returns:
            Normalized embedding vector
        """
        # Process image
        if isinstance(image, Image.Image):
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # Assume it's already a tensor
            inputs = {"pixel_values": image.unsqueeze(0).to(self.device)}
        
        # Generate embedding
        outputs = self.model.get_image_features(**inputs)
        
        # Normalize
        embedding = outputs / outputs.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        return embedding.cpu().numpy()[0]
    
    @torch.no_grad()
    def generate_embeddings_batch(self, images: List[Union[Image.Image, torch.Tensor]]) -> np.ndarray:
        """
        Generate embeddings for batch of images
        
        Args:
            images: List of PIL Images or tensors
            
        Returns:
            Array of normalized embeddings
        """
        # Process images
        if isinstance(images[0], Image.Image):
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # Stack tensors
            inputs = {"pixel_values": torch.stack(images).to(self.device)}
        
        # Generate embeddings
        outputs = self.model.get_image_features(**inputs)
        
        # Normalize
        embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        return embeddings.cpu().numpy()
    
    @torch.no_grad()
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using CLIP text encoder
        
        Args:
            text: Text string to embed
            
        Returns:
            Normalized embedding vector
        """
        # Process text
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        outputs = self.model.get_text_features(**inputs)
        
        # Normalize
        embedding = outputs / outputs.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        return embedding.cpu().numpy()[0]
    
    @torch.no_grad()
    def generate_text_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for batch of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of normalized embeddings
        """
        # Process texts
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        outputs = self.model.get_text_features(**inputs)
        
        # Normalize
        embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        return embeddings.cpu().numpy()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dim
