import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import io
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2
from pathlib import Path

class PDFTrademarkExtractor:
    """Extract trademark images from PDF files"""
    
    def __init__(self, dpi: int = 300, min_size: int = 100, max_images: int = 50):
        """
        Initialize PDF extractor
        
        Args:
            dpi: DPI for PDF to image conversion
            min_size: Minimum image dimension (width or height) in pixels
            max_images: Maximum images to extract per PDF
        """
        self.dpi = dpi
        self.min_size = min_size
        self.max_images = max_images
    
    def extract_images_from_pdf(self, pdf_path: Path) -> List[Tuple[Image.Image, Dict]]:
        """
        Extract all trademark images from a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of (image, metadata) tuples
        """
        images = []
        
        # Try method 1: Extract embedded images using PyMuPDF
        embedded_images = self._extract_embedded_images(pdf_path)
        images.extend(embedded_images)
        
        # If no embedded images found, try method 2: Convert pages to images
        if not images:
            page_images = self._extract_from_pages(pdf_path)
            images.extend(page_images)
        
        # Filter and validate images
        valid_images = self._filter_valid_images(images)
        
        return valid_images[:self.max_images]
    
    def _extract_embedded_images(self, pdf_path: Path) -> List[Tuple[Image.Image, Dict]]:
        """Extract images that are embedded in the PDF"""
        images = []
        
        try:
            pdf_document = fitz.open(str(pdf_path))
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    
                    # Extract image
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    metadata = {
                        'source': 'embedded',
                        'page': page_num + 1,
                        'index': img_index,
                        'width': image.width,
                        'height': image.height
                    }
                    
                    images.append((image, metadata))
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error extracting embedded images from {pdf_path}: {e}")
        
        return images
    
    def _extract_from_pages(self, pdf_path: Path) -> List[Tuple[Image.Image, Dict]]:
        """Convert PDF pages to images and extract trademarks"""
        images = []
        
        try:
            # Convert PDF pages to images
            pages = convert_from_path(str(pdf_path), dpi=self.dpi)
            
            for page_num, page_image in enumerate(pages):
                # Try to detect and extract individual logos/trademarks from the page
                extracted = self._extract_logos_from_page(page_image, page_num + 1)
                images.extend(extracted)
                
                # If no logos detected, use the entire page
                if not extracted:
                    metadata = {
                        'source': 'page',
                        'page': page_num + 1,
                        'index': 0,
                        'width': page_image.width,
                        'height': page_image.height
                    }
                    images.append((page_image, metadata))
        
        except Exception as e:
            print(f"Error converting PDF pages from {pdf_path}: {e}")
        
        return images
    
    def _extract_logos_from_page(self, page_image: Image.Image, page_num: int) -> List[Tuple[Image.Image, Dict]]:
        """
        Detect and extract individual logos/trademarks from a page image
        using contour detection
        """
        logos = []
        
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(page_image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and extract regions
            for idx, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if w < self.min_size or h < self.min_size:
                    continue
                
                # Filter by aspect ratio (avoid very thin regions)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio > 10 or aspect_ratio < 0.1:
                    continue
                
                # Extract region
                cropped = page_image.crop((x, y, x + w, y + h))
                
                metadata = {
                    'source': 'detected',
                    'page': page_num,
                    'index': idx,
                    'width': w,
                    'height': h,
                    'x': x,
                    'y': y
                }
                
                logos.append((cropped, metadata))
        
        except Exception as e:
            print(f"Error detecting logos in page {page_num}: {e}")
        
        return logos
    
    def _filter_valid_images(self, images: List[Tuple[Image.Image, Dict]]) -> List[Tuple[Image.Image, Dict]]:
        """Filter out invalid or too small images"""
        valid = []
        
        for image, metadata in images:
            # Check size
            if image.width < self.min_size or image.height < self.min_size:
                continue
            
            # Check if image is mostly white/blank
            if self._is_blank_image(image):
                continue
            
            valid.append((image, metadata))
        
        return valid
    
    def _is_blank_image(self, image: Image.Image, threshold: float = 0.95) -> bool:
        """Check if image is mostly blank/white"""
        # Convert to grayscale
        gray = image.convert('L')
        
        # Calculate percentage of white pixels
        pixels = np.array(gray)
        white_pixels = np.sum(pixels > 240)
        total_pixels = pixels.size
        
        white_percentage = white_pixels / total_pixels
        
        return white_percentage > threshold
    
    def save_extracted_images(self, images: List[Tuple[Image.Image, Dict]], 
                             output_dir: Path, base_name: str) -> List[Path]:
        """
        Save extracted images to directory
        
        Args:
            images: List of (image, metadata) tuples
            output_dir: Output directory
            base_name: Base name for saved files
            
        Returns:
            List of saved file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        for idx, (image, metadata) in enumerate(images):
            filename = f"{base_name}_page{metadata['page']}_img{idx}.png"
            filepath = output_dir / filename
            
            image.save(filepath, 'PNG')
            saved_paths.append(filepath)
        
        return saved_paths