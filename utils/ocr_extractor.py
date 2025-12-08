"""
OCR Text Extraction Module
Extracts text from trademark images using OCR
"""
import logging
from typing import Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Try to import OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available. OCR will be disabled.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not available. OCR will be disabled.")


class OCRTextExtractor:
    """Extract text from images using OCR"""
    
    def __init__(self, ocr_method: str = "tesseract", language: str = "eng"):
        """
        Initialize OCR extractor
        
        Args:
            ocr_method: OCR method to use ('tesseract' or 'easyocr')
            language: Language code for OCR (e.g., 'eng', 'eng+fra')
        """
        self.ocr_method = ocr_method.lower()
        self.language = language
        self.easyocr_reader = None
        
        if self.ocr_method == "tesseract":
            if not TESSERACT_AVAILABLE:
                raise ImportError("pytesseract is required for Tesseract OCR. Install with: pip install pytesseract")
        elif self.ocr_method == "easyocr":
            if not EASYOCR_AVAILABLE:
                raise ImportError("easyocr is required for EasyOCR. Install with: pip install easyocr")
            # Initialize EasyOCR reader (lazy loading - done on first use)
            self._init_easyocr()
        else:
            raise ValueError(f"Unsupported OCR method: {ocr_method}. Use 'tesseract' or 'easyocr'")
    
    def _init_easyocr(self):
        """Initialize EasyOCR reader (lazy loading)"""
        if self.easyocr_reader is None and EASYOCR_AVAILABLE:
            try:
                logger.info("Initializing EasyOCR reader (this may take a moment on first use)...")
                self.easyocr_reader = easyocr.Reader([self.language], gpu=False)
                logger.info("EasyOCR reader initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                raise
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        return image
    
    def extract_text_tesseract(self, image: Image.Image) -> str:
        """
        Extract text using Tesseract OCR
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text string
        """
        if not TESSERACT_AVAILABLE:
            return ""
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Extract text
            text = pytesseract.image_to_string(
                processed_image,
                lang=self.language,
                config='--psm 6'  # Assume uniform block of text
            )
            
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return ""
    
    def extract_text_easyocr(self, image: Image.Image) -> str:
        """
        Extract text using EasyOCR
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text string
        """
        if not EASYOCR_AVAILABLE:
            return ""
        
        try:
            # Initialize if not already done
            self._init_easyocr()
            
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # If grayscale, convert to RGB
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Extract text
            results = self.easyocr_reader.readtext(img_array)
            
            # Combine all detected text
            text_parts = [result[1] for result in results]
            text = " ".join(text_parts)
            
            return text.strip()
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return ""
    
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from image using configured OCR method
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted and normalized text string
        """
        if self.ocr_method == "tesseract":
            text = self.extract_text_tesseract(image)
        elif self.ocr_method == "easyocr":
            text = self.extract_text_easyocr(image)
        else:
            text = ""
        
        # Normalize text
        normalized_text = self.normalize_text(text)
        
        return normalized_text
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize extracted text for comparison
        
        Args:
            text: Raw extracted text
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters (keep alphanumeric and spaces)
        import re
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Remove multiple spaces
        text = " ".join(text.split())
        
        return text.strip()
    
    def is_available(self) -> bool:
        """Check if OCR is available"""
        if self.ocr_method == "tesseract":
            return TESSERACT_AVAILABLE
        elif self.ocr_method == "easyocr":
            return EASYOCR_AVAILABLE
        return False







