import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import io
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2
from pathlib import Path
import re

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
        
        # Extract text metadata from PDF pages first
        text_metadata = self.extract_text_metadata(pdf_path)
        
        # Try method 1: Extract embedded images using PyMuPDF
        embedded_images = self._extract_embedded_images(pdf_path)
        images.extend(embedded_images)
        
        # If no embedded images found, try method 2: Convert pages to images
        # This can catch small logos that might not be properly embedded
        if not images:
            page_images = self._extract_from_pages(pdf_path)
            images.extend(page_images)
        # Also try page conversion if we found very few embedded images (might have missed small ones)
        elif len(images) < 2:
            page_images = self._extract_from_pages(pdf_path)
            # Add page images (filtering will handle duplicates and invalid images)
            images.extend(page_images)
        
        # Filter and validate images
        valid_images = self._filter_valid_images(images)
        
        # Add text metadata to each image based on page number
        for image, metadata in valid_images:
            page_num = metadata.get('page', 1)
            # Get metadata for this page, or use first page's metadata if not found
            page_metadata = text_metadata.get(page_num, text_metadata.get(1, {}))
            metadata.update(page_metadata)
        
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
                
                # Filter by size (but allow smaller images if they have content)
                min_dim = min(w, h)
                max_dim = max(w, h)
                
                # Absolute minimum - reject tiny regions
                if min_dim < 20:
                    continue
                
                # For small regions (20-50px), check if they have meaningful content
                # We'll check this after extraction, so allow them through for now
                if min_dim < self.min_size:
                    # Still allow through - will be filtered later if blank
                    pass
                
                # Filter by aspect ratio (avoid very thin regions)
                aspect_ratio = max_dim / min_dim if min_dim > 0 else 0
                if aspect_ratio > 15 or aspect_ratio < 0.067:  # More lenient for small logos
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
            # Check minimum size (but allow smaller images if they have meaningful content)
            min_dimension = min(image.width, image.height)
            max_dimension = max(image.width, image.height)
            
            # Reject if too small (absolute minimum)
            if min_dimension < 20:  # Absolute minimum - reject tiny images
                continue
            
            # For images between 20-50px, check if they have meaningful content
            if min_dimension < self.min_size:
                # Check if image has enough non-white content to be a valid logo
                if not self._has_meaningful_content(image, min_content_ratio=0.15):
                    continue
            
            # Check if image is mostly white/blank (but be more lenient for small images)
            blank_threshold = 0.98 if min_dimension < 50 else 0.95
            if self._is_blank_image(image, threshold=blank_threshold):
                continue
            
            # Reject extremely elongated images (likely not logos)
            aspect_ratio = max_dimension / min_dimension if min_dimension > 0 else 0
            if aspect_ratio > 15:  # Very long/thin images are likely not logos
                continue
            
            valid.append((image, metadata))
        
        return valid
    
    def _has_meaningful_content(self, image: Image.Image, min_content_ratio: float = 0.15) -> bool:
        """
        Check if image has meaningful non-white content
        
        Args:
            image: PIL Image
            min_content_ratio: Minimum ratio of non-white pixels (0.0-1.0)
            
        Returns:
            True if image has enough meaningful content
        """
        # Convert to grayscale
        gray = image.convert('L')
        pixels = np.array(gray)
        
        # Count non-white pixels (pixels darker than 240)
        non_white_pixels = np.sum(pixels <= 240)
        total_pixels = pixels.size
        
        content_ratio = non_white_pixels / total_pixels if total_pixels > 0 else 0
        
        return content_ratio >= min_content_ratio
    
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
    
    def extract_text_metadata(self, pdf_path: Path) -> Dict[int, Dict]:
        """
        Extract text from PDF pages and parse metadata (applicant_name, trademark_class)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary mapping page numbers to extracted metadata
        """
        page_metadata = {}
        
        try:
            pdf_document = fitz.open(str(pdf_path))
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # Extract text from page
                text = page.get_text()
                
                # Parse metadata from text
                metadata = self._parse_trademark_metadata(text)
                page_metadata[page_num + 1] = metadata
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error extracting text metadata from {pdf_path}: {e}")
        
        return page_metadata
    
    def extract_text_only_trademarks(self, pdf_path: Path) -> List[Dict]:
        """
        Extract text-only trademarks from PDF (trademarks without images)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of trademark metadata dictionaries (one per trademark entry)
        """
        text_trademarks = []
        
        try:
            pdf_document = fitz.open(str(pdf_path))
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text()
                
                # Parse all trademark entries from this page
                trademarks = self._parse_trademark_entries(text, page_num + 1)
                text_trademarks.extend(trademarks)
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error extracting text-only trademarks from {pdf_path}: {e}")
        
        return text_trademarks
    
    def _parse_trademark_entries(self, text: str, page_num: int) -> List[Dict]:
        """
        Parse multiple trademark entries from a page's text
        Each entry should have: mark, application_no, applicant_name, trademark_class
        
        Args:
            text: Extracted text from PDF page
            page_num: Page number
            
        Returns:
            List of trademark metadata dictionaries
        """
        trademarks = []
        
        if not text:
            return trademarks
        
        # Split text into potential trademark entries
        # Each entry typically starts with application number or mark name
        # Pattern: Look for application number patterns to identify entry boundaries
        
        # First, find all application numbers in the text
        app_date_pattern = r'(\d{6,8})\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        app_matches = list(re.finditer(app_date_pattern, text))
        
        if not app_matches:
            # If no application numbers found, try to parse as single entry
            metadata = self._parse_trademark_metadata(text)
            if metadata.get('application_no') or metadata.get('applicant_name'):
                metadata['page'] = page_num
                metadata['mark'] = self._extract_mark_name(text)
                metadata['source'] = 'text_only'
                trademarks.append(metadata)
            return trademarks
        
        # Process each entry (between application numbers or from start to end)
        for idx, app_match in enumerate(app_matches):
            # Determine the text block for this entry
            if idx == 0:
                # First entry: from start of text to next application number (or end)
                start_pos = 0
            else:
                # Start from previous application number
                start_pos = app_matches[idx - 1].start()
            
            if idx < len(app_matches) - 1:
                # End at next application number
                end_pos = app_matches[idx + 1].start()
            else:
                # Last entry: to end of text
                end_pos = len(text)
            
            entry_text = text[start_pos:end_pos]
            
            # Parse metadata for this entry
            metadata = self._parse_trademark_metadata(entry_text)
            mark_name = self._extract_mark_name(entry_text)
            
            # Only add if we have at least application_no or applicant_name
            if metadata.get('application_no') or metadata.get('applicant_name') or mark_name:
                metadata['page'] = page_num
                metadata['mark'] = mark_name
                metadata['source'] = 'text_only'
                trademarks.append(metadata)
        
        return trademarks
    
    def _extract_mark_name(self, text: str) -> str:
        """
        Extract trademark mark/name from text
        The mark is typically the prominent text before the application number
        
        Args:
            text: Text block containing trademark information
            
        Returns:
            Mark name string
        """
        if not text:
            return ''
        
        # Pattern 1: Mark name is typically before application number, often in bold/larger text
        # Look for text that appears before application number, excluding journal headers
        
        # Remove common headers
        text_clean = re.sub(r'Trade\s+Marks\s+Journal\s+No[.:]?\s*\d+.*?\n', '', text, flags=re.IGNORECASE)
        text_clean = re.sub(r'Class\s+\d+', '', text_clean, flags=re.IGNORECASE)
        
        # Find application number position
        app_date_pattern = r'(\d{6,8})\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        app_match = re.search(app_date_pattern, text_clean)
        
        if app_match:
            # Extract text before application number
            before_app = text_clean[:app_match.start()].strip()
            
            # Split by newlines and get the last substantial line before application number
            lines = [line.strip() for line in before_app.split('\n') if line.strip()]
            
            # Filter out common false positives
            false_positives = {
                'TRADE MARKS', 'TRADEMARK', 'JOURNAL', 'PROPOSED', 'USED',
                'TRADE MARKS JOURNAL', 'TRADEMARK JOURNAL', 'CLASS'
            }
            
            # Get the last non-empty line that's not a false positive
            for line in reversed(lines):
                line_upper = line.upper()
                if (line_upper not in false_positives and 
                    len(line) >= 2 and 
                    not re.match(r'^\d+$', line) and  # Not just numbers
                    not re.match(r'^\d+[/-]\d+[/-]\d+$', line)):  # Not a date
                    # Clean up the mark name
                    mark = line.strip()
                    # Remove extra whitespace
                    mark = ' '.join(mark.split())
                    if 2 <= len(mark) <= 200:  # Reasonable length for a mark
                        return mark
        
        # Pattern 2: If no application number found, look for prominent text (all caps, longer lines)
        lines = [line.strip() for line in text_clean.split('\n') if line.strip()]
        for line in lines:
            # Look for lines that are mostly uppercase and substantial length
            if (len(line) >= 5 and len(line) <= 200 and
                sum(1 for c in line if c.isupper()) / max(len(line), 1) > 0.5):
                line_upper = line.upper()
                if line_upper not in false_positives:
                    return ' '.join(line.split())
        
        return ''
    
    def _parse_trademark_metadata(self, text: str) -> Dict:
        """
        Parse applicant_name, trademark_class, and application_no from extracted text
        
        Args:
            text: Extracted text from PDF page
            
        Returns:
            Dictionary with applicant_name, trademark_class, and application_no
        """
        metadata = {
            'applicant_name': '',
            'trademark_class': '',
            'application_no': ''
        }
        
        if not text:
            return metadata
        
        # Extract trademark class (e.g., "Class 31", "Class 25")
        # Look for patterns like "Class 31", "Class: 31", "CLASS 31"
        class_patterns = [
            r'Class\s*:?\s*(\d+)',
            r'CLASS\s*:?\s*(\d+)',
            r'class\s*:?\s*(\d+)',
        ]
        
        for pattern in class_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['trademark_class'] = match.group(1)
                break
        
        # Extract application number
        # Pattern: 6-8 digits (application number) followed by date
        # Example: "5970384 07/06/2023" -> application_no = "5970384"
        app_date_patterns = [
            r'(\d{6,8})\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # Standard format
            r'Application\s+No[.:]?\s*(\d{6,8})',  # "Application No: 5970384"
            r'APPLICATION\s+NO[.:]?\s*(\d{6,8})',  # "APPLICATION NO: 5970384"
        ]
        
        for pattern in app_date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['application_no'] = match.group(1)
                break
        
        # Extract applicant name
        # Pattern: Application number + date, then applicant name on the first line after
        # Names can be long: "SOUL SOCIETIE FOR ORGANIC FARMING RESEARCH & EDUCATION PRIVATE LIMITED"
        # Names can contain: letters, spaces, &, ., ,, etc.
        
        # Common false positives to exclude
        false_positives = {
            'TRADE MARKS', 'TRADEMARK', 'JOURNAL', 'PROPOSED', 'USED', 
            'TRADE MARKS JOURNAL', 'TRADEMARK JOURNAL', 'CLASS', 'PROPRIETOR',
            'ADDRESS', 'INDIA', 'AGENTS', 'SERVICE', 'APPLICANT', 'AHMEDABAD'
        }
        
        # Pattern 1: Find application number and date, then extract the entire first line after
        # Application number pattern: 6-8 digits followed by date (DD/MM/YYYY or DD-MM-YYYY)
        app_date_pattern = r'(\d{6,8})\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        app_match = re.search(app_date_pattern, text)
        
        # If we found application number here and haven't stored it yet, store it
        if app_match and not metadata['application_no']:
            metadata['application_no'] = app_match.group(1)
        
        if app_match:
            # Find the position after the application number and date
            app_end_pos = app_match.end()
            # Extract text after application number/date
            remaining_text = text[app_end_pos:].strip()
            
            # Split by newlines to get the next line (where the name should be)
            lines = [line.strip() for line in remaining_text.split('\n') if line.strip()]
            
            # Also try splitting by multiple spaces (in case newlines aren't preserved)
            if not lines:
                parts = re.split(r'\s{2,}|\t+', remaining_text)
                lines = [p.strip() for p in parts if p.strip()]
            
            # Process the first line/block after application number/date
            if lines:
                first_line = lines[0]
                
                # Extract the name - can be long, may contain &, ., etc.
                # Stop when we encounter clear address indicators:
                # - Line starting with a number (address number)
                # - PROPRIETOR keyword
                # - Common address keywords
                # - Comma followed by number or address keyword
                
                # Pattern: Extract everything until we hit address indicators
                # Allow: uppercase letters, spaces, &, ., ,, -, /, numbers (for company names like "PRIVATE LIMITED")
                # Stop at: PROPRIETOR, Address keywords, or number at start of next segment
                name_pattern = r'^([A-Z][A-Z0-9\s&.,/-]{1,200}?)(?:\s+(?:PROPRIETOR|Address|RAJGARH|KAROTH|ALWAR|RAJASTHAN|PUNJAB|GUJARAT|MAHARASHTRA|AHMEDABAD|KARAN|KHANNA|SOHAL|COMPLEX|SEHDEV|MARKET|JALANDHAR|1ST|FLOOR|,\s*\d|\d{1,3}\s*,\s*[A-Z]))'
                name_match = re.match(name_pattern, first_line, re.IGNORECASE)
                
                if not name_match:
                    # Simpler pattern: extract until PROPRIETOR or number at start
                    name_match = re.match(r'^([A-Z][A-Z0-9\s&.,/-]{1,200}?)(?:\s+PROPRIETOR|\s+\d{1,3}\s*[,]|$)', first_line, re.IGNORECASE)
                
                if not name_match:
                    # Most permissive: take entire first line if it doesn't start with number
                    if not re.match(r'^\d', first_line):
                        name_match = re.match(r'^([A-Z][^\n]{1,200}?)(?:\s+PROPRIETOR|$)', first_line, re.IGNORECASE)
                
                if name_match:
                    name = name_match.group(1).strip()
                    # Clean up: normalize whitespace, remove trailing commas
                    name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
                    name = name.rstrip(',.')  # Remove trailing punctuation
                    
                    # Validate: should be 2-250 chars (allow long company names)
                    if 2 <= len(name) <= 250:
                        words = name.split()
                        # Must have at least 1 word
                        if len(words) >= 1:
                            # Exclude false positives
                            name_upper = name.upper()
                            if name_upper not in false_positives and not any(fp in name_upper for fp in false_positives):
                                # Check that it's not just numbers or special chars
                                if re.search(r'[A-Z]', name_upper):
                                    metadata['applicant_name'] = name
        
        # Pattern 2: Fallback - Look for "NAME PROPRIETOR" pattern
        if not metadata['applicant_name']:
            proprietor_match = re.search(r'([A-Z][A-Z0-9\s&.,/-]{2,200}?)\s+PROPRIETOR', text, re.IGNORECASE)
            if proprietor_match:
                name = proprietor_match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                name = name.rstrip(',.')
                if 2 <= len(name) <= 250:
                    words = name.split()
                    if len(words) >= 1:
                        name_upper = name.upper()
                        if name_upper not in false_positives and not any(fp in name_upper for fp in false_positives):
                            if re.search(r'[A-Z]', name_upper):
                                metadata['applicant_name'] = name
        
        # Pattern 3: Fallback - Look for text between application number and address patterns
        if not metadata['applicant_name']:
            # Find text between application number/date and common address patterns
            app_date_pattern = r'(\d{6,8})\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            address_pattern = r'(PROPRIETOR|\d{1,3}\s*,\s*[A-Z]|RAJGARH|KAROTH|ALWAR|RAJASTHAN|PUNJAB|Address)'
            
            app_match = re.search(app_date_pattern, text)
            # Capture application number if not already captured
            if app_match and not metadata['application_no']:
                metadata['application_no'] = app_match.group(1)
            if app_match:
                start_pos = app_match.end()
                # Find next address indicator
                remaining = text[start_pos:]
                addr_match = re.search(address_pattern, remaining, re.IGNORECASE)
                
                if addr_match:
                    end_pos = addr_match.start()
                    potential_name = remaining[:end_pos].strip()
                    
                    # Clean and validate
                    potential_name = re.sub(r'\s+', ' ', potential_name)
                    potential_name = potential_name.rstrip(',.')
                    
                    # Split by newline and take first line
                    first_line = potential_name.split('\n')[0].strip()
                    
                    if 2 <= len(first_line) <= 250 and re.search(r'[A-Z]', first_line.upper()):
                        words = first_line.split()
                        if len(words) >= 1:
                            name_upper = first_line.upper()
                            if name_upper not in false_positives and not any(fp in name_upper for fp in false_positives):
                                metadata['applicant_name'] = first_line
        
        # Pattern 4: Last resort - Extract first substantial text block after application number
        # This ensures we always get something if the above patterns fail
        if not metadata['applicant_name']:
            app_date_pattern = r'(\d{6,8})\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            app_match = re.search(app_date_pattern, text)
            
            # Capture application number if not already captured
            if app_match and not metadata['application_no']:
                metadata['application_no'] = app_match.group(1)
            
            if app_match:
                start_pos = app_match.end()
                remaining = text[start_pos:].strip()
                
                # Get first 300 characters (should be enough for longest names)
                first_chunk = remaining[:300]
                
                # Extract first line or first substantial block
                lines = [line.strip() for line in first_chunk.split('\n') if line.strip()]
                if lines:
                    candidate = lines[0]
                else:
                    # No newlines, try to extract until we hit a number or common stop word
                    candidate_match = re.match(r'^([A-Z][A-Z0-9\s&.,/-]{2,200}?)(?:\s+\d|$)', first_chunk, re.IGNORECASE)
                    if candidate_match:
                        candidate = candidate_match.group(1).strip()
                    else:
                        candidate = first_chunk.split()[0] if first_chunk.split() else ''
                
                # Clean and validate
                candidate = re.sub(r'\s+', ' ', candidate).strip()
                candidate = candidate.rstrip(',.')
                
                # Final validation
                if 2 <= len(candidate) <= 250:
                    words = candidate.split()
                    if len(words) >= 1:
                        name_upper = candidate.upper()
                        # Only exclude if it's clearly a false positive
                        if name_upper not in false_positives and not any(fp in name_upper for fp in false_positives):
                            if re.search(r'[A-Z]', name_upper):
                                metadata['applicant_name'] = candidate
        
        return metadata