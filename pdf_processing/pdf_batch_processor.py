from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from .pdf_extractor import PDFTrademarkExtractor

class PDFBatchProcessor:
    """Process batches of PDF files to extract trademarks"""
    
    def __init__(self, pdf_extractor: PDFTrademarkExtractor, 
                 output_dir: Path, logger=None):
        """
        Initialize PDF batch processor
        
        Args:
            pdf_extractor: PDF extractor instance
            output_dir: Directory to save extracted images
            logger: Logger instance
        """
        self.pdf_extractor = pdf_extractor
        self.output_dir = output_dir
        self.logger = logger
        
        self.stats = {
            'total_pdfs': 0,
            'successful_pdfs': 0,
            'failed_pdfs': 0,
            'total_images_extracted': 0,
            'errors': []
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log message"""
        if self.logger:
            if level == "INFO":
                self.logger.info(message)
            elif level == "ERROR":
                self.logger.error(message)
            elif level == "WARNING":
                self.logger.warning(message)
        else:
            print(f"{level}: {message}")
    
    def process_pdf_directory(self, pdf_dir: Path) -> Tuple[pd.DataFrame, Dict]:
        """
        Process all PDFs in a directory
        
        Args:
            pdf_dir: Directory containing PDF files
            
        Returns:
            Tuple of (metadata_dataframe, statistics)
        """
        self.log(f"Processing PDFs from: {pdf_dir}")
        
        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        self.stats['total_pdfs'] = len(pdf_files)
        
        if not pdf_files:
            self.log("No PDF files found!", "WARNING")
            return pd.DataFrame(), self.stats
        
        self.log(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        all_metadata = []
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                metadata_records = self._process_single_pdf(pdf_path)
                all_metadata.extend(metadata_records)
                self.stats['successful_pdfs'] += 1
                
            except Exception as e:
                self.stats['failed_pdfs'] += 1
                self.stats['errors'].append({
                    'pdf_file': str(pdf_path),
                    'error': str(e)
                })
                self.log(f"Error processing {pdf_path}: {e}", "ERROR")
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(all_metadata)
        
        self.log(f"Completed: {self.stats['successful_pdfs']} PDFs processed, "
                f"{self.stats['total_images_extracted']} images extracted")
        
        return metadata_df, self.stats
    
    def _process_single_pdf(self, pdf_path: Path) -> List[Dict]:
        """Process a single PDF file"""
        self.log(f"Processing: {pdf_path.name}")
        
        # Extract images from PDF
        images = self.pdf_extractor.extract_images_from_pdf(pdf_path)
        
        if not images:
            self.log(f"No images extracted from {pdf_path.name}", "WARNING")
            return []
        
        self.log(f"Extracted {len(images)} images from {pdf_path.name}")
        
        # Generate base name for this PDF
        base_name = pdf_path.stem
        
        # Save extracted images
        saved_paths = self.pdf_extractor.save_extracted_images(
            images, self.output_dir, base_name
        )
        
        # Create metadata records
        metadata_records = []
        for idx, (saved_path, (image, extract_meta)) in enumerate(zip(saved_paths, images)):
            # Generate unique trademark ID
            trademark_id = f"{base_name}_TM{idx:03d}"
            
            metadata = {
                'trademark_id': trademark_id,
                'name': f"{base_name} - Image {idx + 1}",
                'class': extract_meta.get('trademark_class', ''),  # Extracted from PDF text
                'applicant_name': extract_meta.get('applicant_name', ''),  # Extracted from PDF text
                'application_no': extract_meta.get('application_no', ''),  # Extracted from PDF text
                'registration_date': '',  # To be filled manually
                'owner': extract_meta.get('applicant_name', ''),  # Use applicant_name as owner
                'image_filename': saved_path.name,
                'source_pdf': pdf_path.name,
                'pdf_page': extract_meta['page'],
                'extraction_method': extract_meta['source'],
                'image_width': extract_meta['width'],
                'image_height': extract_meta['height']
            }
            
            metadata_records.append(metadata)
            self.stats['total_images_extracted'] += 1
        
        return metadata_records
    
    def save_metadata(self, metadata_df: pd.DataFrame, output_path: Path):
        """Save metadata to CSV file"""
        metadata_df.to_csv(output_path, index=False)
        self.log(f"Metadata saved to: {output_path}")