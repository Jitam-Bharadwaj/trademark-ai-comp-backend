from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime

class BatchIndexer:
    """Handles batch indexing of trademarks"""
    
    def __init__(self, preprocessor: ImagePreprocessor, 
                 embedding_generator: EmbeddingGenerator,
                 vector_db: VectorDatabase,
                 logger=None):
        """
        Initialize batch indexer
        
        Args:
            preprocessor: Image preprocessing instance
            embedding_generator: Embedding generation instance
            vector_db: Vector database instance
            logger: Logger instance
        """
        self.preprocessor = preprocessor
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.logger = logger
        
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
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
    
    def load_metadata(self, metadata_path: Path) -> pd.DataFrame:
        """
        Load metadata from CSV or JSON file
        
        Expected columns: trademark_id, name, class, registration_date, owner, image_filename
        """
        if metadata_path.suffix == '.csv':
            return pd.read_csv(metadata_path)
        elif metadata_path.suffix == '.json':
            return pd.read_json(metadata_path)
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")
    
    def process_single_trademark(self, image_path: Path, metadata: Dict) -> bool:
        """Process and index single trademark"""
        try:
            # Preprocess image
            preprocessed = self.preprocessor.preprocess(image_path)
            if preprocessed is None:
                raise ValueError("Preprocessing failed")
            
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(preprocessed)
            
            # Insert into database
            success = self.vector_db.insert_trademark(
                trademark_id=metadata['trademark_id'],
                embedding=embedding,
                metadata=metadata
            )
            
            if success:
                self.stats['successful'] += 1
                return True
            else:
                raise ValueError("Database insertion failed")
                
        except Exception as e:
            self.stats['failed'] += 1
            self.stats['errors'].append({
                'trademark_id': metadata.get('trademark_id', 'unknown'),
                'image_path': str(image_path),
                'error': str(e)
            })
            self.log(f"Error processing {image_path}: {e}", "ERROR")
            return False
    
    def process_batch(self, images_dir: Path, metadata_path: Path, 
                     batch_size: int = 32) -> Dict:
        """
        Process batch of trademarks
        
        Args:
            images_dir: Directory containing trademark images
            metadata_path: Path to metadata file
            batch_size: Batch size for processing
            
        Returns:
            Statistics dictionary
        """
        self.log(f"Starting batch processing from {images_dir}")
        start_time = datetime.now()
        
        # Load metadata
        metadata_df = self.load_metadata(metadata_path)
        self.log(f"Loaded metadata for {len(metadata_df)} trademarks")
        
        # Reset stats
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'start_time': start_time.isoformat()
        }
        
        # Process in batches
        batch_images = []
        batch_metadata = []
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), 
                            desc="Processing trademarks"):
            self.stats['total_processed'] += 1
            
            # Get image path
            image_filename = row.get('image_filename', f"{row['trademark_id']}.png")
            image_path = images_dir / image_filename
            
            if not image_path.exists():
                self.log(f"Image not found: {image_path}", "WARNING")
                self.stats['failed'] += 1
                continue
            
            # Prepare metadata
            metadata = {
                'trademark_id': str(row['trademark_id']),
                'name': row.get('name', ''),
                'trademark_class': str(row.get('class', '')),
                'applicant_name': row.get('applicant_name', ''),
                'application_no': row.get('application_no', ''),
                'registration_date': str(row.get('registration_date', '')),
                'owner': row.get('owner', ''),
                'image_path': str(image_path),
                'indexed_at': datetime.now().isoformat()
            }
            
            # Preprocess
            preprocessed = self.preprocessor.preprocess(image_path)
            if preprocessed is None:
                self.stats['failed'] += 1
                continue
            
            batch_images.append(preprocessed)
            batch_metadata.append(metadata)
            
            # Process batch when full
            if len(batch_images) >= batch_size:
                self._process_batch_embeddings(batch_images, batch_metadata)
                batch_images = []
                batch_metadata = []
        
        # Process remaining
        if batch_images:
            self._process_batch_embeddings(batch_images, batch_metadata)
        
        # Finalize stats
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.stats['end_time'] = end_time.isoformat()
        self.stats['duration_seconds'] = duration
        self.stats['trademarks_per_second'] = self.stats['total_processed'] / duration if duration > 0 else 0
        
        self.log(f"Batch processing completed: {self.stats['successful']} successful, "
                f"{self.stats['failed']} failed in {duration:.2f} seconds")
        
        return self.stats
    
    def _process_batch_embeddings(self, batch_images: List, batch_metadata: List[Dict]):
        """Process batch of embeddings and insert into database"""
        try:
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings_batch(batch_images)
            
            # Prepare for insertion
            trademarks = [
                (metadata['trademark_id'], embedding, metadata)
                for embedding, metadata in zip(embeddings, batch_metadata)
            ]
            
            # Insert into database
            inserted = self.vector_db.insert_trademarks_batch(trademarks)
            self.stats['successful'] += inserted
            
        except Exception as e:
            self.log(f"Error processing batch: {e}", "ERROR")
            # Fallback to individual processing
            for img, metadata in zip(batch_images, batch_metadata):
                try:
                    embedding = self.embedding_generator.generate_embedding(img)
                    success = self.vector_db.insert_trademark(
                        metadata['trademark_id'], embedding, metadata
                    )
                    if success:
                        self.stats['successful'] += 1
                    else:
                        self.stats['failed'] += 1
                except Exception as e2:
                    self.stats['failed'] += 1
                    self.stats['errors'].append({
                        'trademark_id': metadata['trademark_id'],
                        'error': str(e2)
                    })
    
    def save_stats(self, output_path: Path):
        """Save indexing statistics to file"""
        with open(output_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        self.log(f"Stats saved to {output_path}")