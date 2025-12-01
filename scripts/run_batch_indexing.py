"""
Script to run batch indexing of trademarks
Usage: python run_batch_indexing.py --images_dir ./data/trademarks --metadata ./data/metadata.csv
"""

import argparse
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from preprocessing.image_processor import ImagePreprocessor
from models.embedding_model import EmbeddingGenerator
from database.vector_db import VectorDatabase
from indexing.batch_indexer import BatchIndexer
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Batch index trademarks")
    parser.add_argument("--images_dir", type=str, required=True, 
                       help="Directory containing trademark images")
    parser.add_argument("--metadata", type=str, required=True,
                       help="Path to metadata CSV/JSON file")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE,
                       help="Batch size for processing")
    parser.add_argument("--remove_bg", action="store_true",
                       help="Remove image backgrounds")
    
    args = parser.parse_args()
    
    # Setup logger
    Config.create_directories()
    logger = setup_logger(
        "batch_indexing",
        Config.LOG_PATH / f"indexing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logger.info("="*80)
    logger.info("TRADEMARK BATCH INDEXING")
    logger.info("="*80)
    logger.info(f"Images directory: {args.images_dir}")
    logger.info(f"Metadata file: {args.metadata}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Initialize components
    logger.info("Initializing components...")
    
    preprocessor = ImagePreprocessor(
        target_size=(224, 224),
        remove_bg=args.remove_bg
    )
    
    embedding_generator = EmbeddingGenerator(
        model_name=Config.EMBEDDING_MODEL,
        device=Config.DEVICE
    )
    
    vector_db = VectorDatabase(
        host=Config.QDRANT_HOST,
        port=Config.QDRANT_PORT,
        collection_name=Config.QDRANT_COLLECTION_NAME,
        embedding_dim=embedding_generator.get_embedding_dimension(),
        api_key=Config.QDRANT_API_KEY
    )
    
    batch_indexer = BatchIndexer(
        preprocessor=preprocessor,
        embedding_generator=embedding_generator,
        vector_db=vector_db,
        logger=logger
    )
    
    # Run indexing
    logger.info("Starting batch indexing...")
    stats = batch_indexer.process_batch(
        images_dir=Path(args.images_dir),
        metadata_path=Path(args.metadata),
        batch_size=args.batch_size
    )
    
    # Save stats
    stats_file = Config.LOG_PATH / f"indexing_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    batch_indexer.save_stats(stats_file)
    
    # Print summary
    logger.info("="*80)
    logger.info("INDEXING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total processed: {stats['total_processed']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Duration: {stats['duration_seconds']:.2f} seconds")
    logger.info(f"Speed: {stats['trademarks_per_second']:.2f} trademarks/second")
    logger.info(f"Stats saved to: {stats_file}")
    
    if stats['errors']:
        logger.warning(f"Encountered {len(stats['errors'])} errors. Check log file for details.")

if __name__ == "__main__":
    main()
