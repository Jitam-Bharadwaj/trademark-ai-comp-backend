"""
Self Database Indexer - Indexes self DB marks into Qdrant with CLIP text embeddings

This enables cross-modal similarity search between:
- Journal trademark IMAGES (CLIP image embeddings)
- Self database TEXT marks (CLIP text embeddings)

Usage:
  # Full sync (add new, remove deleted)
  python scripts/index_self_db.py
  
  # Force re-index all marks
  python scripts/index_self_db.py --force
  
  # Dry run (show what would be done)
  python scripts/index_self_db.py --dry-run
"""

import os
import sys
import argparse
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Ensure project root is on sys.path
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from utils.logger import setup_logger
from database.vector_db import VectorDatabase
from database.application_queries import application_queries
from models.embedding_model import EmbeddingGenerator

# Constants
SELF_DB_SOURCE = "self_database"
# UUID namespace for generating deterministic UUIDs from application IDs
SELF_DB_UUID_NAMESPACE = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')

# Ensure log directory exists
Config.LOG_PATH.mkdir(parents=True, exist_ok=True)

logger: logging.Logger = setup_logger(
    "index_self_db",
    Config.LOG_PATH / f"index_self_db_{datetime.now().strftime('%Y%m%d')}.log",
    level=getattr(logging, Config.LOG_LEVEL),
)


def get_point_id(application_id: int) -> str:
    """
    Generate deterministic UUID from application ID.
    Uses UUID5 with a fixed namespace to ensure same app_id always produces same UUID.
    """
    return str(uuid.uuid5(SELF_DB_UUID_NAMESPACE, f"self_db_{application_id}"))


def extract_application_id_from_metadata(metadata: Dict) -> int:
    """Extract application ID from point metadata"""
    return metadata.get('application_id', 0)


def init_vector_db() -> VectorDatabase:
    """Initialize vector database connection"""
    logger.info("Connecting to vector database...")
    
    if Config.QDRANT_API_KEY:
        qdrant_url = Config.QDRANT_HOST if Config.QDRANT_HOST.startswith('http') else f"https://{Config.QDRANT_HOST}"
        vector_db = VectorDatabase(
            host=qdrant_url,
            port=Config.QDRANT_PORT,
            collection_name=Config.QDRANT_COLLECTION_NAME,
            embedding_dim=Config.EMBEDDING_DIMENSION,
            api_key=Config.QDRANT_API_KEY
        )
    else:
        vector_db = VectorDatabase(
            host=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT,
            collection_name=Config.QDRANT_COLLECTION_NAME,
            embedding_dim=Config.EMBEDDING_DIMENSION,
            api_key=None
        )
    
    logger.info("Connected to vector database")
    return vector_db


def init_embedding_generator() -> EmbeddingGenerator:
    """Initialize CLIP embedding generator"""
    logger.info("Initializing CLIP embedding generator...")
    embedding_generator = EmbeddingGenerator()
    logger.info(f"CLIP model loaded (embedding dim: {embedding_generator.embedding_dim})")
    return embedding_generator


def get_self_db_marks() -> Dict[int, Dict]:
    """
    Get all marks from self database (MySQL)
    
    Returns:
        Dict mapping application_id to mark data
    """
    logger.info("Fetching marks from self database (MySQL)...")
    
    # Get all marks
    marks = application_queries.get_all_marks(use_cache=False)
    
    if not marks:
        logger.warning("No marks found in self database")
        return {}
    
    # Get application details for all marks
    app_ids = [m['application_id'] for m in marks if m.get('application_id')]
    app_details = application_queries.get_applications_by_ids(app_ids)
    
    # Build result dict
    result = {}
    for mark in marks:
        app_id = mark.get('application_id')
        if app_id and mark.get('mark'):
            details = app_details.get(app_id, {})
            result[app_id] = {
                'application_id': app_id,
                'mark': mark.get('mark', ''),
                'trademark_class': details.get('trademark_class', ''),
                'applicant_name': details.get('applicant_name', ''),
                'application_no': details.get('application_no', '')
            }
    
    logger.info(f"Found {len(result)} marks in self database")
    return result


def get_existing_self_db_points(vector_db: VectorDatabase) -> Dict[int, Dict]:
    """
    Get all existing self DB points from Qdrant
    
    Returns:
        Dict mapping application_id to point data
    """
    logger.info("Fetching existing self DB points from Qdrant...")
    
    points = vector_db.get_points_by_source(SELF_DB_SOURCE)
    
    result = {}
    for point in points:
        point_id = point['point_id']
        metadata = point.get('metadata', {})
        
        # Get application_id from metadata (stored when we indexed)
        app_id = extract_application_id_from_metadata(metadata)
        
        if app_id:
            result[app_id] = {
                'point_id': point_id,
                'metadata': metadata,
                'vector': point['vector']
            }
        else:
            logger.warning(f"No application_id in metadata for point: {point_id}")
    
    logger.info(f"Found {len(result)} existing self DB points in Qdrant")
    return result


def sync_self_db_to_qdrant(vector_db: VectorDatabase, 
                           embedding_generator: EmbeddingGenerator,
                           force: bool = False,
                           dry_run: bool = False) -> Dict[str, int]:
    """
    Sync self database marks to Qdrant vector database
    
    Args:
        vector_db: VectorDatabase instance
        embedding_generator: EmbeddingGenerator instance
        force: If True, re-index all marks (delete and re-add)
        dry_run: If True, only show what would be done
        
    Returns:
        Dict with counts: {'added': n, 'deleted': n, 'unchanged': n}
    """
    stats = {'added': 0, 'deleted': 0, 'unchanged': 0, 'failed': 0}
    
    # Get current state
    self_db_marks = get_self_db_marks()
    existing_points = get_existing_self_db_points(vector_db)
    
    self_db_ids = set(self_db_marks.keys())
    existing_ids = set(existing_points.keys())
    
    # Calculate what needs to be done
    if force:
        # Force mode: re-index everything
        to_add = self_db_ids
        to_delete = existing_ids
        unchanged = set()
    else:
        # Incremental mode
        to_add = self_db_ids - existing_ids  # In self DB but not in Qdrant
        to_delete = existing_ids - self_db_ids  # In Qdrant but not in self DB
        unchanged = self_db_ids & existing_ids  # In both
    
    logger.info("=" * 50)
    logger.info("Sync Plan:")
    logger.info(f"  Self DB marks:      {len(self_db_ids)}")
    logger.info(f"  Existing in Qdrant: {len(existing_ids)}")
    logger.info(f"  To ADD:             {len(to_add)}")
    logger.info(f"  To DELETE:          {len(to_delete)}")
    logger.info(f"  Unchanged:          {len(unchanged)}")
    logger.info("=" * 50)
    
    if dry_run:
        logger.info("DRY RUN - No changes will be made")
        if to_add:
            logger.info(f"Would add: {list(to_add)[:10]}{'...' if len(to_add) > 10 else ''}")
        if to_delete:
            logger.info(f"Would delete: {list(to_delete)[:10]}{'...' if len(to_delete) > 10 else ''}")
        return {'added': len(to_add), 'deleted': len(to_delete), 'unchanged': len(unchanged)}
    
    # Delete removed marks
    if to_delete:
        logger.info(f"Deleting {len(to_delete)} removed marks from Qdrant...")
        point_ids_to_delete = [get_point_id(app_id) for app_id in to_delete]
        deleted = vector_db.delete_points_by_ids(point_ids_to_delete)
        stats['deleted'] = deleted
        logger.info(f"Deleted {deleted} points")
    
    # Add new marks
    if to_add:
        logger.info(f"Adding {len(to_add)} new marks to Qdrant...")
        
        # Process in batches
        batch_size = 50
        to_add_list = list(to_add)
        
        for i in range(0, len(to_add_list), batch_size):
            batch_ids = to_add_list[i:i + batch_size]
            batch_marks = [self_db_marks[app_id] for app_id in batch_ids]
            
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(to_add_list) + batch_size - 1) // batch_size}")
            
            # Generate embeddings for batch
            texts = [m['mark'] for m in batch_marks]
            
            try:
                embeddings = embedding_generator.generate_text_embeddings_batch(texts)
                
                # Prepare points for upsert
                points_to_add = []
                for j, (app_id, mark_data) in enumerate(zip(batch_ids, batch_marks)):
                    point_id = get_point_id(app_id)
                    embedding = embeddings[j]
                    
                    metadata = {
                        'source': SELF_DB_SOURCE,
                        'application_id': app_id,
                        'mark': mark_data['mark'],
                        'trademark_class': mark_data.get('trademark_class', ''),
                        'applicant_name': mark_data.get('applicant_name', ''),
                        'application_no': mark_data.get('application_no', ''),
                        'indexed_at': datetime.now().isoformat(),
                        'embedding_type': 'clip_text'
                    }
                    
                    points_to_add.append((point_id, embedding, metadata))
                
                # Upsert batch
                added = vector_db.upsert_points_batch(points_to_add)
                stats['added'] += added
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                stats['failed'] += len(batch_ids)
    
    stats['unchanged'] = len(unchanged)
    
    logger.info("=" * 50)
    logger.info("Sync Complete:")
    logger.info(f"  Added:     {stats['added']}")
    logger.info(f"  Deleted:   {stats['deleted']}")
    logger.info(f"  Unchanged: {stats['unchanged']}")
    logger.info(f"  Failed:    {stats['failed']}")
    logger.info("=" * 50)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Index self database marks into Qdrant with CLIP text embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal sync (add new, remove deleted)
  python scripts/index_self_db.py
  
  # Force re-index all marks
  python scripts/index_self_db.py --force
  
  # Dry run (show what would be done)
  python scripts/index_self_db.py --dry-run
        """
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-index all marks (delete existing and re-add)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    # Create directories
    Config.create_directories()
    
    print("=" * 60)
    print("Self Database Indexer")
    print("=" * 60)
    print(f"Mode: {'FORCE RE-INDEX' if args.force else 'INCREMENTAL SYNC'}")
    print(f"Dry Run: {args.dry_run}")
    print("=" * 60)
    
    try:
        # Initialize components
        vector_db = init_vector_db()
        embedding_generator = init_embedding_generator()
        
        # Run sync
        stats = sync_self_db_to_qdrant(
            vector_db=vector_db,
            embedding_generator=embedding_generator,
            force=args.force,
            dry_run=args.dry_run
        )
        
        print("\n" + "=" * 60)
        print("Indexing Complete!")
        print(f"  Added:     {stats['added']}")
        print(f"  Deleted:   {stats['deleted']}")
        print(f"  Unchanged: {stats['unchanged']}")
        if stats.get('failed', 0) > 0:
            print(f"  Failed:    {stats['failed']}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

