from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType, Range, ScrollRequest
from typing import List, Dict, Optional, Tuple
import uuid
import numpy as np
from datetime import datetime, timedelta

class VectorDatabase:
    """Manages vector database operations using Qdrant"""
    
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 collection_name: str = "trademarks", embedding_dim: int = 512,
                 api_key: Optional[str] = None):
        """
        Initialize vector database connection
        
        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of collection
            embedding_dim: Dimension of embeddings
            api_key: API key for cloud instance
        """
        # Connect to Qdrant
        if api_key:
            # Cloud instance - use URL
            if host.startswith('http'):
                
                self.client = QdrantClient(url=host, api_key=api_key)
            else:
                self.client = QdrantClient(host=host, port=port, api_key=api_key)
        else:
            # Local instance
            self.client = QdrantClient(host=host, port=port)
        
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Create collection if doesn't exist
        self._create_collection()
        
        # Try to create indexes for commonly filtered fields
        self._ensure_indexes()
    
    def _create_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")
    
    def _ensure_indexes(self):
        """Create indexes for commonly filtered fields if they don't exist"""
        try:
            # Try to create index for trademark_class (commonly used for filtering)
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="trademark_class",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"Created index for 'trademark_class'")
            except Exception as e:
                # Index might already exist or collection might not support it
                if "already exists" not in str(e).lower():
                    pass  # Silently ignore
            
            # Try to create index for source (used for filtering self_database vs pdf_extraction)
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"Created index for 'source'")
            except Exception as e:
                # Index might already exist
                if "already exists" not in str(e).lower():
                    pass  # Silently ignore
                    
        except Exception as e:
            print(f"Note: Index creation not available or failed: {e}")
            print("Post-filtering will be used for filtering")
    
    def insert_trademark(self, trademark_id: str, embedding: np.ndarray, 
                        metadata: Dict) -> bool:
        """
        Insert single trademark into database
        
        Args:
            trademark_id: Unique trademark identifier
            embedding: Embedding vector
            metadata: Additional metadata (name, class, date, etc.)
            
        Returns:
            Success status
        """
        try:
            point = PointStruct(
                id=trademark_id,
                vector=embedding.tolist(),
                payload=metadata
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            return True
        except Exception as e:
            print(f"Error inserting trademark {trademark_id}: {e}")
            return False
    
    def insert_trademarks_batch(self, trademarks: List[Tuple[str, np.ndarray, Dict]]) -> int:
        """
        Insert batch of trademarks
        
        Args:
            trademarks: List of (id, embedding, metadata) tuples
            
        Returns:
            Number of successfully inserted trademarks
        """
        points = []
        for trademark_id, embedding, metadata in trademarks:
            point = PointStruct(
                id=trademark_id,
                vector=embedding.tolist(),
                payload=metadata
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return len(points)
        except Exception as e:
            print(f"Error inserting batch: {e}")
            return 0
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10,
                      score_threshold: Optional[float] = None,
                      filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar trademarks
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_dict: Metadata filters (e.g., {"trademark_class": "25"})
                        If Qdrant doesn't have indexes, filtering is done in post-processing
            
        Returns:
            List of similar trademarks with scores
        """
        # Build filter if provided
        query_filter = None
        use_qdrant_filter = False
        
        if filter_dict:
            try:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                query_filter = Filter(must=conditions)
                use_qdrant_filter = True
            except Exception as e:
                # If filter building fails, we'll do post-filtering
                print(f"Warning: Could not build Qdrant filter, will use post-filtering: {e}")
                use_qdrant_filter = False
        
        # Search using query_points (newer API)
        query_vector_list = query_embedding.tolist()
        
        # Get more results if we need to post-filter (to ensure we have enough after filtering)
        search_limit = top_k * 3 if filter_dict and not use_qdrant_filter else top_k
        
        # Build parameters dict
        query_params = {
            "collection_name": self.collection_name,
            "query": query_vector_list,
            "limit": search_limit,
            "with_payload": True,
            "with_vectors": False
        }
        
        if score_threshold is not None:
            query_params["score_threshold"] = score_threshold
        
        # Try to use Qdrant filter if available
        if query_filter is not None and use_qdrant_filter:
            try:
                query_params["query_filter"] = query_filter
                results = self.client.query_points(**query_params)
            except Exception as e:
                # If filter fails (e.g., index not found), fall back to post-filtering
                print(f"Warning: Qdrant filter failed (likely missing index), using post-filtering: {e}")
                # Remove filter from params and search without it
                query_params.pop("query_filter", None)
                results = self.client.query_points(**query_params)
                use_qdrant_filter = False
        else:
            results = self.client.query_points(**query_params)
        
        # Format results
        formatted_results = []
        for result in results.points:
            # Clamp similarity score to valid range [0, 1] to avoid Pydantic validation errors
            similarity_score = max(0.0, min(1.0, float(result.score)))
            
            # Post-filter if Qdrant filter wasn't used or failed
            if filter_dict and not use_qdrant_filter:
                metadata = result.payload or {}
                # Check if this result matches all filter criteria
                matches_filter = True
                for key, value in filter_dict.items():
                    metadata_value = metadata.get(key)
                    # Convert to string for comparison (handles both string and numeric values)
                    if str(metadata_value) != str(value):
                        matches_filter = False
                        break
                
                if not matches_filter:
                    continue  # Skip this result
            
            formatted_results.append({
                'trademark_id': result.id,
                'similarity_score': similarity_score,
                'metadata': result.payload
            })
        
        # Limit to top_k after post-filtering
        return formatted_results[:top_k]
    
    def get_trademark(self, trademark_id: str) -> Optional[Dict]:
        """Get trademark by ID"""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[trademark_id]
            )
            if result:
                return {
                    'trademark_id': result[0].id,
                    'metadata': result[0].payload
                }
            return None
        except Exception as e:
            print(f"Error retrieving trademark {trademark_id}: {e}")
            return None
    
    def delete_trademark(self, trademark_id: str) -> bool:
        """Delete trademark by ID"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[trademark_id]
            )
            return True
        except Exception as e:
            print(f"Error deleting trademark {trademark_id}: {e}")
            return False
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)
        return {
            'total_trademarks': info.points_count,
            'vector_dimension': info.config.params.vectors.size,
            'distance_metric': info.config.params.vectors.distance
        }
    
    def clear_collection(self):
        """Delete all points in collection (use with caution!)"""
        self.client.delete_collection(self.collection_name)
        self._create_collection()
        print(f"Collection {self.collection_name} cleared")
    
    def get_trademarks_by_date_range(self, start_date: datetime, end_date: datetime, 
                                      source_filter: Optional[List[str]] = None,
                                      limit: int = 10000) -> List[Dict]:
        """
        Get all trademarks indexed within a date range
        
        Args:
            start_date: Start datetime (inclusive)
            end_date: End datetime (inclusive)
            source_filter: Optional list of sources to filter by (e.g., ['pdf_extraction', 'pdf_text_extraction'])
            limit: Maximum number of results to return
            
        Returns:
            List of trademark dictionaries with id, metadata, and vector
        """
        try:
            # Convert dates to ISO format strings for comparison
            start_str = start_date.isoformat()
            end_str = end_date.isoformat()
            
            # Scroll through all points and filter by indexed_at
            all_trademarks = []
            offset = None
            
            while True:
                # Scroll through points in batches
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                points, next_offset = results
                
                for point in points:
                    if point.payload:
                        indexed_at = point.payload.get('indexed_at', '')
                        
                        # Check if indexed_at is within the date range
                        if indexed_at and start_str <= indexed_at <= end_str:
                            # Check source filter if provided
                            if source_filter:
                                source = point.payload.get('source', '')
                                if source not in source_filter:
                                    continue
                            
                            all_trademarks.append({
                                'trademark_id': point.id,
                                'metadata': point.payload,
                                'vector': point.vector
                            })
                            
                            if len(all_trademarks) >= limit:
                                break
                
                if next_offset is None or len(all_trademarks) >= limit:
                    break
                    
                offset = next_offset
            
            print(f"Found {len(all_trademarks)} trademarks between {start_str} and {end_str}")
            return all_trademarks
            
        except Exception as e:
            print(f"Error getting trademarks by date range: {e}")
            return []
    
    def get_trademarks_by_monday(self, monday_date: Optional[datetime] = None,
                                  include_sources: Optional[List[str]] = None) -> List[Dict]:
        """
        Get all trademarks indexed on a specific Monday (journal upload day)
        
        Args:
            monday_date: The Monday date to filter by. If None, uses the most recent Monday.
            include_sources: Optional list of sources to include (defaults to journal sources)
            
        Returns:
            List of trademark dictionaries
        """
        # Calculate Monday date if not provided
        if monday_date is None:
            today = datetime.now()
            # Get the most recent Monday (0 = Monday)
            days_since_monday = today.weekday()
            monday_date = today - timedelta(days=days_since_monday)
        
        # Set time range for the entire Monday
        start_of_day = monday_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = monday_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Default sources for journal uploads
        if include_sources is None:
            include_sources = ['pdf_extraction', 'pdf_text_extraction']
        
        return self.get_trademarks_by_date_range(
            start_date=start_of_day,
            end_date=end_of_day,
            source_filter=include_sources
        )
    
    def get_trademarks_by_journal_monday(self, monday_date: Optional[datetime] = None,
                                          include_sources: Optional[List[str]] = None,
                                          limit: int = 10000) -> List[Dict]:
        """
        Get all trademarks that belong to a specific journal Monday date.
        Uses the 'journal_monday_date' metadata field for exact matching.
        
        Args:
            monday_date: The Monday date to filter by. If None, uses the most recent Monday.
            include_sources: Optional list of sources to include (defaults to journal sources)
            limit: Maximum number of results to return
            
        Returns:
            List of trademark dictionaries
        """
        # Calculate Monday date if not provided
        if monday_date is None:
            today = datetime.now()
            days_since_monday = today.weekday()
            monday_date = today - timedelta(days=days_since_monday)
        
        monday_date = monday_date.replace(hour=0, minute=0, second=0, microsecond=0)
        monday_str = monday_date.strftime("%Y-%m-%d")
        
        # Default sources for journal uploads
        if include_sources is None:
            include_sources = ['pdf_extraction', 'pdf_text_extraction']
        
        try:
            all_trademarks = []
            offset = None
            
            while True:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                points, next_offset = results
                
                for point in points:
                    if point.payload:
                        # Check journal_monday_date field
                        journal_monday = point.payload.get('journal_monday_date', '')
                        
                        if journal_monday == monday_str:
                            # Check source filter if provided
                            if include_sources:
                                source = point.payload.get('source', '')
                                if source not in include_sources:
                                    continue
                            
                            all_trademarks.append({
                                'trademark_id': point.id,
                                'metadata': point.payload,
                                'vector': point.vector
                            })
                            
                            if len(all_trademarks) >= limit:
                                break
                
                if next_offset is None or len(all_trademarks) >= limit:
                    break
                    
                offset = next_offset
            
            print(f"Found {len(all_trademarks)} trademarks for journal Monday {monday_str}")
            return all_trademarks
            
        except Exception as e:
            print(f"Error getting trademarks by journal Monday: {e}")
            return []
    
    def get_all_trademarks_paginated(self, batch_size: int = 1000) -> List[Dict]:
        """
        Get all trademarks from the collection with pagination
        
        Args:
            batch_size: Number of points to retrieve per batch
            
        Returns:
            List of all trademark dictionaries
        """
        try:
            all_trademarks = []
            offset = None
            
            while True:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                points, next_offset = results
                
                for point in points:
                    all_trademarks.append({
                        'trademark_id': point.id,
                        'metadata': point.payload,
                        'vector': point.vector if point.vector else None
                    })
                
                if next_offset is None:
                    break
                    
                offset = next_offset
            
            print(f"Retrieved {len(all_trademarks)} total trademarks")
            return all_trademarks
            
        except Exception as e:
            print(f"Error getting all trademarks: {e}")
            return []
    
    def get_points_by_source(self, source: str, limit: int = 100000) -> List[Dict]:
        """
        Get all points with a specific source value
        
        Args:
            source: Source value to filter by (e.g., 'self_database', 'pdf_extraction')
            limit: Maximum number of results
            
        Returns:
            List of point dictionaries with id, metadata, vector
        """
        try:
            all_points = []
            offset = None
            
            while True:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                points, next_offset = results
                
                for point in points:
                    if point.payload:
                        point_source = point.payload.get('source', '')
                        if point_source == source:
                            all_points.append({
                                'point_id': point.id,
                                'metadata': point.payload,
                                'vector': point.vector
                            })
                            
                            if len(all_points) >= limit:
                                break
                
                if next_offset is None or len(all_points) >= limit:
                    break
                    
                offset = next_offset
            
            print(f"Found {len(all_points)} points with source '{source}'")
            return all_points
            
        except Exception as e:
            print(f"Error getting points by source: {e}")
            return []
    
    def point_exists(self, point_id: str) -> bool:
        """
        Check if a point with given ID exists in the collection
        
        Args:
            point_id: The point ID to check
            
        Returns:
            True if exists, False otherwise
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            return len(result) > 0
        except Exception as e:
            print(f"Error checking point existence: {e}")
            return False
    
    def get_existing_point_ids(self, point_ids: List[str]) -> set:
        """
        Check which of the given point IDs already exist in the collection
        
        Args:
            point_ids: List of point IDs to check
            
        Returns:
            Set of existing point IDs
        """
        try:
            # Batch check for efficiency
            existing = set()
            batch_size = 100
            
            for i in range(0, len(point_ids), batch_size):
                batch = point_ids[i:i + batch_size]
                result = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=batch
                )
                for point in result:
                    existing.add(point.id)
            
            return existing
        except Exception as e:
            print(f"Error checking existing point IDs: {e}")
            return set()
    
    def delete_points_by_ids(self, point_ids: List[str]) -> int:
        """
        Delete multiple points by their IDs
        
        Args:
            point_ids: List of point IDs to delete
            
        Returns:
            Number of points deleted
        """
        if not point_ids:
            return 0
            
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            print(f"Deleted {len(point_ids)} points")
            return len(point_ids)
        except Exception as e:
            print(f"Error deleting points: {e}")
            return 0
    
    def upsert_point(self, point_id: str, embedding: np.ndarray, metadata: Dict) -> bool:
        """
        Insert or update a single point with a specific string ID
        
        Args:
            point_id: String ID for the point (e.g., 'self_12345')
            embedding: Embedding vector
            metadata: Point metadata
            
        Returns:
            Success status
        """
        try:
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=metadata
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            return True
        except Exception as e:
            print(f"Error upserting point {point_id}: {e}")
            return False
    
    def upsert_points_batch(self, points: List[Tuple[str, np.ndarray, Dict]]) -> int:
        """
        Insert or update multiple points with specific string IDs
        
        Args:
            points: List of (point_id, embedding, metadata) tuples
            
        Returns:
            Number of successfully upserted points
        """
        if not points:
            return 0
            
        try:
            point_structs = []
            for point_id, embedding, metadata in points:
                point_structs.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=metadata
                ))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=point_structs
            )
            return len(point_structs)
        except Exception as e:
            print(f"Error upserting batch: {e}")
            return 0
    
    def search_similar_by_source(self, query_embedding: np.ndarray, source: str,
                                  top_k: int = 10, score_threshold: Optional[float] = None) -> List[Dict]:
        """
        Search for similar vectors filtered by source
        
        Args:
            query_embedding: Query embedding vector
            source: Source to filter by (e.g., 'self_database')
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar points with scores
        """
        try:
            # Build filter for source
            query_filter = Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source))
                ]
            )
            
            query_params = {
                "collection_name": self.collection_name,
                "query": query_embedding.tolist(),
                "limit": top_k,
                "with_payload": True,
                "with_vectors": False,
                "query_filter": query_filter
            }
            
            if score_threshold is not None:
                query_params["score_threshold"] = score_threshold
            
            results = self.client.query_points(**query_params)
            
            formatted_results = []
            for result in results.points:
                similarity_score = max(0.0, min(1.0, float(result.score)))
                formatted_results.append({
                    'point_id': result.id,
                    'similarity_score': similarity_score,
                    'metadata': result.payload
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching similar by source: {e}")
            # Fallback to post-filtering
            return self._search_similar_by_source_fallback(
                query_embedding, source, top_k, score_threshold
            )
    
    def _search_similar_by_source_fallback(self, query_embedding: np.ndarray, source: str,
                                            top_k: int, score_threshold: Optional[float]) -> List[Dict]:
        """Fallback method using post-filtering when Qdrant filter fails"""
        try:
            query_params = {
                "collection_name": self.collection_name,
                "query": query_embedding.tolist(),
                "limit": top_k * 5,  # Get more to allow for filtering
                "with_payload": True,
                "with_vectors": False
            }
            
            if score_threshold is not None:
                query_params["score_threshold"] = score_threshold
            
            results = self.client.query_points(**query_params)
            
            formatted_results = []
            for result in results.points:
                if result.payload and result.payload.get('source') == source:
                    similarity_score = max(0.0, min(1.0, float(result.score)))
                    formatted_results.append({
                        'point_id': result.id,
                        'similarity_score': similarity_score,
                        'metadata': result.payload
                    })
                    
                    if len(formatted_results) >= top_k:
                        break
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in fallback search: {e}")
            return []