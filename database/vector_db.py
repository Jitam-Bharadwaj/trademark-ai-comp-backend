from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from typing import List, Dict, Optional, Tuple
import uuid
import numpy as np

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
                    print(f"Note: Could not create index for 'trademark_class': {e}")
                    print("Post-filtering will be used instead")
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