import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class QdrantManager:
    """Helper class for Qdrant vector database operations"""
    
    def __init__(self):
        """Initialize the Qdrant client"""
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
    
    def get_collections(self) -> List[str]:
        """Get a list of all collections"""
        collections = self.client.get_collections().collections
        return [collection.name for collection in collections]
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        collections = self.get_collections()
        return collection_name in collections
    
    def create_collection(self, collection_name: str, vector_size: int = 1536, recreate: bool = False) -> bool:
        """
        Create a new collection for embeddings
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the embedding vectors
            recreate: Whether to recreate the collection if it exists
            
        Returns:
            True if the collection was created or already exists
        """
        try:
            # Check if collection exists
            if self.collection_exists(collection_name):
                if recreate:
                    logger.info(f"Recreating collection '{collection_name}'...")
                    self.client.delete_collection(collection_name=collection_name)
                else:
                    logger.info(f"Collection '{collection_name}' already exists.")
                    return True
            
            # Create collection with text index for "text" field
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            
            # Create text index for the "text" field to enable text search
            self.create_text_index(collection_name, "text")
            
            logger.info(f"Created new collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def create_text_index(self, collection_name: str, field_name: str) -> bool:
        """
        Create a text index for a field in a collection
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field to index
            
        Returns:
            True if the index was created successfully
        """
        try:
            # Create text index for improved search
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=f"payload.{field_name}",
                field_schema=qdrant_models.TextIndexParams(
                    type="text",
                    tokenizer=qdrant_models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True
                )
            )
            logger.info(f"Created text index for '{field_name}' in collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error creating text index: {e}")
            return False
    
    def vector_search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        filter_condition: Optional[qdrant_models.Filter] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity search
        
        Args:
            collection_name: Name of the collection to search
            query_vector: Embedding vector of the query
            filter_condition: Optional filter to apply to the search
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=filter_condition,
                limit=limit
            )
            
            # Convert results to a more usable format
            formatted_results = []
            for hit in results:
                # Extract payload and metadata
                result = {
                    "id": hit.id,
                    "score": hit.score,
                }
                
                # Add payload if available
                if hasattr(hit, 'payload'):
                    result["text"] = hit.payload.get("text", "")
                    result["metadata"] = hit.payload.get("metadata", {})
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []

# Singleton instance
qdrant_manager = QdrantManager()