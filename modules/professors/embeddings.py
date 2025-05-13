import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import PointStruct

from utils.qdrant_helper import qdrant_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "professors"
DATA_DIR = "data/processed"
EMBEDDING_MODEL = "text-embedding-3-small"  # Using OpenAI's small embedding model
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ingest professor information into Qdrant vector database")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory containing faculty data files")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it exists")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="Embedding model to use")
    return parser.parse_args()

def load_professor_files(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all professor information files from the specified directory
    
    Args:
        data_dir: Directory containing faculty data files
        
    Returns:
        List of professor document dictionaries
    """
    documents = []
    data_path = Path(data_dir)
    
    # Look for the professors.json file
    file_path = data_path / "professors.json"
    
    if not file_path.exists():
        logger.warning(f"Professor data file not found at {file_path}")
        return []
    
    try:
        logger.info(f"Processing: {file_path}")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            professors = json.load(f)
        
        # Process each professor as a separate document
        for i, professor in enumerate(professors):
            # Add an ID if not present
            if "id" not in professor:
                professor["id"] = f"professor-{i}"
            
            # Ensure text field exists for embedding
            if "text" not in professor and "heading" in professor:
                professor["text"] = professor["heading"]
            
            # Enhance metadata
            enhanced_professor = enhance_professor_metadata(professor)
            documents.append(enhanced_professor)
        
        logger.info(f"Added {len(documents)} professor documents")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
    
    return documents

def enhance_professor_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance professor metadata for better searchability
    
    Args:
        item: Original professor item
        
    Returns:
        Enhanced professor item with additional metadata
    """
    # Create a copy to avoid modifying the original
    enhanced = item.copy()
    
    # Add module identifier to the metadata
    if "metadata" not in enhanced:
        enhanced["metadata"] = {}
    
    enhanced["metadata"]["module"] = "professors"
    enhanced["metadata"]["type"] = "professor_document"
    
    # Ensure keywords field exists
    if "keywords" not in enhanced:
        enhanced["keywords"] = []
    
    # Add relevant keywords based on professor info
    if "heading" in enhanced and enhanced["heading"] not in enhanced["keywords"]:
        enhanced["keywords"].append(enhanced["heading"])
    
    if "professor" not in enhanced["keywords"]:
        enhanced["keywords"].append("professor")
        
    if "faculty" not in enhanced["keywords"]:
        enhanced["keywords"].append("faculty")
    
    return enhanced

def generate_embeddings_and_upload(args, documents: List[Dict[str, Any]]):
    """
    Generate embeddings for professor documents and upload to Qdrant
    
    Args:
        args: Command line arguments
        documents: List of professor documents to embed
    """
    try:
        # Initialize OpenAI embeddings with the specified model
        embeddings = OpenAIEmbeddings(
            model=args.model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        logger.info(f"Using embedding model: {args.model}")
        
        # Ensure the collection exists
        qdrant_manager.create_collection(
            collection_name=args.collection,
            vector_size=EMBEDDING_DIMENSION,  # Embedding dimension
            recreate=args.recreate
        )
        
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        
        # Process in batches to avoid overloading the API
        batch_size = 10
        total_points = 0
        
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            batch_docs = documents[i:end_idx]
            
            try:
                # Extract text content for embeddings
                texts = [doc.get("text", "") for doc in batch_docs]
                
                # Generate embeddings
                batch_embeddings = embeddings.embed_documents(texts)
                
                # Prepare points
                points = []
                for j, (doc, embedding_vector) in enumerate(zip(batch_docs, batch_embeddings)):
                    # Create point ID - either use the document's ID or generate one
                    point_id = doc.get("id", f"professor-{total_points + j}")
                    
                    # Ensure point_id is a string or integer
                    if isinstance(point_id, str) and not point_id.isdigit():
                        numeric_id = total_points + j
                    else:
                        try:
                            numeric_id = int(point_id)
                        except ValueError:
                            numeric_id = total_points + j
                    
                    # Create metadata structure
                    metadata = doc.get("metadata", {})
                    
                    # Create point with embedding and payload
                    points.append(
                        PointStruct(
                            id=numeric_id,
                            vector=embedding_vector,
                            payload=doc  # Store the entire document as payload
                        )
                    )
                
                # Upload to Qdrant
                qdrant_manager.client.upsert(
                    collection_name=args.collection,
                    points=points
                )
                
                total_points += len(batch_docs)
                logger.info(f"Processed {total_points}/{len(documents)} documents")
                
            except Exception as e:
                logger.error(f"Error processing batch {i}-{end_idx}: {e}")
        
        logger.info(f"Successfully ingested {total_points} professor documents into Qdrant")
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")

def main():
    """Main process for ingesting professor information"""
    logger.info("Starting professor information ingestion...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load professor files
    documents = load_professor_files(args.data_dir)
    logger.info(f"Found {len(documents)} total professor documents/chunks")
    
    if not documents:
        logger.warning("No documents found. Please check the faculty data directory.")
        return
    
    # Generate embeddings and upload to Qdrant
    generate_embeddings_and_upload(args, documents)
    
    logger.info("Professor information ingestion complete!")

if __name__ == "__main__":
    main()