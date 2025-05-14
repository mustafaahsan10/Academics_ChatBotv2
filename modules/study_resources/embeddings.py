import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import PointStruct

# Add the project root directory to the Python path
# This allows the script to find modules in the parent directories
project_root = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, project_root)

# Now import from utils
from utils.qdrant_helper import qdrant_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "study_resources"
DATA_DIR = "data/processed"
EMBEDDING_MODEL = "text-embedding-3-small"  # Using OpenAI's small embedding model
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ingest study resources information into Qdrant vector database")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory containing resources data files")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it exists")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="Embedding model to use")
    return parser.parse_args()

def load_resource_files(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all study resource files from the specified directory
    
    Args:
        data_dir: Directory containing resource data files
        
    Returns:
        List of resource document dictionaries
    """
    documents = []
    data_path = Path(data_dir)
    
    # Look for the study_resource.json file (updated filename)
    file_path = data_path / "study_resource.json"
    
    if not file_path.exists():
        logger.warning(f"Study resources data file not found at {file_path}")
        return []
    
    try:
        logger.info(f"Processing: {file_path}")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            resources = json.load(f)
        
        # Process each resource as a separate document
        for i, resource in enumerate(resources):
            # Map the fields from the actual JSON structure
            mapped_resource = {
                "id": resource.get("id", f"resource-{i}"),
                "title": f"Study Material for {resource.get('Course Name', '')}",
                "course_name": resource.get("Course Name", ""),
                "resource_type": "Study Guide",
                "description": resource.get("Study Material", ""),
                "exam_links": resource.get("Exam Links", ""),
            }
            
            # Generate text field for embedding
            text = f"Course: {mapped_resource['course_name']}. Study Material: {mapped_resource['description']}. Exam Links: {mapped_resource['exam_links']}"
            mapped_resource["text"] = text
            
            # Enhance metadata
            enhanced_resource = enhance_resource_metadata(mapped_resource)
            documents.append(enhanced_resource)
        
        logger.info(f"Added {len(documents)} resource documents")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
    
    return documents

def enhance_resource_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance resource metadata for better searchability
    
    Args:
        item: Original resource item
        
    Returns:
        Enhanced resource item with additional metadata
    """
    # Create a copy to avoid modifying the original
    enhanced = item.copy()
    
    # Add module identifier to the metadata
    if "metadata" not in enhanced:
        enhanced["metadata"] = {}
    
    enhanced["metadata"]["module"] = "study_resources"
    enhanced["metadata"]["type"] = "resource_document"
    
    # Ensure keywords field exists
    if "keywords" not in enhanced:
        enhanced["keywords"] = []
    
    # Add relevant keywords based on course name
    course_name = enhanced.get("course_name", "").lower()
    if course_name:
        # Add the course name as a keyword
        if course_name not in enhanced["keywords"]:
            enhanced["keywords"].append(course_name)
        
        # Extract important terms from course name
        course_terms = course_name.split()
        for term in course_terms:
            if len(term) > 3 and term not in enhanced["keywords"]:  # Only add substantial terms
                enhanced["keywords"].append(term)
        
        # Add "study material" as a keyword
        if "study material" not in enhanced["keywords"]:
            enhanced["keywords"].append("study material")
            
        # Add specific course-related keywords
        if "computer" in course_name and "computer science" not in enhanced["keywords"]:
            enhanced["keywords"].append("computer science")
            
        if "data" in course_name and "data structures" not in enhanced["keywords"]:
            enhanced["keywords"].append("data structures")
            
        if "algorithm" in course_name and "algorithms" not in enhanced["keywords"]:
            enhanced["keywords"].append("algorithms")
            
        if "database" in course_name and "database" not in enhanced["keywords"]:
            enhanced["keywords"].append("database")
            
        if "web" in course_name and "web development" not in enhanced["keywords"]:
            enhanced["keywords"].append("web development")
            
        if "machine" in course_name and "machine learning" not in enhanced["keywords"]:
            enhanced["keywords"].append("machine learning")
            
        if "artificial" in course_name and "ai" not in enhanced["keywords"]:
            enhanced["keywords"].append("ai")
            
        if "cyber" in course_name and "cybersecurity" not in enhanced["keywords"]:
            enhanced["keywords"].append("cybersecurity")
    
    return enhanced

def generate_embeddings_and_upload(args, documents: List[Dict[str, Any]]):
    """
    Generate embeddings for resource documents and upload to Qdrant
    
    Args:
        args: Command line arguments
        documents: List of resource documents to embed
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
                    point_id = doc.get("id", f"resource-{total_points + j}")
                    
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
        
        logger.info(f"Successfully ingested {total_points} resource documents into Qdrant")
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")

def main():
    """Main process for ingesting study resource information"""
    logger.info("Starting study resources information ingestion...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load resource files
    documents = load_resource_files(args.data_dir)
    logger.info(f"Found {len(documents)} total resource documents/chunks")
    
    if not documents:
        logger.warning("No documents found. Please check the study resources data directory.")
        return
    
    # Generate embeddings and upload to Qdrant
    generate_embeddings_and_upload(args, documents)
    
    logger.info("Study resources information ingestion complete!")

if __name__ == "__main__":
    main()