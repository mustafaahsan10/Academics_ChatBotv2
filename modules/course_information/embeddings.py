import os
import json
import glob
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
COLLECTION_NAME = "course_information"
# You can change this to directly point to your specific file:
DATA_FILE = "data/processed/admission_guide_PDF_extracted_text.json"  # Direct file path
EMBEDDING_MODEL = "text-embedding-3-small"  # Using OpenAI's small embedding model
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ingest course information into Qdrant vector database")
    parser.add_argument("--data_path", default=DATA_FILE, help="Path to the data file or directory")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it exists")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="Embedding model to use")
    return parser.parse_args()

def load_course_files(data_path: str) -> List[Dict[str, Any]]:
    """
    Load course information from the specified file or directory
    
    Args:
        data_path: Path to the data file or directory
        
    Returns:
        List of course document dictionaries
    """
    documents = []
    path = Path(data_path)
    
    # Check if path is a file or directory
    if path.is_file():
        # Process a single file
        file_paths = [path]
    else:
        # Get all JSON files in the directory
        file_paths = list(path.glob("**/*.json"))
    
    if not file_paths:
        logger.warning(f"No course files found at {data_path}")
        return []
    
    for file_path in file_paths:
        try:
            logger.info(f"Processing: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse JSON content
            try:
                items = json.loads(content)
                logger.info(f"Loaded JSON with {len(items)} items from {file_path}")
                
                # Diagnostic logs
                if isinstance(items, list):
                    logger.info(f"File contains a list with {len(items)} items")
                    logger.info(f"First item ID: {items[0].get('id', 'No ID')}")
                    logger.info(f"Last item ID: {items[-1].get('id', 'No ID')}")
                
                # Handle different JSON structures
                if isinstance(items, list):
                    for item in items:
                        # Add a module identifier to the metadata
                        if "metadata" in item:
                            item["metadata"]["module"] = "course_information"
                        else:
                            item["metadata"] = {"module": "course_information"}
                        documents.append(item)
                elif isinstance(items, dict):
                    if "chunks" in items:
                        # Handle export format
                        for chunk in items["chunks"]:
                            if "metadata" in chunk:
                                chunk["metadata"]["module"] = "course_information"
                            else:
                                chunk["metadata"] = {"module": "course_information"}
                            documents.append(chunk)
                    else:
                        # Single document
                        if "metadata" in items:
                            items["metadata"]["module"] = "course_information"
                        else:
                            items["metadata"] = {"module": "course_information"}
                        documents.append(items)
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON in {file_path}")
                continue
            
            logger.info(f"Added {len(documents)} documents from {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

def generate_embeddings_and_upload(args, documents: List[Dict[str, Any]]):
    """
    Generate embeddings for course documents and upload to Qdrant
    
    Args:
        args: Command line arguments
        documents: List of course documents to embed
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
                    point_id = doc.get("id", f"course-{total_points + j}")
                    
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
                    if not metadata:
                        metadata = {
                            "heading": doc.get("heading", ""),
                            "keywords": doc.get("keywords", []),
                            "module": "course_information",
                            "type": "course_document"
                        }
                        
                        # Add course details if available
                        course_details = doc.get("course_details")
                        if course_details:
                            metadata["course_details"] = course_details
                            
                        # Add prerequisite details if available
                        prerequisite_details = doc.get("prerequisite_details")
                        if prerequisite_details:
                            metadata["prerequisite_details"] = prerequisite_details
                    
                    # Create point with embedding, payload with text and metadata
                    points.append(
                        PointStruct(
                            id=numeric_id,
                            vector=embedding_vector,
                            payload={
                                "text": doc.get("text", ""),
                                "metadata": metadata
                            }
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
        
        logger.info(f"Successfully ingested {total_points} course documents into Qdrant")
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")

def main():
    """Main process for ingesting course information"""
    logger.info("Starting course information ingestion...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load course files - now works with either a file or directory
    documents = load_course_files(args.data_path)
    logger.info(f"Found {len(documents)} total course documents/chunks")
    
    if not documents:
        logger.warning("No documents found. Please check the file path or directory.")
        return
    
    # Generate embeddings and upload to Qdrant
    generate_embeddings_and_upload(args, documents)
    
    logger.info("Course information ingestion complete!")

if __name__ == "__main__":
    main()