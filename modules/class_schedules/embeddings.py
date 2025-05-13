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
COLLECTION_NAME = "class_schedules"
DATA_DIR = "data/schedules"
EMBEDDING_MODEL = "text-embedding-3-small"  # Using OpenAI's small embedding model
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ingest class schedule information into Qdrant vector database")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory containing schedule data files")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it exists")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="Embedding model to use")
    return parser.parse_args()

def load_schedule_files(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all class schedule files from the specified directory
    
    Args:
        data_dir: Directory containing schedule data files
        
    Returns:
        List of schedule document dictionaries
    """
    documents = []
    data_path = Path(data_dir)
    
    # Get all JSON files in the directory
    json_files = list(data_path.glob("**/*.json"))
    
    if not json_files:
        logger.warning(f"No schedule files found in {data_dir}")
        return []
    
    for file_path in json_files:
        try:
            logger.info(f"Processing: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse JSON content
            try:
                items = json.loads(content)
                
                # Handle different JSON structures
                if isinstance(items, list):
                    for item in items:
                        # Extract and enhance schedule information
                        enhanced_item = enhance_schedule_metadata(item)
                        documents.append(enhanced_item)
                elif isinstance(items, dict):
                    if "chunks" in items:
                        # Handle export format
                        for chunk in items["chunks"]:
                            enhanced_chunk = enhance_schedule_metadata(chunk)
                            documents.append(enhanced_chunk)
                    else:
                        enhanced_item = enhance_schedule_metadata(items)
                        documents.append(enhanced_item)
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON in {file_path}")
                continue
            
            logger.info(f"Added {len(documents)} schedule documents from {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return documents

def enhance_schedule_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance schedule metadata for better searchability
    
    Args:
        item: Original schedule item
        
    Returns:
        Enhanced schedule item with additional metadata
    """
    # Create a copy to avoid modifying the original
    enhanced = item.copy()
    
    # Add module identifier to the metadata
    if "metadata" not in enhanced:
        enhanced["metadata"] = {}
    
    enhanced["metadata"]["module"] = "class_schedules"
    enhanced["metadata"]["type"] = "schedule_document"
    
    # Add searchable day and time information from sessions if present
    if "sessions" in enhanced:
        days = []
        times = []
        
        for session in enhanced["sessions"]:
            if "day" in session and session["day"] not in days:
                days.append(session["day"])
            if "time_slot" in session:
                times.append(session["time_slot"])
        
        enhanced["metadata"]["days"] = days
        enhanced["metadata"]["times"] = times
    
    # Ensure keywords include schedule-related terms
    if "keywords" in enhanced:
        schedule_keywords = ["schedule", "timetable", "class time", "session"]
        for keyword in schedule_keywords:
            if keyword not in enhanced["keywords"]:
                enhanced["keywords"].append(keyword)
    else:
        enhanced["keywords"] = ["schedule", "timetable", "class time", "session"]
    
    return enhanced

def generate_embeddings_and_upload(args, documents: List[Dict[str, Any]]):
    """
    Generate embeddings for schedule documents and upload to Qdrant
    
    Args:
        args: Command line arguments
        documents: List of schedule documents to embed
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
                    point_id = doc.get("id", f"schedule-{total_points + j}")
                    
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
                            "module": "class_schedules",
                            "type": "schedule_document"
                        }
                    
                    # Add additional schedule-specific fields to payload for searching
                    payload = {
                        "text": doc.get("text", ""),
                        "metadata": metadata
                    }
                    
                    # Add additional schedule fields directly to payload for search
                    for field in ["course_code", "course_name", "program", "sessions"]:
                        if field in doc:
                            payload[field] = doc[field]
                    
                    # Create point with embedding and payload
                    points.append(
                        PointStruct(
                            id=numeric_id,
                            vector=embedding_vector,
                            payload=payload
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
        
        logger.info(f"Successfully ingested {total_points} schedule documents into Qdrant")
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")

def main():
    """Main process for ingesting class schedule information"""
    logger.info("Starting class schedule information ingestion...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load schedule files
    documents = load_schedule_files(args.data_dir)
    logger.info(f"Found {len(documents)} total schedule documents/chunks")
    
    if not documents:
        logger.warning("No documents found. Please check the schedule data directory.")
        return
    
    # Generate embeddings and upload to Qdrant
    generate_embeddings_and_upload(args, documents)
    
    logger.info("Class schedule information ingestion complete!")

if __name__ == "__main__":
    main()