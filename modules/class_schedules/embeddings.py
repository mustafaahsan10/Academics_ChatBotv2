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
DATA_DIR = "data/processed/Class_Schedule.json"
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

def load_schedule_files(data_path: str) -> List[Dict[str, Any]]:
    """
    Load class schedule data from the specified path
    
    Args:
        data_path: Path to the schedule data file or directory
        
    Returns:
        List of schedule document dictionaries
    """
    documents = []
    path = Path(data_path)
    
    # Check if the path is a file or directory
    if path.is_file():
        # Direct file path provided
        try:
            logger.info(f"Processing file: {path}")
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse JSON content
            try:
                items = json.loads(content)
                
                # Handle different JSON structures
                if isinstance(items, list):
                    documents.extend(items)
                    logger.info(f"Added {len(items)} schedule documents from {path}")
                elif isinstance(items, dict):
                    if "schedules" in items and isinstance(items["schedules"], list):
                        # Handle nested structure
                        documents.extend(items["schedules"])
                        logger.info(f"Added {len(items['schedules'])} schedule documents from {path}")
                    else:
                        # Single document
                        documents.append(items)
                        logger.info(f"Added 1 schedule document from {path}")
                    
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON in {path}")
                
        except Exception as e:
            logger.error(f"Error processing file {path}: {e}")
    
    else:
        # Directory path provided
        json_files = [file for file in path.glob("**/*.json") if "Class_Schedule" in file.name]
        
        if not json_files:
            logger.warning(f"No schedule files found in {data_path}")
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
                            # Ensure the document structure matches what we expect
                            if all(field in item for field in ["id", "heading", "course_code", "course_name", "sessions", "text", "keywords"]):
                                documents.append(item)
                            else:
                                logger.warning(f"Skipping item with incomplete fields: {item.get('id', 'unknown')}")
                    elif isinstance(items, dict):
                        if "schedules" in items and isinstance(items["schedules"], list):
                            # Handle nested structure
                            for schedule in items["schedules"]:
                                documents.append(schedule)
                        else:
                            # Single document
                            documents.append(items)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON in {file_path}")
                    continue
                
                logger.info(f"Added {len(documents)} schedule documents from {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    
    return documents

def prepare_schedule_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a schedule document for embedding by ensuring all required fields are present
    and normalized
    
    Args:
        doc: The original schedule document
        
    Returns:
        Enhanced and normalized schedule document
    """
    # Create a copy to avoid modifying the original
    prepared_doc = doc.copy()
    
    # Ensure required fields exist
    required_fields = ["id", "heading", "text", "course_code", "course_name", "sessions", "keywords"]
    for field in required_fields:
        if field not in prepared_doc:
            if field == "id":
                prepared_doc["id"] = f"schedule-{hash(prepared_doc.get('heading', '') + prepared_doc.get('course_code', ''))}"
            elif field == "keywords":
                prepared_doc["keywords"] = []
            elif field == "sessions":
                prepared_doc["sessions"] = []
            else:
                prepared_doc[field] = ""
    
    # Ensure sessions has the correct structure
    for i, session in enumerate(prepared_doc["sessions"]):
        if not isinstance(session, dict):
            prepared_doc["sessions"][i] = {"day": "", "time_slot": ""}
        else:
            if "day" not in session:
                session["day"] = ""
            if "time_slot" not in session:
                session["time_slot"] = ""
    
    # Extract day and time information for searchable text
    days_info = []
    times_info = []
    for session in prepared_doc["sessions"]:
        if session["day"] and session["day"] not in days_info:
            days_info.append(session["day"])
        if session["time_slot"] and session["time_slot"] not in times_info:
            times_info.append(session["time_slot"])
    
    # Enhance the text field with session information for better searching
    if prepared_doc["text"] and not any(day in prepared_doc["text"] for day in days_info):
        sessions_text = ", ".join([f"{s['day']} at {s['time_slot']}" for s in prepared_doc["sessions"] if s["day"] and s["time_slot"]])
        if sessions_text:
            prepared_doc["text"] += f" Sessions: {sessions_text}."
    
    # Enhance keywords
    existing_keywords = set(prepared_doc["keywords"])
    
    # Add course code and name to keywords
    if prepared_doc["course_code"] and prepared_doc["course_code"] not in existing_keywords:
        prepared_doc["keywords"].append(prepared_doc["course_code"])
    
    if prepared_doc["course_name"] and prepared_doc["course_name"] not in existing_keywords:
        prepared_doc["keywords"].append(prepared_doc["course_name"])
    
    # Add days and times to keywords
    for day in days_info:
        if day and day not in existing_keywords:
            prepared_doc["keywords"].append(day)
    
    for time in times_info:
        if time and time not in existing_keywords:
            prepared_doc["keywords"].append(time)
    
    # Add standard schedule keywords if they don't exist
    standard_keywords = ["schedule", "timetable", "class time", "session"]
    for keyword in standard_keywords:
        if keyword not in existing_keywords:
            prepared_doc["keywords"].append(keyword)
    
    return prepared_doc

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
            
            # Prepare documents
            prepared_docs = [prepare_schedule_document(doc) for doc in batch_docs]
            
            try:
                # Extract text content for embeddings
                texts = [doc.get("text", "") for doc in prepared_docs]
                
                # Generate embeddings
                batch_embeddings = embeddings.embed_documents(texts)
                
                # Prepare points
                points = []
                for j, (doc, embedding_vector) in enumerate(zip(prepared_docs, batch_embeddings)):
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
                    
                    # Create metadata structure for search filtering
                    metadata = {
                        "module": "class_schedules",
                        "type": "schedule_document",
                        "heading": doc.get("heading", ""),
                        "days": [session.get("day", "") for session in doc.get("sessions", []) if session.get("day")],
                        "times": [session.get("time_slot", "") for session in doc.get("sessions", []) if session.get("time_slot")]
                    }
                    
                    # Create payload for storage and search
                    payload = {
                        "id": doc.get("id", ""),
                        "heading": doc.get("heading", ""),
                        "text": doc.get("text", ""),
                        "course_code": doc.get("course_code", ""),
                        "course_name": doc.get("course_name", ""),
                        "program": doc.get("program", ""),
                        "sessions": doc.get("sessions", []),
                        "keywords": doc.get("keywords", []),
                        "metadata": metadata
                    }
                    
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
    logger.info(f"Found {len(documents)} total schedule documents")
    
    if not documents:
        logger.warning("No documents found. Please check the schedule data directory.")
        return
    
    # Generate embeddings and upload to Qdrant
    generate_embeddings_and_upload(args, documents)
    
    logger.info("Class schedule information ingestion complete!")

if __name__ == "__main__":
    main()