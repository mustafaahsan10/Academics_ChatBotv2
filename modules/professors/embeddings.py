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
DATA_FILE = "data/processed/professor_data.json"  # Direct file path
EMBEDDING_MODEL = "text-embedding-3-small"  # Using OpenAI's small embedding model
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ingest professor information into Qdrant vector database")
    parser.add_argument("--data_path", default=DATA_FILE, help="Path to the data file or directory")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it exists")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="Embedding model to use")
    return parser.parse_args()

def load_professor_files(data_path: str) -> List[Dict[str, Any]]:
    """
    Load professor information from the specified file or directory
    
    Args:
        data_path: Path to the data file or directory
        
    Returns:
        List of professor document dictionaries
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
        logger.warning(f"No professor files found at {data_path}")
        return []
    
    for file_path in file_paths:
        try:
            logger.info(f"Processing: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse JSON content
            try:
                items = json.loads(content)
                
                # Handle different JSON structures - professor data is typically a list
                if isinstance(items, list):
                    # Process and add each professor entry
                    for item in items:
                        # Add a module identifier to the metadata
                        if "metadata" not in item:
                            item["metadata"] = {"module": "professors"}
                        else:
                            item["metadata"]["module"] = "professors"
                        documents.append(item)
                elif isinstance(items, dict):
                    # Single document
                    if "metadata" not in items:
                        items["metadata"] = {"module": "professors"}
                    else:
                        items["metadata"]["module"] = "professors"
                    documents.append(items)
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON in {file_path}")
                continue
            
            logger.info(f"Added {len(documents)} professor documents from {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return documents

def prepare_professor_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare and enhance the professor document for embedding
    
    Args:
        doc: The original professor document
        
    Returns:
        Enhanced professor document for better search
    """
    enhanced_doc = doc.copy()
    
    # Create a concatenated text field if it doesn't exist or is empty
    if "main_text" not in enhanced_doc or not enhanced_doc["main_text"]:
        main_text_parts = []
        
        # Add professor name and title
        if enhanced_doc.get("professor_name"):
            main_text_parts.append(f"Professor: {enhanced_doc['professor_name']}")
        
        if enhanced_doc.get("professor_title"):
            main_text_parts.append(f"Title: {enhanced_doc['professor_title']}")
        
        # Add contact info
        if enhanced_doc.get("email"):
            main_text_parts.append(f"Email: {enhanced_doc['email']}")
        
        if enhanced_doc.get("phone"):
            main_text_parts.append(f"Phone: {enhanced_doc['phone']}")
        
        if enhanced_doc.get("office_location"):
            main_text_parts.append(f"Office Location: {enhanced_doc['office_location']}")
        
        # Add university and department
        if enhanced_doc.get("university"):
            main_text_parts.append(f"University: {enhanced_doc['university']}")
        
        if enhanced_doc.get("faculty_department_institute"):
            main_text_parts.append(f"Department: {enhanced_doc['faculty_department_institute']}")
        
        # Add work title for publications
        if enhanced_doc.get("work_title"):
            main_text_parts.append(f"Work: {enhanced_doc['work_title']}")
        
        # Join everything
        enhanced_doc["main_text"] = " ".join(main_text_parts)
    
    # Create a heading field if it doesn't exist
    if "heading" not in enhanced_doc or not enhanced_doc["heading"]:
        if enhanced_doc.get("professor_name"):
            enhanced_doc["heading"] = enhanced_doc["professor_name"]
            if enhanced_doc.get("professor_title"):
                enhanced_doc["heading"] += f", {enhanced_doc['professor_title']}"
        elif enhanced_doc.get("work_title"):
            enhanced_doc["heading"] = enhanced_doc["work_title"]
    
    # Ensure the text field exists for embedding
    if "text" not in enhanced_doc or not enhanced_doc["text"]:
        enhanced_doc["text"] = enhanced_doc.get("main_text", "")
    
    # Make sure keywords field exists
    if "keywords" not in enhanced_doc:
        enhanced_doc["keywords"] = []
    
    # Add important keywords
    keywords = set(enhanced_doc["keywords"])
    
    # Add professor name to keywords
    if enhanced_doc.get("professor_name") and enhanced_doc["professor_name"] not in keywords:
        keywords.add(enhanced_doc["professor_name"])
    
    # Add chunk type as keyword
    if enhanced_doc.get("chunk_type") and enhanced_doc["chunk_type"] not in keywords:
        keywords.add(enhanced_doc["chunk_type"])
    
    # Add standard professor keywords if they don't exist
    standard_keywords = ["professor", "faculty", "instructor", "teacher", "academic", "contact", "office"]
    for keyword in standard_keywords:
        if keyword not in keywords:
            keywords.add(keyword)
    
    enhanced_doc["keywords"] = list(keywords)
    
    return enhanced_doc

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
            
            # Prepare documents for embedding
            prepared_docs = [prepare_professor_document(doc) for doc in batch_docs]
            
            try:
                # Extract text content for embeddings
                texts = [doc.get("text", "") or doc.get("main_text", "") for doc in prepared_docs]
                
                # Generate embeddings
                batch_embeddings = embeddings.embed_documents(texts)
                
                # Prepare points
                points = []
                for j, (doc, embedding_vector) in enumerate(zip(prepared_docs, batch_embeddings)):
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
                    
                    # Create enhanced metadata structure for search filtering
                    metadata = {
                        "module": "professors",
                        "type": doc.get("chunk_type", "professor_document"),
                        "professor_name": doc.get("professor_name", ""),
                        "heading": doc.get("heading", ""),
                        "keywords": doc.get("keywords", [])
                    }
                    
                    # Create payload with all relevant fields
                    payload = {
                        "id": doc.get("id", ""),
                        "heading": doc.get("heading", ""),
                        "text": doc.get("text", "") or doc.get("main_text", ""),
                        "professor_name": doc.get("professor_name", ""),
                        "professor_title": doc.get("professor_title", ""),
                        "university": doc.get("university", ""),
                        "faculty_department_institute": doc.get("faculty_department_institute", ""),
                        "office_location": doc.get("office_location", ""),
                        "phone": doc.get("phone", ""),
                        "email": doc.get("email", ""),
                        "work_title": doc.get("work_title", ""),
                        "keywords": doc.get("keywords", []),
                        "chunk_type": doc.get("chunk_type", ""),
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
        
        logger.info(f"Successfully ingested {total_points} professor documents into Qdrant")
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")

def main():
    """Main process for ingesting professor information"""
    logger.info("Starting professor information ingestion...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load professor files - now works with either a file or directory
    documents = load_professor_files(args.data_path)
    logger.info(f"Found {len(documents)} total professor documents")
    
    if not documents:
        logger.warning("No documents found. Please check the file path or directory.")
        return
    
    # Generate embeddings and upload to Qdrant
    generate_embeddings_and_upload(args, documents)
    
    logger.info("Professor information ingestion complete!")

if __name__ == "__main__":
    main()