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
DATA_FILE = "data/processed/professor_data.json"  # Direct path to the professor data file
EMBEDDING_MODEL = "text-embedding-3-small"  # Using OpenAI's small embedding model
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ingest professor information into Qdrant vector database")
    parser.add_argument("--data_file", default=DATA_FILE, help="Path to the professor data file")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it exists")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="Embedding model to use")
    return parser.parse_args()

def load_professor_data(data_file: str) -> List[Dict[str, Any]]:
    """
    Load professor data from the specified file
    
    Args:
        data_file: Path to the professor data file
        
    Returns:
        List of professor document dictionaries
    """
    documents = []
    file_path = Path(data_file)
    
    if not file_path.exists():
        logger.warning(f"Professor data file not found at {file_path}")
        return []
    
    try:
        logger.info(f"Processing professor data from: {file_path}")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse JSON content
        try:
            professors = json.loads(content)
            
            # Handle different JSON structures
            if isinstance(professors, list):
                # Process each professor document
                for doc in professors:
                    # Ensure all necessary fields are present
                    enhanced_doc = prepare_professor_document(doc)
                    documents.append(enhanced_doc)
            elif isinstance(professors, dict):
                # Handle single document or nested structure
                if "professors" in professors and isinstance(professors["professors"], list):
                    # Handle a collection of professors
                    for doc in professors["professors"]:
                        enhanced_doc = prepare_professor_document(doc)
                        documents.append(enhanced_doc)
                else:
                    # Single professor document
                    enhanced_doc = prepare_professor_document(professors)
                    documents.append(enhanced_doc)
            
            logger.info(f"Loaded {len(documents)} professor documents")
            
        except json.JSONDecodeError:
            logger.error(f"Error parsing JSON in {file_path}")
            
    except Exception as e:
        logger.error(f"Error processing professor data: {e}")
    
    return documents

def prepare_professor_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a professor document for embedding by ensuring all required fields
    
    Args:
        doc: The original professor document
        
    Returns:
        Enhanced professor document
    """
    # Create a copy to avoid modifying the original
    prepared_doc = doc.copy()
    
    # Ensure all required fields exist
    required_fields = ["id", "heading", "text", "keywords"]
    for field in required_fields:
        if field not in prepared_doc:
            if field == "id":
                # Generate a unique ID if not present
                heading = prepared_doc.get("heading", "unknown-professor")
                prepared_doc["id"] = f"professor-{hash(heading)}"
            elif field == "keywords":
                prepared_doc["keywords"] = []
            else:
                prepared_doc[field] = ""
    
    # Add standard professor-related keywords if not present
    standard_keywords = ["professor", "faculty", "instructor", "academic", "contact"]
    for keyword in standard_keywords:
        if keyword not in prepared_doc["keywords"]:
            prepared_doc["keywords"].append(keyword)
    
    # Extract professor name from heading if possible
    if "heading" in prepared_doc and "professor_name" not in prepared_doc:
        heading = prepared_doc["heading"].lower()
        if "contact information" in heading:
            name_part = heading.split("contact information")[0].strip()
            prepared_doc["professor_name"] = name_part
        elif "biography" in heading and len(prepared_doc["text"]) > 20:
            # Try to extract the name from the first line of biography
            first_line = prepared_doc["text"].split("\n")[0]
            if "dr." in first_line.lower() or "professor" in first_line.lower():
                name_parts = first_line.split()
                if len(name_parts) >= 2:
                    prepared_doc["professor_name"] = " ".join(name_parts[:2])
    
    # Create metadata for search filtering
    prepared_doc["metadata"] = {
        "module": "professors",
        "type": "professor_document",
        "heading": prepared_doc.get("heading", ""),
        "document_source": prepared_doc.get("document_source", "")
    }
    
    # Enhance text for better searchability
    if "text" in prepared_doc and "keywords" in prepared_doc:
        # Add professor specializations or topics to help with searches
        specialization_words = [kw for kw in prepared_doc["keywords"] 
                              if kw not in standard_keywords and len(kw) > 3]
        if specialization_words and not any(word.lower() in prepared_doc["text"].lower() 
                                          for word in specialization_words):
            prepared_doc["text"] += f" Specializations and topics: {', '.join(specialization_words)}."
    
    return prepared_doc

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
            vector_size=EMBEDDING_DIMENSION,
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
                    
                    # Create payload with all relevant data for searching
                    payload = {
                        "id": doc.get("id", ""),
                        "heading": doc.get("heading", ""),
                        "text": doc.get("text", ""),
                        "keywords": doc.get("keywords", []),
                        "professor_name": doc.get("professor_name", ""),
                        "document_source": doc.get("document_source", ""),
                        "metadata": doc.get("metadata", {})
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
    
    # Load professor data
    documents = load_professor_data(args.data_file)
    logger.info(f"Found {len(documents)} total professor documents")
    
    if not documents:
        logger.warning("No documents found. Please check the professor data file path.")
        return
    
    # Generate embeddings and upload to Qdrant
    generate_embeddings_and_upload(args, documents)
    
    logger.info("Professor information ingestion complete!")

if __name__ == "__main__":
    main()