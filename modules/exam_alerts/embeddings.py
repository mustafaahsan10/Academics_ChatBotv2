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
COLLECTION_NAME = "exam_alerts"
DATA_DIR = "data/processed/exam_data.json"
EMBEDDING_MODEL = "text-embedding-3-small"  # Using OpenAI's small embedding model
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ingest exam information into Qdrant vector database")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Directory containing exam data files")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it exists")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="Embedding model to use")
    return parser.parse_args()

def load_exam_files(data_path: str) -> List[Dict[str, Any]]:
    """
    Load exam information files from the specified path
    
    Args:
        data_path: Path to the exam data file
        
    Returns:
        List of exam document dictionaries
    """
    documents = []
    path = Path(data_path)
    
    if not path.exists():
        logger.warning(f"Exam data file not found at {path}")
        return []
    
    try:
        logger.info(f"Processing: {path}")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            exams = json.load(f)
        
        # Process each exam as a separate document
        for i, exam in enumerate(exams):
            # Normalize data structure
            normalized_exam = {
                "id": exam.get("id", f"exam-{i}"),
                "course_code": exam.get("Course Code", ""),
                "course_name": exam.get("Course Name", ""),
                "date": exam.get("Exam Date", ""),
                "time": exam.get("Exam Time", ""),
                "location": exam.get("Location", "Not specified"),
                "type": exam.get("Type", "Final Exam"),
                "text": exam.get("Text", "")
            }
            
            # Ensure text field exists for embedding
            if not normalized_exam["text"]:
                normalized_exam["text"] = f"Course: {normalized_exam['course_code']} {normalized_exam['course_name']}. Date: {normalized_exam['date']}. Time: {normalized_exam['time']}. Location: {normalized_exam['location']}. Type: {normalized_exam['type']}."
            
            # Enhance metadata
            enhanced_exam = enhance_exam_metadata(normalized_exam)
            documents.append(enhanced_exam)
        
        logger.info(f"Added {len(documents)} exam documents")
    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
    
    return documents

def enhance_exam_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance exam metadata for better searchability
    
    Args:
        item: Original exam item
        
    Returns:
        Enhanced exam item with additional metadata
    """
    # Create a copy to avoid modifying the original
    enhanced = item.copy()
    
    # Add module identifier to the metadata
    if "metadata" not in enhanced:
        enhanced["metadata"] = {}
    
    enhanced["metadata"]["module"] = "exam_alerts"
    enhanced["metadata"]["type"] = "exam_document"
    
    # Create or update keywords list
    if "keywords" not in enhanced:
        enhanced["keywords"] = []
    
    # Add course code and name to keywords
    course_code = enhanced.get("course_code")
    course_name = enhanced.get("course_name")
    
    if course_code and course_code not in enhanced["keywords"]:
        enhanced["keywords"].append(course_code)
    
    if course_name and course_name not in enhanced["keywords"]:
        enhanced["keywords"].append(course_name)
    
    # Add exam type to keywords
    exam_type = enhanced.get("type", "").lower()
    if exam_type and exam_type not in enhanced["keywords"]:
        enhanced["keywords"].append(exam_type)
    
    # Add standardized keywords based on exam type
    if "final" in exam_type and "final exam" not in enhanced["keywords"]:
        enhanced["keywords"].append("final exam")
    
    if "midterm" in exam_type and "midterm exam" not in enhanced["keywords"]:
        enhanced["keywords"].append("midterm exam")
    
    if "quiz" in exam_type and "quiz" not in enhanced["keywords"]:
        enhanced["keywords"].append("quiz")
    
    # Add general exam-related keywords
    standard_keywords = ["exam", "assessment", "test", "schedule"]
    for keyword in standard_keywords:
        if keyword not in enhanced["keywords"]:
            enhanced["keywords"].append(keyword)
    
    # Add date as a keyword for filtering by date
    if enhanced.get("date") and enhanced.get("date") not in enhanced["keywords"]:
        enhanced["keywords"].append(enhanced.get("date"))
    
    return enhanced

def generate_embeddings_and_upload(args, documents: List[Dict[str, Any]]):
    """
    Generate embeddings for exam documents and upload to Qdrant
    
    Args:
        args: Command line arguments
        documents: List of exam documents to embed
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
                    point_id = doc.get("id", f"exam-{total_points + j}")
                    
                    # Ensure point_id is a string or integer
                    if isinstance(point_id, str) and not point_id.isdigit():
                        numeric_id = total_points + j
                    else:
                        try:
                            numeric_id = int(point_id)
                        except ValueError:
                            numeric_id = total_points + j
                    
                    # Create point with embedding and payload
                    points.append(
                        PointStruct(
                            id=numeric_id,
                            vector=embedding_vector,
                            payload={
                                "id": doc.get("id", ""),
                                "course_code": doc.get("course_code", ""),
                                "course_name": doc.get("course_name", ""),
                                "date": doc.get("date", ""),
                                "time": doc.get("time", ""),
                                "location": doc.get("location", ""),
                                "type": doc.get("type", ""),
                                "text": doc.get("text", ""),
                                "keywords": doc.get("keywords", []),
                                "metadata": doc.get("metadata", {})
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
        
        logger.info(f"Successfully ingested {total_points} exam documents into Qdrant")
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")

def main():
    """Main process for ingesting exam information"""
    logger.info("Starting exam information ingestion...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load exam files
    documents = load_exam_files(args.data_dir)
    logger.info(f"Found {len(documents)} total exam documents/chunks")
    
    if not documents:
        logger.warning("No documents found. Please check the exam data directory.")
        return
    
    # Generate embeddings and upload to Qdrant
    generate_embeddings_and_upload(args, documents)
    
    logger.info("Exam information ingestion complete!")

if __name__ == "__main__":
    main()