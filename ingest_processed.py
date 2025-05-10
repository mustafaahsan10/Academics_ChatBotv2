import os
import json
import glob
from pathlib import Path
import argparse
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load environment variables
load_dotenv()

# Configure parser
parser = argparse.ArgumentParser(description="Ingest processed university data into Qdrant vector database")
parser.add_argument("--data_dir", default="data/processed", help="Directory containing processed data files")
parser.add_argument("--collection", default="university_data", help="Qdrant collection name")
parser.add_argument("--recreate", action="store_true", help="Recreate the collection if it exists")
args = parser.parse_args()

def setup_qdrant():
    """Set up Qdrant collection with recreate option"""
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    # Delete collection if it exists and recreate flag is set
    if args.collection in collection_names and args.recreate:
        print(f"Recreating collection '{args.collection}'...")
        client.delete_collection(collection_name=args.collection)
        collection_exists = False
    else:
        collection_exists = args.collection in collection_names
    
    # Create collection if it doesn't exist
    if not collection_exists:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(
                size=1536,  # OpenAI embedding dimension
                distance=Distance.COSINE
            )
        )
        print(f"Created new '{args.collection}' collection")
    else:
        print(f"Collection '{args.collection}' already exists")
    
    return client

def load_processed_files(data_dir):
    """Load all processed JSON-formatted text files from the processed directory"""
    documents = []
    processed_dir = Path(data_dir)
    
    # Get all files in the processed directory
    text_files = list(processed_dir.glob("**/*.txt"))
    json_files = list(processed_dir.glob("**/*.json"))
    all_files = text_files + json_files
    
    if not all_files:
        print(f"No files found in {processed_dir}")
        return []
    
    for file_path in all_files:
        try:
            print(f"Processing: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Try to parse content as JSON
            try:
                items = json.loads(content)
                if isinstance(items, list):
                    documents.extend(items)
                elif isinstance(items, dict):
                    if "chunks" in items:
                        # Handle export format
                        documents.extend(items["chunks"])
                    else:
                        documents.append(items)
            except json.JSONDecodeError:
                # Handle plain text files with JSON-like structure
                # Split by JSON objects if possible
                parts = content.strip().split('\n\n')
                for part in parts:
                    try:
                        item = json.loads(part.strip())
                        documents.append(item)
                    except json.JSONDecodeError:
                        # Add as plain text with basic metadata
                        documents.append({
                            "heading": Path(file_path).stem,
                            "text": part,
                            "keywords": []
                        })
            
            print(f"Added {len(documents)} documents from {file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return documents

def generate_embeddings_and_upload(documents):
    """Generate embeddings for processed documents and upload to Qdrant"""
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize Qdrant client
    client = setup_qdrant()
    
    print(f"Generating embeddings for {len(documents)} documents...")
    
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
                # Create metadata structure
                metadata = {
                    "heading": doc.get("heading", ""),
                    "keywords": doc.get("keywords", []),
                    "doc_id": doc.get("id", total_points + j),
                    "source": "processed_data"
                }
                
                # Create point with embedding, payload with text and metadata
                points.append(
                    PointStruct(
                        id=total_points + j,
                        vector=embedding_vector,
                        payload={
                            "text": doc.get("text", ""),
                            "metadata": metadata
                        }
                    )
                )
            
            # Upload to Qdrant
            client.upsert(
                collection_name=args.collection,
                points=points
            )
            
            total_points += len(batch_docs)
            print(f"Processed {total_points}/{len(documents)} documents")
            
        except Exception as e:
            print(f"Error processing batch {i}-{end_idx}: {e}")
    
    print(f"Successfully ingested {total_points} documents into Qdrant")

def main():
    """Main process for ingesting processed data"""
    print("Starting processed data ingestion...")
    
    # Load processed files
    documents = load_processed_files(args.data_dir)
    print(f"Found {len(documents)} total documents/chunks")
    
    if not documents:
        print("No documents found. Please check the processed data directory.")
        return
    
    # Generate embeddings and upload to Qdrant
    generate_embeddings_and_upload(documents)
    
    print("Processed data ingestion complete!")

if __name__ == "__main__":
    main() 