import os
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

def main():
    """Delete a collection from Qdrant"""
    parser = argparse.ArgumentParser(description="Delete a collection from Qdrant")
    parser.add_argument("--collection", default="course_information", help="Qdrant collection name to delete")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    
    # Connect to Qdrant
    print(f"Connecting to Qdrant...")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # Check if collection exists
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if args.collection not in collection_names:
            print(f"Collection '{args.collection}' does not exist. Nothing to delete.")
            return
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return
    
    # Confirm deletion
    if not args.confirm:
        confirmation = input(f"Are you sure you want to delete collection '{args.collection}'? This cannot be undone. (y/n): ")
        if confirmation.lower() not in ["y", "yes"]:
            print("Deletion cancelled.")
            return
    
    # Delete collection
    try:
        client.delete_collection(collection_name=args.collection)
        print(f"Collection '{args.collection}' has been deleted.")
    except Exception as e:
        print(f"Error deleting collection: {e}")

if __name__ == "__main__":
    main() 