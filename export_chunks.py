import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime

# Load environment variables
load_dotenv()

def main():
    """Export all chunks and metadata from Qdrant vector database"""
    parser = argparse.ArgumentParser(description="Export chunks and metadata from Qdrant")
    parser.add_argument("--output", default="data/exported_chunks.json", help="Output JSON file path")
    parser.add_argument("--collection", default="university_data", help="Qdrant collection name")
    parser.add_argument("--limit", type=int, default=10000, help="Maximum number of chunks to export")
    args = parser.parse_args()
    
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
            print(f"Error: Collection '{args.collection}' not found!")
            print(f"Available collections: {', '.join(collection_names) if collection_names else 'none'}")
            return
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return
    
    # Retrieve all points with pagination
    print(f"Retrieving points from collection '{args.collection}'...")
    
    all_points = []
    offset = 0
    batch_size = 100
    
    while True:
        try:
            # Scroll through points in batches
            results = client.scroll(
                collection_name=args.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Skip the actual vectors to reduce size
            )
            
            points = results[0]
            if not points:
                break
                
            all_points.extend([{
                "id": point.id,
                "payload": point.payload
            } for point in points])
            
            offset += len(points)
            print(f"Retrieved {len(all_points)} points so far...")
            
            if len(all_points) >= args.limit:
                print(f"Reached limit of {args.limit} points.")
                all_points = all_points[:args.limit]
                break
                
        except Exception as e:
            print(f"Error retrieving points at offset {offset}: {e}")
            break
    
    print(f"Retrieved {len(all_points)} total points.")
    
    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate statistics
    source_types = Counter([point["payload"].get("type", "unknown") for point in all_points])
    sources = Counter([Path(point["payload"].get("source", "unknown")).name for point in all_points])
    
    # Create the output data structure
    output_data = {
        "metadata": {
            "export_date": datetime.now().isoformat(),
            "total_chunks": len(all_points),
            "collection_name": args.collection,
            "statistics": {
                "source_types": dict(source_types),
                "top_sources": dict(sources.most_common(10))
            }
        },
        "chunks": all_points
    }
    
    # Save to JSON file
    print(f"Saving data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Export complete! Saved {len(all_points)} chunks to {output_path}")
    print("\nStatistics:")
    print(f"  Document types:")
    for doc_type, count in source_types.most_common():
        print(f"    - {doc_type}: {count} chunks")
    
    print(f"\n  Top sources:")
    for source, count in sources.most_common(5):
        print(f"    - {source}: {count} chunks")

if __name__ == "__main__":
    main() 