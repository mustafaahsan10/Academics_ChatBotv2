import os
import json
import argparse
from dotenv import load_dotenv
import logging
from tabulate import tabulate  # You might need to install this: pip install tabulate

from utils.qdrant_helper import qdrant_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Inspect embeddings in Qdrant collections")
    parser.add_argument("--collection", default=None, help="Specific collection to inspect (leave empty to see all collections)")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to retrieve")
    parser.add_argument("--detailed", action="store_true", help="Show detailed vector information")
    return parser.parse_args()

def list_collections():
    """List all collections in Qdrant"""
    try:
        collections = qdrant_manager.client.get_collections().collections
        if not collections:
            logger.info("No collections found in Qdrant")
            return []
        
        collection_names = [collection.name for collection in collections]
        logger.info(f"Found {len(collection_names)} collections: {', '.join(collection_names)}")
        return collection_names
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return []

def get_collection_info(collection_name):
    """Get information about a specific collection"""
    try:
        collection_info = qdrant_manager.client.get_collection(collection_name)
        points_count = qdrant_manager.client.count(collection_name=collection_name).count
        
        # Get vector size and distance metric safely
        vector_size = collection_info.config.params.vectors.size
        vector_distance = collection_info.config.params.vectors.distance
        
        # Create info dictionary with safe defaults
        info = {
            "name": collection_name,
            "vector_size": vector_size,
            "vector_distance": vector_distance,
            "points_count": points_count,
            "indexed": True  # Default to True since most collections are indexed
        }
        
        return info
    except Exception as e:
        logger.error(f"Error getting collection info for {collection_name}: {e}")
        return None

def get_sample_points(collection_name, limit=5):
    """Get sample points from a collection"""
    try:
        # Search with an empty filter to get random points
        results = qdrant_manager.client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_vectors=True,
            with_payload=True
        )
        
        return results[0]  # First element contains the points
    except Exception as e:
        logger.error(f"Error getting sample points from {collection_name}: {e}")
        return []

def display_collection_info(collection_info):
    """Display collection information in a nice format"""
    if not collection_info:
        return
    
    print("\n=== Collection Information ===")
    info_table = [
        ["Name", collection_info["name"]],
        ["Vector Size", collection_info["vector_size"]],
        ["Distance Metric", collection_info["vector_distance"]],
        ["Total Points", collection_info["points_count"]]
    ]
    print(tabulate(info_table, tablefmt="grid"))

def display_sample_points(points, detailed=False):
    """Display sample points in a readable format"""
    if not points:
        print("No points found in collection")
        return
    
    print(f"\n=== Sample Points ({len(points)}) ===")
    
    for i, point in enumerate(points):
        print(f"\n--- Point {i+1} (ID: {point.id}) ---")
        
        # Display payload
        if point.payload:
            print("\nPayload:")
            # Extract text (typically the content)
            if "text" in point.payload:
                text = point.payload["text"]
                # Truncate if too long
                if len(text) > 100:
                    text = text[:100] + "..."
                print(f"Text: {text}")
                
            # Extract metadata
            if "metadata" in point.payload:
                print("\nMetadata:")
                metadata = point.payload["metadata"]
                for key, value in metadata.items():
                    # Format lists nicely
                    if isinstance(value, list) and len(value) > 3:
                        value = value[:3] + ["..."]
                    print(f"  {key}: {value}")
        
        # Display vector (only if detailed)
        if detailed and point.vector:
            print("\nVector (first 5 dimensions):")
            vector_sample = point.vector[:5]
            print(f"  {vector_sample}...")
            print(f"  Total dimensions: {len(point.vector)}")
        
        print("-" * 50)

def main():
    """Main function to inspect Qdrant collections"""
    args = parse_arguments()
    
    print("Connecting to Qdrant...")
    
    if args.collection:
        # Inspect a specific collection
        collection_info = get_collection_info(args.collection)
        if collection_info:
            display_collection_info(collection_info)
            
            # Get sample points
            sample_points = get_sample_points(args.collection, args.limit)
            display_sample_points(sample_points, args.detailed)
        else:
            logger.error(f"Collection '{args.collection}' not found or error occurred")
    else:
        # List all collections and their info
        collections = list_collections()
        
        if not collections:
            logger.info("No collections found in Qdrant")
            return
            
        print("\n=== All Collections ===")
        collection_table = []
        
        for collection_name in collections:
            info = get_collection_info(collection_name)
            if info:
                collection_table.append([
                    info["name"],
                    info["vector_size"],
                    info["vector_distance"],
                    info["points_count"]
                ])
        
        if collection_table:
            print(tabulate(collection_table, 
                          headers=["Collection", "Vector Size", "Distance", "Points Count"],
                          tablefmt="grid"))
            
            # Ask if user wants to inspect a specific collection
            print("\nTo inspect a specific collection, run with --collection COLLECTION_NAME")

if __name__ == "__main__":
    main() 