import os
import argparse
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from utils.qdrant_helper import qdrant_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def search_collection(collection_name, query_text, limit=5):
    """
    Search for documents in a Qdrant collection based on a text query
    
    Args:
        collection_name: Name of the collection to search
        query_text: The query text
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents
    """
    try:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Generate embedding vector for the query
        query_vector = embeddings.embed_query(query_text)
        
        # Search in Qdrant
        search_results = qdrant_manager.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        
        # Format results for display
        formatted_results = []
        for result in search_results:
            # Extract payload
            payload = result.payload
            
            # Extract metadata if available
            metadata = payload.get("metadata", {})
            
            # Format the result
            formatted_result = {
                "score": round(result.score, 3),
                "text": payload.get("text", ""),
                "heading": metadata.get("heading", ""),
                "keywords": metadata.get("keywords", []),
                "module": metadata.get("module", ""),
                "id": result.id
            }
            
            formatted_results.append(formatted_result)
            
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error searching collection: {e}")
        return []

def display_results(results, query):
    """Display search results in a simple format"""
    print(f"\nSearch results for: '{query}'")
    print(f"Found {len(results)} results\n")
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results):
        print(f"--- Result {i+1} (Score: {result['score']}) ---")
        
        if result['heading']:
            print(f"Heading: {result['heading']}")
            
        # Display text (truncate if too long)
        text = result['text']
        if len(text) > 150:
            text = text[:150] + "..."
        print(f"Content: {text}")
        
        # Display keywords if available
        if result['keywords']:
            keywords_str = ", ".join(result['keywords'][:5])
            if len(result['keywords']) > 5:
                keywords_str += ", ..."
            print(f"Keywords: {keywords_str}")
            
        print()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple search utility for Qdrant collections")
    parser.add_argument("--collection", required=True, help="Collection to search")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()
    
    print(f"Searching for '{args.query}' in collection '{args.collection}'...")
    
    results = search_collection(
        collection_name=args.collection,
        query_text=args.query,
        limit=args.limit
    )
    
    display_results(results, args.query)

if __name__ == "__main__":
    main() 