import os
import argparse
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from tabulate import tabulate

from utils.qdrant_helper import qdrant_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test search functionality on Qdrant embeddings")
    parser.add_argument("--collection", required=True, help="Collection to search in")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model to use")
    return parser.parse_args()

def search_embeddings(collection_name, query_text, limit=5, model="text-embedding-3-small"):
    """
    Search for documents in the collection using a query
    
    Args:
        collection_name: Name of the Qdrant collection
        query_text: Search query text
        limit: Number of results to return
        model: Embedding model to use
        
    Returns:
        List of search results
    """
    try:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        logger.info(f"Generating embedding for query: '{query_text}'")
        
        # Generate embedding for the query
        query_embedding = embeddings.embed_query(query_text)
        
        # Search in Qdrant
        search_results = qdrant_manager.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )
        
        return search_results
    
    except Exception as e:
        logger.error(f"Error searching embeddings: {e}")
        return []

def display_search_results(results, query):
    """Display search results in a readable format"""
    if not results:
        print(f"No results found for query: '{query}'")
        return
    
    print(f"\n=== Search Results for: '{query}' ===")
    print(f"Found {len(results)} results\n")
    
    result_table = []
    
    for i, result in enumerate(results):
        # Extract necessary info
        score = round(result.score, 3)
        text = result.payload.get("text", "")
        
        # Truncate text if too long
        if len(text) > 100:
            text = text[:100] + "..."
            
        # Extract metadata
        metadata = result.payload.get("metadata", {})
        metadata_str = ""
        
        if "heading" in metadata:
            metadata_str += f"Heading: {metadata['heading']}\n"
            
        if "module" in metadata:
            metadata_str += f"Module: {metadata['module']}\n"
            
        if "keywords" in metadata and metadata["keywords"]:
            keywords = metadata["keywords"]
            if len(keywords) > 3:
                keyword_str = ", ".join(keywords[:3]) + "..."
            else:
                keyword_str = ", ".join(keywords)
            metadata_str += f"Keywords: {keyword_str}"
        
        result_table.append([i+1, score, text, metadata_str])
    
    # Display table
    print(tabulate(result_table, 
                  headers=["#", "Score", "Text", "Metadata"],
                  tablefmt="grid"))

def main():
    """Main function to test search functionality"""
    args = parse_arguments()
    
    print(f"Searching for '{args.query}' in collection '{args.collection}'...")
    
    # Search for documents
    results = search_embeddings(
        collection_name=args.collection,
        query_text=args.query,
        limit=args.limit,
        model=args.model
    )
    
    # Display results
    display_search_results(results, args.query)

if __name__ == "__main__":
    main() 