import os
import json
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, IsNotEmpty
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
from tabulate import tabulate
import re

# Load environment variables
load_dotenv()

def main():
    """Test and compare search approaches in the Qdrant database"""
    parser = argparse.ArgumentParser(description="Test search approaches in Qdrant")
    parser.add_argument("--output", default="search_results.json", help="Output JSON file for results")
    parser.add_argument("--collection", default="university_data", help="Qdrant collection name")
    parser.add_argument("--query", help="Specific query to test")
    parser.add_argument("--queries", help="File with list of queries to test")
    args = parser.parse_args()
    
    if not args.query and not args.queries:
        print("Please provide either --query or --queries parameter")
        return
    
    # Connect to Qdrant
    print(f"Connecting to Qdrant...")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Get queries to test
    queries = []
    if args.query:
        queries.append(args.query)
    elif args.queries:
        try:
            with open(args.queries, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading queries file: {e}")
            return
    
    # Run tests on each query
    results = []
    for query in queries:
        print(f"\nTesting query: {query}")
        
        # Classify query type
        query_type = classify_query(query)
        print(f"Classified as: {query_type}")
        
        # Extract entities
        entities = extract_entities(query)
        print(f"Extracted entities: {entities}")
        
        # Run different search approaches
        query_embedding = embeddings.embed_query(query)
        
        # Approach 1: Standard search without filters
        standard_results = client.search(
            collection_name=args.collection,
            query_vector=query_embedding,
            limit=3
        )
        
        # Approach 2: Search with metadata type filter
        metadata_filter = create_metadata_filter(query_type)
        metadata_results = []
        if metadata_filter:
            metadata_results = client.search(
                collection_name=args.collection,
                query_vector=query_embedding,
                query_filter=metadata_filter,
                limit=3
            )
        
        # Approach 3: Search with entity filter
        entity_filter = create_entity_filter(entities)
        entity_results = []
        if entity_filter:
            entity_results = client.search(
                collection_name=args.collection,
                query_vector=query_embedding,
                query_filter=entity_filter,
                limit=3
            )
        
        # Approach 4: Combined metadata and entity filter
        combined_results = []
        if metadata_filter and entity_filter:
            combined_filter = Filter(
                must=[metadata_filter, entity_filter]
            )
            combined_results = client.search(
                collection_name=args.collection,
                query_vector=query_embedding,
                query_filter=combined_filter,
                limit=3
            )
        
        # Format results for comparison
        query_result = {
            "query": query,
            "query_type": query_type,
            "entities": entities,
            "results": {
                "standard": format_results(standard_results),
                "metadata_filtered": format_results(metadata_results),
                "entity_filtered": format_results(entity_results),
                "combined_filtered": format_results(combined_results)
            }
        }
        
        results.append(query_result)
        
        # Display results in tabular format
        display_result_comparison(query_result)
    
    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")

def classify_query(query):
    """Classify the query type"""
    classifiers = {
        "course": r"\b(course|class|syllabus|prerequisite|credit)\b",
        "exam": r"\b(exam|test|assessment|grade|score)\b",
        "faculty": r"\b(professor|instructor|faculty|teacher|lecturer|staff)\b",
        "schedule": r"\b(schedule|timetable|time|date|calendar|when)\b",
        "library": r"\b(library|book|resource|borrow|rent)\b"
    }
    
    query_type = "general"
    confidence = 0
    
    for category, pattern in classifiers.items():
        matches = re.findall(pattern, query.lower())
        if len(matches) > confidence:
            query_type = category
            confidence = len(matches)
    
    return query_type

def extract_entities(query):
    """Extract key entities from query"""
    entities = {}
    
    # Extract course codes (e.g., CSC 226)
    course_pattern = r'\b([A-Z]{2,4})\s?(\d{3}[A-Z]?)\b'
    course_matches = re.findall(course_pattern, query)
    if course_matches:
        entities["course_codes"] = course_matches
        
    # Extract names (simple heuristic, not ideal)
    name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
    name_matches = re.findall(name_pattern, query)
    if name_matches:
        entities["names"] = name_matches
        
    # Extract dates
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    date_matches = re.findall(date_pattern, query)
    if date_matches:
        entities["dates"] = date_matches
        
    return entities

def create_metadata_filter(query_type):
    """Create metadata filter based on query type"""
    if query_type == "general":
        return None
        
    if query_type == "course":
        return Filter(
            should=[
                FieldCondition(
                    key="metadata.type", 
                    match=MatchValue(value="course_catalog")
                ),
                FieldCondition(
                    key="metadata.type", 
                    match=MatchValue(value="syllabus")
                )
            ]
        )
        
    if query_type == "exam":
        return Filter(
            should=[
                FieldCondition(
                    key="metadata.type", 
                    match=MatchValue(value="exam_schedule")
                )
            ]
        )
        
    if query_type == "faculty":
        return Filter(
            should=[
                FieldCondition(
                    key="metadata.type", 
                    match=MatchValue(value="faculty_info")
                )
            ]
        )
        
    if query_type == "library":
        return Filter(
            should=[
                FieldCondition(
                    key="metadata.type", 
                    match=MatchValue(value="library_resource")
                )
            ]
        )
        
    if query_type == "schedule":
        return Filter(
            should=[
                FieldCondition(
                    key="metadata.entities.dates", 
                    match=IsNotEmpty()
                )
            ]
        )
        
    return None

def create_entity_filter(entities):
    """Create filter based on extracted entities"""
    if not entities:
        return None
    
    conditions = []
    
    if "course_codes" in entities and entities["course_codes"]:
        conditions.append(
            Filter(
                should=[
                    FieldCondition(
                        key="metadata.entities.course_codes", 
                        match=MatchAny(any=entities["course_codes"])
                    ),
                    FieldCondition(
                        key="metadata.chunk_entities.course_codes", 
                        match=MatchAny(any=entities["course_codes"])
                    )
                ]
            )
        )
    
    if "names" in entities and entities["names"]:
        conditions.append(
            Filter(
                should=[
                    FieldCondition(
                        key="metadata.entities.people", 
                        match=MatchAny(any=entities["names"])
                    ),
                    FieldCondition(
                        key="metadata.chunk_entities.people", 
                        match=MatchAny(any=entities["names"])
                    )
                ]
            )
        )
    
    if not conditions:
        return None
    
    if len(conditions) == 1:
        return conditions[0]
    
    return Filter(
        must=conditions
    )

def format_results(results):
    """Format search results for display and analysis"""
    formatted = []
    
    for hit in results:
        text = hit.payload.get("text", "")[:150] + "..." if len(hit.payload.get("text", "")) > 150 else hit.payload.get("text", "")
        metadata = hit.payload.get("metadata", {})
        
        source = metadata.get("source", "Unknown")
        if isinstance(source, str):
            source = os.path.basename(source)
        
        doc_type = metadata.get("type", "Unknown")
        doc_title = metadata.get("structure", {}).get("doc_title", "Unknown")
        
        formatted.append({
            "score": hit.score,
            "text_preview": text,
            "source": source,
            "type": doc_type,
            "title": doc_title
        })
    
    return formatted

def display_result_comparison(query_result):
    """Display a tabular comparison of search approaches"""
    query = query_result["query"]
    print(f"\nResults for query: '{query}'")
    print(f"Query type: {query_result['query_type']}")
    
    approaches = [
        ("Standard Search", query_result["results"]["standard"]),
        ("Metadata Filter", query_result["results"]["metadata_filtered"]),
        ("Entity Filter", query_result["results"]["entity_filtered"]),
        ("Combined Filter", query_result["results"]["combined_filtered"])
    ]
    
    for approach_name, results in approaches:
        if not results:
            print(f"\n{approach_name}: No results")
            continue
            
        print(f"\n{approach_name}:")
        
        table_data = []
        for i, result in enumerate(results):
            table_data.append([
                i+1,
                result["score"],
                result["type"],
                result["title"],
                result["text_preview"]
            ])
        
        headers = ["#", "Score", "Type", "Title", "Preview"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main() 