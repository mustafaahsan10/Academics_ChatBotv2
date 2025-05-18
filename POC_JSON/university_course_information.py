import json
import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
from nltk.corpus import stopwords
import time
import PyPDF2
import re

# Set up OpenRouter with OpenAI client
OPENROUTER_API_KEY = "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"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

OPENAI_API_KEY = "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"
embeddings_client = OpenAI(api_key=OPENAI_API_KEY)

# Connect to Qdrant (local or cloud)
qdrant_client = QdrantClient(
    url="https://8b6857da-0682-417b-a31b-2a83bef2cab3.us-east-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.dWjs7ZnPcyo0lbk1tvelYBim14HKNwDm1qfWTKaoVoQ"
)

"""## PDF Text Extraction"""

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF and split into chunks with metadata."""
    print(f"Processing PDF file: {pdf_path}")

    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        print(f"PDF has {num_pages} pages")

        # Extract text from each page
        all_text = ""
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            all_text += text + "\n\n"

    # Split the text into chunks of reasonable size (~1000 characters)
    chunks = []

    # Split by sections/paragraphs (adjust as needed based on PDF structure)
    raw_chunks = re.split(r'\n\s*\n', all_text)

    current_chunk = ""
    for chunk in raw_chunks:
        if not chunk.strip():
            continue

        if len(current_chunk) + len(chunk) < 1000:
            current_chunk += chunk + "\n\n"
        else:
            if current_chunk:
                # Extract potential keywords (simple heuristic: capitalized words)
                keywords = re.findall(r'\b[A-Z][A-Za-z]{2,}\b', current_chunk)
                keywords = list(set([k for k in keywords if k.lower() not in stopwords.words('english')]))[:10]

                chunks.append({
                    "text": current_chunk.strip(),
                    "keywords": keywords
                })
            current_chunk = chunk + "\n\n"

    # Add the last chunk if not empty
    if current_chunk.strip():
        keywords = re.findall(r'\b[A-Z][A-Za-z]{2,}\b', current_chunk)
        keywords = list(set([k for k in keywords if k.lower() not in stopwords.words('english')]))[:10]

        chunks.append({
            "text": current_chunk.strip(),
            "keywords": keywords
        })

    print(f"Split PDF into {len(chunks)} chunks")
    return chunks

"""## File Handling and Embedding Generation"""

def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Read JSON file and return its contents."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a text using OpenAI's text-embeddings-small-3 model."""
    start_time = time.time()
    response = embeddings_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding_time = time.time() - start_time
    print(f"Embedding time: {embedding_time} seconds")
    return response.data[0].embedding

def create_payload(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Create a payload with text, keywords, and metadata for Qdrant."""
    text = entry.get("text", "")
    keywords = entry.get("keywords", [])

    return {
        "text": text,
        "keywords": keywords,
    }

"""## Qdrant Operations"""

def create_collection(collection_name: str, vector_size: int = 1536):
    """Create a collection in Qdrant if it doesn't exist."""
    try:
        qdrant_client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists")
    except Exception:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection {collection_name}")

def process_and_upload_data(data: List[Dict[str, Any]], collection_name: str):
    """Process each entry, generate embedding, and upload to Qdrant."""
    batch_size = 10  # Process in batches to avoid API rate limits

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        points = []
        for j, entry in enumerate(batch):
            # Create payload with text and keywords
            payload = create_payload(entry)

            # Generate embedding for text content
            embedding = generate_embedding(entry["text"])

            # Add to points
            points.append(models.PointStruct(
                id=i+j,
                vector=embedding,
                payload=payload
            ))

        # Upload batch to Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

        print(f"Uploaded batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")

"""## PDF Processing and Uploading"""

def process_pdf_and_upload(pdf_path: str, collection_name: str = "admission_course_guide"):
    """Process a PDF file and upload its embeddings to Qdrant."""
    # Extract text from PDF
    pdf_data = extract_text_from_pdf(pdf_path)

    # Create collection
    create_collection(collection_name)

    # Process and upload data
    process_and_upload_data(pdf_data, collection_name)

    print(f"PDF {pdf_path} processed and uploaded to Qdrant collection {collection_name}")
    return pdf_data

"""## Append PDF to Collection"""

def append_pdf_to_collection(pdf_path: str, collection_name: str = "admission_course_guide"):
    """Process a PDF file and append its embeddings to an existing Qdrant collection."""
    # Extract text from PDF
    pdf_data = extract_text_from_pdf(pdf_path)

    # Verify collection exists
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        print(f"Found existing collection {collection_name}")
    except Exception:
        print(f"Collection {collection_name} does not exist, creating it...")
        create_collection(collection_name)

    # Get the count of existing points to avoid ID conflicts
    collection_info = qdrant_client.get_collection(collection_name)
    existing_points = 12
    print(f"Collection has {existing_points} existing points")

    # Process in batches to avoid API rate limits
    batch_size = 10
    for i in range(0, len(pdf_data), batch_size):
        batch = pdf_data[i:i+batch_size]

        points = []
        for j, entry in enumerate(batch):
            # Create payload with text and keywords
            payload = create_payload(entry)

            # Generate embedding for text content
            embedding = generate_embedding(entry["text"])

            # Add to points with offset IDs to avoid conflicts
            points.append(models.PointStruct(
                id=existing_points + i + j,
                vector=embedding,
                payload=payload
            ))

        # Upload batch to Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

        print(f"Uploaded batch {i//batch_size + 1}/{(len(pdf_data) + batch_size - 1)//batch_size}")

    print(f"PDF {pdf_path} processed and appended to Qdrant collection {collection_name}")
    return pdf_data

"""## Search Function"""

def search_qdrant_simple(query: str, collection_name: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Perform simple search in Qdrant for a single query."""
    # Generate embedding for the query
    embedding = generate_embedding(query)

    start_time = time.time()
    # Perform search
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=embedding,
        limit=limit,
        with_payload=True,
        score_threshold=0.4
    )
    print(search_results)
    search_time = time.time() - start_time
    print(f"Search time: {search_time} seconds")

    start_time_1 = time.time()
    results = []
    for scored_point in search_results.points:
        results.append({
            "id": scored_point.id,
            "score": scored_point.score,
            "payload": scored_point.payload
        })
    format_time = time.time() - start_time_1
    print(f"Format time: {format_time} seconds")

    return results

"""## Response Generation"""

def generate_response(query: str, context: List[Dict[str, Any]]) -> str:
    """Generate a response using OpenAI based on retrieved context."""
    # Prepare context text from search results
    start_time = time.time()
    context_text = "\n\n".join([
        f"Document {i+1}:\nText: {item['payload']['text']}\nKeywords: {', '.join(item['payload']['keywords'])}"
        for i, item in enumerate(context)
    ])
    context_time = time.time() - start_time
    print(f"Context time: {context_time} seconds")

    system_prompt = """
    You are an authoritative academic assistant for Notre Dame University (NDU) providing precise information based on the retrieved documents.

    IMPORTANT GUIDELINES:
    1. Provide ONLY ONE definitive answer based on the highest relevance matches in the context.
    2. If multiple potential answers exist, choose the one with the strongest evidence in the retrieved documents.

    Your goal is to provide the single most accurate answer as if you were an official university representative.
    """

    user_prompt = f"Question: {query}\n\nContext:\n{context_text}"
    start_time_1 = time.time()
    response = client.chat.completions.create(
        model="openai/gpt-4o",  # Using a powerful model for response generation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    response_time = time.time() - start_time_1
    print(f"Response time: {response_time} seconds")

    return response.choices[0].message.content

"""## RAG Pipeline"""

def rag_pipeline_simple(query: str, collection_name: str = "admission_course_guide"):
    """Complete RAG pipeline from user query to response."""
    print(f"Original query: {query}")

    # Search Qdrant with a single query
    search_results = search_qdrant_simple(query, collection_name, limit=3)

    # Generate response
    response = generate_response(query, search_results)

    return {
        "original_query": query,
        "search_results": search_results,
        "response": response
    }

"""## Example Usage - Process PDF"""

# Example: Process a PDF file and upload to Qdrant
# pdf_path = "../data/raw/admission guide.pdf"  # Update with your PDF path
# 
# # Check if the PDF file exists
# if os.path.exists(pdf_path):
#     # Process the PDF and upload to Qdrant
#     processed_data = process_pdf_and_upload(pdf_path)
#     print("PDF processed and uploaded successfully")
# else:
#     print(f"Error: PDF file not found at {pdf_path}")
#     print("Please specify the correct path to your PDF file")

"""## Example Usage - Append Another PDF"""

# If you want to append another document to the same collection:
# second_pdf_path = "../data/raw/Nazir Hawi.pdf"
# if os.path.exists(second_pdf_path):
#     append_pdf_to_collection(second_pdf_path)
#     print("Second PDF appended successfully")
# else:
#     print(f"Warning: Second PDF file not found at {second_pdf_path}")

"""## Example Usage - Test Query"""

# Test the pipeline with a sample query
# start_time = time.time()
# result = rag_pipeline_simple("How many credits does a computer science major have?")
# end_time = time.time()
# print(f"Total time taken: {end_time - start_time} seconds")
# 
# # Display the response
# print("\nFinal Response:")
# print(result["response"])

"""## Variations"""

def generate_query_variations(query: str) -> List[str]:
    """Generate variations of the query using OpenAI."""
    system_prompt = """
    Create one alternative versions of the user's query.
    Each version should:
    1. Maintain the original meaning
    2. Use different wording or phrasing
    3. Be a complete, well-formed question

    Return ONLY two variations, one per line, with no additional text.
    """

    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.7,
        max_tokens=200
    )

    variations_text = response.choices[0].message.content
    variations = [line.strip() for line in variations_text.split('\n') if line.strip()]

    # Ensure we have exactly 2 variations
    if len(variations) > 1:
        variations = variations[:1]
    while len(variations) < 1:
        variations.append(query)  # Use original query as fallback

    return variations

# --- Main Function ---
def main():
    """Main function to load processed JSON files and run example queries."""
    # Set paths for JSON files instead of PDFs
    primary_json_path = "data/processed/study_resource.json"
    # secondary_json_path = "../data/processed/nazir_hawi.json"
    collection_name = "study_resource_json"
    
    # Load and process primary JSON file
    if os.path.exists(primary_json_path):
        print("--- Loading Primary JSON File ---")
        processed_data = read_json_file(primary_json_path)
        
        # Create collection and upload data
        create_collection(collection_name)
        process_and_upload_data(processed_data, collection_name)
        print(f"Processed data uploaded to Qdrant collection {collection_name}")
    else:
        print(f"Error: JSON file not found at {primary_json_path}")
    
    # Uncomment to load and process secondary JSON file
    # if os.path.exists(secondary_json_path):
    #     print("\n--- Loading Secondary JSON File ---")
    #     secondary_data = read_json_file(secondary_json_path)
    #     
    #     # Get count of existing points to avoid ID conflicts
    #     collection_info = qdrant_client.get_collection(collection_name)
    #     existing_points = 12  # Should be dynamically calculated in production
    #     
    #     # Process and upload secondary data with offset IDs
    #     batch_size = 10
    #     for i in range(0, len(secondary_data), batch_size):
    #         batch = secondary_data[i:i+batch_size]
    #         
    #         points = []
    #         for j, entry in enumerate(batch):
    #             payload = create_payload(entry)
    #             embedding = generate_embedding(entry["text"])
    #             
    #             points.append(models.PointStruct(
    #                 id=existing_points + i + j,
    #                 vector=embedding,
    #                 payload=payload
    #             ))
    #         
    #         qdrant_client.upsert(
    #             collection_name=collection_name,
    #             points=points
    #         )
    #     print("Secondary data appended successfully")
    # else:
    #     print(f"Warning: Secondary JSON file not found at {secondary_json_path}")
    
    # Run example query
    print("\n--- Running Example Query ---")
    query = "What is the study resource?"
    start_time = time.time()
    result = rag_pipeline_simple(query, collection_name)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
    
    # Display the response
    print("\nFinal Response:")
    print(result["response"])
    
    # Testing query variations
    print("\n--- Testing Query Variations ---")
    variations = generate_query_variations(query)
    print("Query variations:")
    for i, variation in enumerate(variations):
        print(f"{i+1}. {variation}")

# Execute main function when script is run directly
if __name__ == "__main__":
    main()