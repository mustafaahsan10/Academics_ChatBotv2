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

# Download stopwords if not already downloaded
# nltk.download('stopwords')

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
        score_threshold=0.3
    )
    print(search_results)
    search_time = time.time() - start_time
    print(f"Search time: {search_time} seconds")
    # Format results

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
    3. Focus exclusively on directly answering the user's question with specific facts from the context.
    4. If a direct answer isn't clearly available in the context, state this clearly rather than speculating.
    5. Format your answer concisely using bold for key facts and figures.
    6. Avoid listing multiple possibilities or alternatives unless specifically requested.

    Your goal is to provide the single most accurate answer as if you were an official university representative.
    """

    user_prompt = f"Question: {query}\n\nContext:\n{context_text}"
    start_time_1 = time.time()
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",  # Using a powerful model for response generation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    response_time = time.time() - start_time_1
    print(f"Response time: {response_time} seconds")

    return response.choices[0].message.content

# def rag_pipeline(query: str, collection_name: str = "admission_course_guide"):
#     """Complete RAG pipeline from user query to response."""
#     print(f"Original query: {query}")

#     # Search Qdrant
#     search_results = search_qdrant(query, collection_name, limit=3)

#     # Generate response
#     response = generate_response(query, search_results)

#     return {
#         "original_query": query,
#         "search_results": search_results,
#         "response": response
#     }

def rag_pipeline_simple(query: str, collection_name: str = "admission_course_guide"):
    """Complete RAG pipeline from user query to response."""
    print(f"Original query: {query}")

    # Search Qdrant with a single query
    search_results = search_qdrant_simple(query, collection_name, limit=5)

    # Generate response
    response = generate_response(query, search_results)

    return {
        "original_query": query,
        "search_results": search_results,
        "response": response
    }

# Example usage
if __name__ == "__main__":
    # Process a PDF file and upload to Qdrant
    pdf_path = "../data/raw/admission guide.pdf"  # Update with your PDF path
    
    # Check if the PDF file exists
    if os.path.exists(pdf_path):
        # Process the PDF and upload to Qdrant
        # processed_data = process_pdf_and_upload(pdf_path)
        
        # print(processed_data)
        # # If you want to append another document to the same collection:
        # second_pdf_path = "../data/raw/Nazir Hawi.pdf"
        # if os.path.exists(second_pdf_path):
        #     append_pdf_to_collection(second_pdf_path)
        
        # Test the pipeline with a sample query
        start_time = time.time()
        result = rag_pipeline_simple("How many credits does a computer science major have?")
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time} seconds")
        
        # Display the response
        print("\nFinal Response:")
        print(result["response"])
    else:
        print(f"Error: PDF file not found at {pdf_path}")
        print("Please specify the correct path to your PDF file")