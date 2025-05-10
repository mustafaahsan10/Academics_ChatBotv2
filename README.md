# University Assistant Chatbot

A comprehensive university assistant chatbot system with RAG (Retrieval Augmented Generation) capabilities to answer questions about courses, schedules, faculty, admission requirements, and more.

## Features

- **Hybrid Search**: Combines vector similarity with keyword and heading matching to provide more accurate responses
- **Document Structure Awareness**: Preserves document hierarchies and headings for contextual understanding
- **Multi-language Support**: Currently supports English and Arabic interfaces
- **Various Data Source Handling**: Process PDFs, DOCXs, TXTs, CSVs, and Excel files
- **Pre-processed Data Support**: Can ingest both raw and pre-processed structured data
- **Rich Metadata**: Extracts and utilizes headings, keywords, and entities for improved retrieval

## System Components

### Data Ingestion

- `ingest_data.py`: Process raw document files (PDF, DOCX, TXT, Excel, CSV) and extract structured data
- `ingest_processed.py`: Ingest pre-processed structured data from the `data/processed` directory

### Chatbot Core

- `utils/chatbot.py`: Contains the core RAG implementation with hybrid search capabilities
- `app.py`: Streamlit-based user interface for interacting with the chatbot

### Utilities

- `delete_collection.py`: Delete collections from Qdrant
- `export_chunks.py`: Export processed chunks from the vector database
- `test_search.py`: Test search functionality independently

## Setup and Usage

### Environment Setup

1. Create a `.env` file based on the `env_template.txt`
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Data Ingestion

For raw data:
```
python ingest_data.py --data_dir data/raw --recreate
```

For pre-processed data:
```
python ingest_processed.py --data_dir data/processed
```

### Running the Application

```
streamlit run app.py
```

## Hybrid Search Architecture

The system implements a sophisticated hybrid search approach that combines:

1. **Vector Similarity**: Semantic similarity using OpenAI embeddings
2. **Keyword Matching**: Exact/partial matches of keywords in text
3. **Heading Matching**: Finding matches within document headings
4. **Entity Matching**: Extracting and matching specific entities like course codes

The search results are combined and re-ranked based on a weighted scoring system:
- Vector similarity (70% base weight)
- Keyword matches (30% base weight)
- Heading matches (20% boost per match)
- Entity matches (30% boost per match)

## Data Structure

### Raw Data Processing

Raw documents are processed to extract:
- Document text content
- Headings and document structure
- Metadata like document type, keywords, etc.

### Processed Data Format

The processed data directory contains JSON-structured files with:
- Heading information
- Text content
- Keywords and entities

## Future Enhancements

- Support for more languages
- Improved entity extraction
- Personalized search based on user profiles
- Integration with university systems for real-time data 