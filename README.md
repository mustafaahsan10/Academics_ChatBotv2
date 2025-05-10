# University Assistant Chatbot

A bilingual (English/Arabic) university assistant chatbot built with Streamlit, LangChain, and Qdrant. It provides answers to common university questions about courses, schedules, exams, faculty information, and library resources.

## Features

- **Academic Information**: Get details on courses, majors, prerequisites, class schedules, exams, etc.
- **Faculty Info**: Access professor details, office hours, teaching courses, etc.
- **Library Resources**: Check book availability, borrowing process, etc.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd university-assistant-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

1. Rename `env_template.txt` to `.env`
2. Add your API keys:
   - OPENAI_API_KEY: Your OpenAI API key
   - OPENROUTER_API_KEY: Your Openrouter API key
   - QDRANT_API_KEY: Your Qdrant API key
   - QDRANT_URL: Your Qdrant Cluster URL

### 4. Prepare Data

1. Place your university data files in the `data/raw` directory:
   - PDFs (course catalogs, admission guides, faculty info)
   - Excel files (course materials, book catalogs)
   - DOC/DOCX files
   - TXT files

### 5. Ingest Data to Vector Database

Run the data ingestion script to process files and upload embeddings to Qdrant:

```bash
python ingest_data.py
```

### 6. Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

## Project Structure

```
university-assistant-chatbot/
├── app.py                   # Main Streamlit application
├── ingest_data.py           # Data ingestion and vectorization script
├── requirements.txt         # Python dependencies
├── env_template.txt         # Template for environment variables
├── README.md                # This file
├── utils/
│   └── chatbot.py           # Chatbot implementation with RAG
└── data/
    ├── raw/                 # Raw data files (PDFs, Excel, etc.)
    └── processed/           # Processed data (if needed)
```

## Usage

1. Select your preferred language (English or Arabic) from the sidebar
2. Type your university-related questions in the chat input

## Limitations

- The chatbot can only answer questions based on the data provided
- Arabic responses might have lower fidelity initially as most reference material is in English
- Documents with complex layouts (images, graphs, tables) might not be processed correctly 