# YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about any YouTube video with available captions. The application extracts video transcripts, generates embeddings using Ollama, stores them in a FAISS vector database, and retrieves relevant context to answer user queries accurately.

## Features

* Extracts transcripts from YouTube videos
* Automatic transcript chunking using LangChain
* Local embeddings using BGE-M3 via Ollama
* FAISS vector database for efficient similarity search
* MMR (Max Marginal Relevance) retrieval for diverse context
* Persistent FAISS indexing to avoid regenerating embeddings
* Dynamic YouTube video selection
* Local LLM inference using Llama 3.2
* Offline RAG pipeline after transcript retrieval

## Tech Stack

* Python
* LangChain
* Ollama
* Llama 3.2
* BGE-M3 Embeddings
* FAISS
* YouTube Transcript API

## Project Workflow

```text
YouTube Video
      │
      ▼
Transcript Extraction
      │
      ▼
Text Chunking
      │
      ▼
BGE-M3 Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
Retriever (MMR)
      │
      ▼
Llama 3.2
      │
      ▼
Answer Generation
```

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd youtube-rag-chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Ollama Setup

Install Ollama and pull the required models:

```bash
ollama pull llama3.2
ollama pull bge-m3
```

Start Ollama:

```bash
ollama serve
```

## Usage

Run the application:

```bash
python main.py
```

Enter a YouTube video ID:

```text
Enter YouTube Video ID:
Gfr50f6ZBvo
```

Ask questions:

```text
ask a question:
What is LangChain?
```

Exit:

```text
q
```

## Persistent Indexing

The application automatically creates a FAISS index for each video.

First run:

```text
Fetch Transcript
Create Chunks
Generate Embeddings
Create FAISS Index
Save Index
```

Subsequent runs:

```text
Load Existing FAISS Index
```

This significantly reduces startup time.


## Future Improvements

* Stream LLM responses token-by-token
* Multi-video knowledge base
* Conversation memory
* Web interface using Streamlit or Flask
* Citation of retrieved transcript chunks

## License

This project is intended for educational and learning purposes.

## Author

Naimish Sorathiya