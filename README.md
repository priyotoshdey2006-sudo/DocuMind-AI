# DocuMind AI — Hybrid RAG Document Assistant

**Ask questions about your PDF documents and get AI-powered answers with source citations.**

DocuMind AI is a document question-answering application built using Retrieval-Augmented Generation (RAG). It combines keyword-based retrieval, vector search, and an LLM to answer questions using the content of uploaded PDF documents.

## Features

- **PDF Upload:** Upload and process PDF documents.
- **Hybrid Search:** Combines BM25 keyword retrieval with dense vector search.
- **Vector Database:** Uses ChromaDB to store document embeddings.
- **Local Embeddings:** Uses Ollama with `nomic-embed-text`.
- **AI Answers:** Generates responses using Google's Gemini API.
- **Source Citations:** Shows document names and page references.
- **Document Filtering:** Restricts retrieval to the selected document.
- **Modern Interface:** React-based frontend with a FastAPI backend.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| PDF Processing | PyMuPDF |
| Keyword Search | BM25 |
| Vector Database | ChromaDB |
| Embeddings | Ollama, nomic-embed-text |
| AI Model | Google Gemini |
| Styling | CSS |

## Architecture

1. Upload a PDF document.
2. Extract text and split it into chunks.
3. Generate embeddings using Ollama.
4. Store embeddings in ChromaDB.
5. Retrieve relevant chunks using BM25 and dense vector search.
6. Combine retrieval results using Reciprocal Rank Fusion.
7. Generate an answer using Gemini.
8. Display the answer with source citations.

## Project Structure

```text
DocuMind-AI/
├── api/
│   └── main.py
├── app/
│   ├── answer.py
│   ├── ingest.py
│   ├── retrieval.py
│   └── ui.py
├── frontend/
├── scripts/
├── tests/
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DocuMind-AI
```

### 2. Create and activate a Python virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Never commit your real API key.

### 5. Install Ollama and download the embedding model

Install Ollama, then run:

```bash
ollama pull nomic-embed-text
```

Make sure Ollama is running locally.

### 6. Start the backend

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Backend API: `http://127.0.0.1:8000`

### 7. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the frontend URL displayed by Vite, usually `http://localhost:5173`.

## Current Development Status

This project is actively being developed. Core PDF ingestion, hybrid retrieval, document filtering, Gemini-based answer generation, and source citations have been implemented and tested locally.

Cross-encoder reranking and more comprehensive retrieval evaluation are planned improvements.

## Future Improvements

- Cross-encoder reranking
- Improved retrieval evaluation metrics
- Better document management
- Conversation history
- Deployment and cloud hosting
- Authentication and user management

## Author

**Priyotosh Dey**

B.Tech — Computer Science and Engineering (AI & ML)

GitHub: [priyotoshdey2006-sudo](https://github.com/priyotoshdey2006-sudo)

---

*Built as a learning and portfolio project exploring Hybrid RAG, information retrieval, and AI-powered document understanding.*