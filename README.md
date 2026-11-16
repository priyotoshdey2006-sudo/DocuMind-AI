\# DocuMind AI — Hybrid RAG Document Assistant



An AI-powered PDF question-answering system using Hybrid Retrieval-Augmented Generation (RAG).



\## Features



\- Upload and process PDF documents.

\- Extract text using PyMuPDF.

\- Keyword search using BM25.

\- Dense vector search using Ollama embeddings.

\- Store embeddings in ChromaDB.

\- Combine search results using Reciprocal Rank Fusion (RRF).

\- Generate AI answers using Google Gemini.

\- Display source document names and page citations.

\- Filter retrieval by selected document.

\- Modern React frontend with FastAPI backend.



\## Tech Stack



\- \*\*Frontend:\*\* React, TypeScript, Vite

\- \*\*Backend:\*\* Python, FastAPI

\- \*\*PDF Processing:\*\* PyMuPDF

\- \*\*Keyword Retrieval:\*\* BM25

\- \*\*Vector Database:\*\* ChromaDB

\- \*\*Embedding Model:\*\* Ollama `nomic-embed-text`

\- \*\*LLM:\*\* Google Gemini



\## How It Works



1\. Upload a PDF.

2\. Extract and split document text into chunks.

3\. Generate embeddings using Ollama.

4\. Store vectors in ChromaDB.

5\. Retrieve relevant chunks using BM25 and dense search.

6\. Combine results using Reciprocal Rank Fusion.

7\. Generate an answer using Gemini.

8\. Display the answer with source citations.



\## Project Structure



```text

DocuMind-AI/

├── api/

├── app/

├── frontend/

├── scripts/

├── tests/

├── requirements.txt

├── requirements-lock.txt

├── .gitignore

└── README.md

```



\## Installation



\### 1. Clone the repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd DocuMind-AI

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



Activate on Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\### 4. Configure Gemini API key



Create a `.env` file in the project root:



```env

GEMINI\_API\_KEY=your\_gemini\_api\_key

```



Never upload your real API key to GitHub.



\### 5. Install Ollama embedding model



Install Ollama and run:



```bash

ollama pull nomic-embed-text

```



Make sure Ollama is running.



\### 6. Run the backend



```bash

python -m uvicorn api.main:app --reload --port 8000

```



Backend: `http://127.0.

