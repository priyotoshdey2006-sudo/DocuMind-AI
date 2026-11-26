# DocuMind AI
### Intelligent Hybrid RAG Document Assistant

**DocuMind AI is an AI-powered document intelligence application that allows users to ask natural-language questions about PDF documents and receive context-grounded answers with source references.**

It combines keyword retrieval, semantic vector search, document-level filtering, and Large Language Model (LLM) generation through a Retrieval-Augmented Generation (RAG) architecture.

---

## Table of Contents

1. Project Overview
2. Problem Statement and Target Users
3. Key Features
4. Technology Stack
5. System Architecture
6. AI and Prompt Engineering
7. Assumptions and Limitations
8. Project Structure
9. Installation and Local Setup
10. Testing and Validation
11. Live Deployment
12. Security and Accessibility
13. Future Improvements
14. Author

---

## 1. Project Overview

Searching through lengthy PDF documents manually can be time-consuming. Traditional keyword searches may also fail to identify relevant information when users phrase their questions differently from the original document.

DocuMind AI solves this problem by retrieving relevant passages from a user's selected PDF before generating an answer using an AI language model.

Unlike a traditional chatbot, DocuMind AI uses document-based retrieval to provide contextual answers grounded in the uploaded content.

**Project Focus:** Document Intelligence and AI-powered Productivity.

**Target Users:**
- Students who need help understanding academic notes and study materials.
- Professionals working with lengthy PDF documents.
- Researchers searching for specific information.
- Anyone who wants to interact with personal documents using natural language.

## 2. Problem Statement

Reading and searching through large PDF documents requires time and effort. Basic keyword searches may not understand the semantic meaning of a question.

DocuMind AI addresses this challenge through hybrid information retrieval, combining keyword-based and semantic search to identify relevant document content and generate contextual answers.

## 3. Key Features

- **PDF Upload:** Upload and process PDF documents.
- **Text Extraction:** Extract document text and page information using PyMuPDF.
- **Text Chunking:** Divide extracted content into smaller passages.
- **Hybrid Search:** Combine BM25 keyword retrieval with dense vector search.
- **Rank Fusion:** Combine search results using Reciprocal Rank Fusion (RRF).
- **Vector Database:** Store document embeddings using ChromaDB.
- **Local Embeddings:** Generate embeddings using Ollama and `nomic-embed-text`.
- **AI Answer Generation:** Generate contextual responses using Google's Gemini API.
- **Source Citations:** Display source document names and available page references.
- **Document Filtering:** Restrict retrieval to the document selected by the user.
- **Document Management:** View and delete indexed documents.
- **Modern Interface:** React and TypeScript frontend connected to a FastAPI backend.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| PDF Processing | PyMuPDF |
| Keyword Retrieval | BM25 |
| Dense Retrieval | ChromaDB |
| Embeddings | Ollama, nomic-embed-text |
| AI Model | Google Gemini |
| API Communication | REST API |
| Automated Testing | pytest |

## 5. System Architecture

DocuMind AI follows a Retrieval-Augmented Generation architecture.

### Document Processing Workflow

1. The user uploads a PDF document.
2. The backend validates and processes the uploaded file.
3. PyMuPDF extracts text and page information.
4. Extracted text is divided into smaller chunks.
5. Ollama generates vector embeddings for the chunks.
6. ChromaDB stores the embeddings and associated document metadata.

### Question Answering Workflow

1. The user selects a document and enters a natural-language question.
2. The frontend sends the question and selected document to the FastAPI backend.
3. BM25 retrieves relevant passages using keyword matching.
4. Dense retrieval identifies semantically relevant passages.
5. Reciprocal Rank Fusion combines the retrieval results.
6. Relevant document context is passed to the Gemini model.
7. Gemini generates an answer based on the retrieved context.
8. The frontend displays the answer and available source citations.

### High-Level Architecture

```text
             USER
               |
               v
     React + TypeScript Frontend
               |
            REST API
               |
               v
         FastAPI Backend
               |
       +-------+--------+
       |                |
       v                v
  PDF Processing   Question Processing
       |                |
       v                v
  PyMuPDF          Selected Document
       |               Filter
       v                |
  Text Chunking         |
       |          +-----+------+
       v          |            |
  Ollama        BM25       Dense Search
  Embeddings      |            |
       |          +-----+------+
       v                |
   ChromaDB             v
                   RRF Ranking
                        |
                        v
                Retrieved Context
                        |
                        v
                 Gemini LLM
                        |
                        v
              Answer + Citations
```

## 6. AI and Prompt Engineering

DocuMind AI uses a context-grounded prompting approach.

Instead of asking the language model to answer questions using unrestricted general knowledge, the application supplies relevant passages retrieved from the selected PDF.

The answer-generation instructions are designed to:

- Prioritize the retrieved document context.
- Answer the user's actual question.
- Avoid presenting unsupported information as document facts.
- Communicate when the available context is insufficient.
- Preserve relevant source information for citations.
- Follow the selected-document restriction.

### Contextual Decision-Making

The system follows this logic:

- If the user selects a document, retrieval is restricted to that document.
- If relevant passages are found, those passages are supplied to the answer-generation model.
- If the retrieved information is insufficient, the response should communicate that limitation.
- If the user submits a greeting, the application can handle it without unnecessary document retrieval.

This approach connects the user's question, selected document, retrieval process, and AI-generated response.

## 7. Assumptions and Limitations

### Assumptions

- Users provide valid PDF documents.
- Uploaded documents contain extractable text.
- Ollama is available when using local embeddings.
- A valid Gemini API key is configured for AI answer generation.
- Retrieved passages provide sufficient context for document-related questions.

### Limitations

- Scanned PDFs may require OCR support.
- Answer quality depends on document quality and retrieval relevance.
- Gemini answer generation requires API connectivity.
- Local embedding generation requires Ollama and sufficient system resources.
- The application is not designed to replace professional legal, medical, or financial advice.

## 8. Project Structure

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
│   ├── src/
│   ├── package.json
│   └── ...
├── scripts/
├── tests/
│   └── test_retrieval.py
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
└── README.md
```

## 9. Installation and Local Setup

### Prerequisites

- Python version compatible with the project dependencies.
- Node.js and npm.
- Git.
- Ollama.
- A Google Gemini API key.

### Step 1: Clone the Repository

```bash
git clone https://github.com/priyotoshdey2006-sudo/DocuMind-AI.git
cd DocuMind-AI
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it using Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_FALLBACK_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT=120000
GEMINI_MAX_RETRIES=2
GEMINI_RETRY_DELAY=2
```

Use model names supported by your Gemini API account.

### Step 5: Install and Configure Ollama

Install Ollama and download the embedding model:

```bash
ollama pull nomic-embed-text
```

Ensure that Ollama is running locally.

### Step 6: Start the Backend

From the project root:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Backend URL:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

### Step 7: Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
```

Create a `.env` file inside the `frontend` directory:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Start the frontend:

```bash
npm run dev
```

Open the local frontend URL displayed by Vite, usually:

`http://localhost:5173`

## 10. Testing and Validation

Automated testing is used to validate the retrieval functionality.

### Automated Retrieval Unit Test Results

The retrieval test suite was executed successfully.

| Metric | Result |
|---|---:|
| Total Tests Executed | 10 |
| Passed | 10 |
| Failed | 0 |
| Test Type | Mock-based Retrieval Unit Tests |

**Execution result:**

```text
10 passed, 1 warning in 9.34s
```

The tests cover:

- Tokenization and text processing.
- Empty BM25 indexes.
- BM25 search results.
- Source-specific BM25 filtering.
- Document listing and counts.
- Dense retrieval behavior.
- Dense-search source filtering.
- Hybrid retrieval combining BM25 and dense results.
- Hybrid result limits.

Run the automated tests using:

```bash
python -m pytest -v
```

### Additional Validation Scenarios

| Test Scenario | Expected Behaviour |
|---|---|
| Valid PDF upload | Document is processed successfully |
| Invalid file upload | Application rejects unsupported files |
| Empty document | Application handles missing extractable content |
| Document-specific question | Retrieval uses the selected document |
| Relevant question | Answer is generated using retrieved context |
| Irrelevant question | Application communicates insufficient context |
| Document deletion | Document is removed from the indexed collection |
| Gemini API failure | Application handles generation errors |
| Large PDF upload | File-size validation is enforced |

Validate the frontend production build:

```bash
cd frontend
npm run build
```

**Testing scope:** The 10 passing tests are mock-based retrieval unit tests. They do not establish that every deployed workflow has passed end-to-end testing.

## 11. Live Deployment

DocuMind AI is deployed using separate frontend and backend services.

| Component | Platform | Deployment URL |
|---|---|---|
| Frontend | Vercel | https://docu-mind-ai-jade.vercel.app |
| Backend API | Render | https://documind-ai-x8i0.onrender.com |
| API Documentation | FastAPI / Render | https://documind-ai-x8i0.onrender.com/docs |

The React frontend communicates with the FastAPI backend through the `VITE_API_URL` environment variable.

**Live Application:**  
https://docu-mind-ai-jade.vercel.app

## 12. Security and Accessibility

### Security Practices

- Store API keys in environment variables.
- Never hard-code credentials in application source code.
- Exclude `.env` files and secrets from Git.
- Validate uploaded file types and file sizes.
- Restrict document operations to valid indexed documents.
- Handle API failures without exposing sensitive information.

### Accessibility Goals

- Use semantic HTML elements.
- Provide accessible labels for form controls.
- Support keyboard navigation.
- Maintain readable text and sufficient contrast.
- Provide clear error and status messages.
- Maintain responsive layouts across screen sizes.

## 13. Future Improvements

- OCR support for scanned PDFs.
- More comprehensive automated and integration testing.
- Advanced retrieval-quality evaluation.
- Improved accessibility and usability.
- Conversation history.
- Authentication and user management.
- Support for additional document formats.
- Advanced document analytics.

## 14. Author

**Priyotosh Dey**

B.Tech — Computer Science and Engineering (AI & ML)

- **GitHub:** https://github.com/priyotoshdey2006-sudo
- **LinkedIn:** https://www.linkedin.com/in/priyotosh-dey-96203a3aa/

---

**DocuMind AI — Making document understanding smarter through Retrieval-Augmented Generation.**

Built as an AI and Machine Learning project exploring hybrid information retrieval, prompt engineering, and practical AI-powered document intelligence.