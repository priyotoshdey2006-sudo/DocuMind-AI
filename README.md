# DocuMind AI — Intelligent Hybrid RAG Document Assistant

**An AI-powered document intelligence assistant that understands PDF content, retrieves relevant information, and generates contextual answers with source citations.**

## 1. Project Overview

DocuMind AI is an intelligent document question-answering application built using Retrieval-Augmented Generation (RAG).

It helps users interact with their PDF documents through natural-language questions instead of manually searching through lengthy documents.

The application combines keyword-based retrieval, dense vector search, document filtering, and Large Language Model (LLM) generation to produce relevant, context-aware answers grounded in uploaded documents.

Unlike a basic chatbot, DocuMind AI retrieves relevant document passages before generating an answer. This helps keep responses connected to the actual source material.

## 2. Chosen Challenge Vertical and Target Persona

**Project focus:** Document Intelligence and AI-powered Productivity.

**Target persona:**
- Students who need to understand academic notes and study materials.
- Professionals who work with lengthy PDF documents.
- Researchers who need to locate specific information in documents.
- Users who want to ask questions about their own documents.

**Problem being addressed:**

Reading and searching through lengthy PDF documents can be time-consuming. Traditional keyword searches may also fail to understand the meaning or context of a question.

DocuMind AI addresses this problem by combining semantic and keyword-based retrieval with AI-generated answers.

*Note: Document Intelligence and Productivity is the project's working vertical description. If the challenge platform provides a specific vertical label, use that exact label in the final submission.*

## 3. Core Features

- **PDF Upload:** Upload and process PDF documents.
- **Text Extraction:** Extract document content and divide it into manageable chunks.
- **Hybrid Search:** Combine BM25 keyword retrieval with dense vector search.
- **Vector Database:** Store document embeddings using ChromaDB.
- **Local Embeddings:** Generate embeddings using Ollama and `nomic-embed-text`.
- **AI Answer Generation:** Use Google's Gemini API to generate contextual answers.
- **Source Citations:** Display source document names and page references.
- **Document Filtering:** Restrict retrieval to the selected document.
- **Document Management:** Upload, view, and delete indexed documents.
- **Modern Interface:** Use a React and TypeScript frontend connected to a FastAPI backend.

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
| Styling | CSS |

## 5. System Architecture

The application follows a retrieval-augmented generation architecture.

### Document Processing Workflow

1. The user uploads a PDF document.
2. The backend validates and processes the uploaded file.
3. PyMuPDF extracts text and page information.
4. Extracted text is divided into smaller chunks.
5. Ollama generates vector embeddings for the chunks.
6. ChromaDB stores the embeddings and associated document metadata.

### Question Answering Workflow

1. The user selects a document and enters a natural-language question.
2. The backend receives the question and selected document.
3. BM25 retrieves relevant passages using keyword matching.
4. Dense retrieval finds semantically relevant passages.
5. Reciprocal Rank Fusion combines retrieval results.
6. Relevant context is passed to the Gemini model.
7. Gemini generates an answer grounded in the retrieved document content.
8. The frontend displays the answer and available source citations.

## 6. AI and Prompt Engineering Approach

DocuMind AI uses a context-grounded prompting approach.

Instead of asking the language model to answer questions using unrestricted general knowledge, the application supplies relevant passages retrieved from the selected PDF.

The answer-generation instructions are designed to:

- Use the retrieved document context as the primary source of information.
- Answer the user's actual question.
- Avoid presenting unsupported information as document facts.
- Communicate when the available context is insufficient.
- Preserve relevant source information for citations.
- Follow the selected document restriction.

### Contextual Decision-Making

The system uses the following logic:

- If the user selects a document, retrieval is restricted to that document.
- If relevant passages are found, those passages are supplied to the answer-generation model.
- If the retrieved information is insufficient, the answer should communicate that limitation.
- If the user submits a greeting, the application can handle it without unnecessary document retrieval.

This approach connects the user's question, selected document, retrieval process, and AI-generated response.

## 7. Assumptions and Limitations

### Assumptions

- Users provide valid PDF documents.
- Uploaded documents contain extractable text.
- Ollama is available when using local embeddings.
- A valid Gemini API key is configured for AI answer generation.
- Retrieved passages provide the context needed to answer document-related questions.
- Users understand that AI-generated responses should be checked against the original source when accuracy is critical.

### Limitations

- Scanned PDFs may require OCR support if they do not contain extractable text.
- Answer quality depends on document quality and retrieval relevance.
- Gemini answer generation requires API connectivity.
- Local embedding generation requires Ollama and sufficient system resources.
- The current application is not designed to replace professional legal, medical, or financial advice.

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
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
└── README.md
```

## 9. Installation and Local Setup

### Prerequisites

- Python 3.11 or compatible version supported by the dependencies.
- Node.js and npm.
- Git.
- Ollama.
- A Google Gemini API key.

### Step 1: Clone the repository

```bash
git clone https://github.com/priyotoshdey2006-sudo/DocuMind-AI.git
cd DocuMind-AI
```

### Step 2: Create a Python virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install backend dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure environment variables

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

**Security:** Never commit `.env` or expose your API key in source code, screenshots, or public repositories.

### Step 5: Install and configure Ollama

Install Ollama and download the embedding model:

```bash
ollama pull nomic-embed-text
```

Ensure Ollama is running locally.

### Step 6: Start the backend

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Backend:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

### Step 7: Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local frontend URL displayed by Vite, usually:

http://localhost:5173

## 10. Testing and Validation

Testing is important for ensuring that the application works reliably.

The following test scenarios should be validated:

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

Run automated tests, where implemented, using:

```bash
pytest -v
```

Validate the frontend production build using:

```bash
cd frontend
npm run build
```

*Test scenarios describe the validation plan. Only report tests as passed when they have actually been executed successfully.*

## 11. Deployment

The FastAPI backend is deployed on Render.

**Backend URL:**

https://documind-ai-x8i0.onrender.com

**API documentation:**

https://documind-ai-x8i0.onrender.com/docs

The frontend requires a publicly accessible deployment for users to access the complete application through a browser.

Deployment configuration and environment variables must be maintained securely.

## 12. Security and Accessibility

### Security Practices

- Keep API keys in environment variables.
- Exclude secrets and local environment files from Git.
- Validate uploaded file types and file sizes.
- Restrict document operations to valid indexed documents.
- Handle API failures without exposing secret information.

### Accessibility Goals

- Use semantic HTML elements.
- Provide accessible labels for form controls.
- Support keyboard navigation.
- Maintain readable text and sufficient contrast.
- Provide clear error and status messages.
- Maintain responsive layouts across screen sizes.

## 13. Future Improvements

- OCR support for scanned PDFs.
- Advanced retrieval evaluation.
- More comprehensive automated tests.
- Improved accessibility.
- Conversation history.
- Authentication and user management.
- Additional document formats.
- More advanced document analytics.

## 14. Author

**Priyotosh Dey**

B.Tech — Computer Science and Engineering (AI & ML)

GitHub: https://github.com/priyotoshdey2006-sudo

LinkedIn: https://www.linkedin.com/in/priyotosh-dey-96203a3aa/

---

**DocuMind AI — Making document understanding smarter through Retrieval-Augmented Generation.**

Built as an AI and Machine Learning project exploring hybrid information retrieval, prompt engineering, and practical AI-powered document intelligence.