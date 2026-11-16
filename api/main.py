

import uuid
from pathlib import Path
from threading import Lock

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.answer import answer_question
from app.ingest import ingest_pdf
from app.retrieval import HybridRetriever


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

MAX_FILE_SIZE = 20 * 1024 * 1024

RAW_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Application
# --------------------------------------------------

app = FastAPI(
    title="DocuMind-AI API",
    description="Hybrid RAG document question-answering API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = HybridRetriever()
data_lock = Lock()


# --------------------------------------------------
# Request models
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    source: str | None = None


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "application": "DocuMind-AI",
        "indexed_chunks": retriever.collection.count(),
    }


# --------------------------------------------------
# List indexed documents
# --------------------------------------------------

@app.get("/api/documents")
def get_documents():
    with data_lock:
        documents = retriever.list_documents()

    return {
        "documents": documents,
        "total_documents": len(documents),
    }


# --------------------------------------------------
# PDF upload
# --------------------------------------------------

def process_pdf_upload(contents: bytes, original_filename: str):
    upload_id = uuid.uuid4().hex

    stored_filename = f"{upload_id}_{original_filename}"
    stored_path = RAW_DIR / stored_filename

    try:
        stored_path.write_bytes(contents)

        chunks = ingest_pdf(stored_path)

        for chunk in chunks:
            chunk["id"] = f"{upload_id}-{chunk['id']}"
            chunk["metadata"]["source"] = original_filename

        with data_lock:
            chunks_added = retriever.add_chunks(chunks)

        return {
            "message": "PDF processed successfully.",
            "filename": original_filename,
            "chunks_added": chunks_added,
        }

    except Exception:
        stored_path.unlink(missing_ok=True)
        raise


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing.",
        )

    filename = file.filename.replace("\\", "/").split("/")[-1]

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF size must not exceed 20 MB.",
        )

    if not contents.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )

    try:
        return await run_in_threadpool(
            process_pdf_upload,
            contents,
            filename,
        )

    except (ValueError, RuntimeError, pymupdf.FileDataError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {error}",
        ) from error

    finally:
        await file.close()


# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------

def process_question(question: str, source: str | None):
    with data_lock:
        return answer_question(
            question,
            retriever,
            source=source,
        )


@app.post("/api/chat")
async def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    source = request.source

    if source is not None:
        with data_lock:
            available_documents = {
                document["filename"]
                for document in retriever.list_documents()
            }

        if source not in available_documents:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {source}",
            )

    try:
        result = await run_in_threadpool(
            process_question,
            question,
            source,
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {error}",
        ) from error
