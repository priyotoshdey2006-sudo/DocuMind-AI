
import os
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

RAW_DIR = Path(
    os.getenv(
        "UPLOAD_DIR",
        str(BASE_DIR / "data" / "raw"),
    )
)

MAX_FILE_SIZE = 20 * 1024 * 1024

RAW_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="DocuMind-AI API",
    description="Hybrid RAG document question-answering API",
    version="1.2.0",
)


# --------------------------------------------------
# CORS Configuration
# --------------------------------------------------

frontend_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173",
)

allowed_origins = [
    origin.strip()
    for origin in frontend_origins.split(",")
    if origin.strip()
]

if "http://localhost:5173" not in allowed_origins:
    allowed_origins.append("http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Retriever and Thread Lock
# --------------------------------------------------

retriever = HybridRetriever()

data_lock = Lock()


# --------------------------------------------------
# Request Models
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
    )

    source: str | None = None


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/api/health")
def health_check():

    return {
        "status": "online",
        "application": "DocuMind-AI",
        "indexed_chunks": retriever.collection.count(),
    }


# --------------------------------------------------
# Get All Indexed Documents
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
# Delete a Document
# --------------------------------------------------

@app.delete("/api/documents/{filename}")
def delete_document(filename: str):

    filename = Path(filename).name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF filename.",
        )

    with data_lock:

        available_documents = {
            document["filename"]
            for document in retriever.list_documents()
        }

        if filename not in available_documents:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        try:

            deleted_chunks = retriever.delete_document(filename)

            deleted_files = 0

            for path in RAW_DIR.iterdir():

                if (
                    path.is_file()
                    and path.name.endswith("_" + filename)
                ):
                    path.unlink(missing_ok=True)
                    deleted_files += 1

            return {
                "message": "Document deleted successfully.",
                "filename": filename,
                "deleted_chunks": deleted_chunks,
                "deleted_files": deleted_files,
            }

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=f"Document deletion failed: {error}",
            ) from error


# --------------------------------------------------
# Process PDF Upload
# --------------------------------------------------

def process_pdf_upload(
    contents: bytes,
    original_filename: str,
):

    upload_id = uuid.uuid4().hex

    stored_filename = f"{upload_id}_{original_filename}"

    stored_path = RAW_DIR / stored_filename

    try:

        stored_path.write_bytes(contents)

        chunks = ingest_pdf(stored_path)

        if not chunks:
            raise ValueError(
                "No readable text found in the PDF."
            )

        for chunk in chunks:

            chunk["id"] = (
                f"{upload_id}-{chunk['id']}"
            )

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


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing.",
        )

    filename = (
        file.filename
        .replace("\\", "/")
        .split("/")[-1]
    )

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

    # PDF header is normally at the beginning,
    # but some valid PDFs have leading bytes.
    if b"%PDF-" not in contents[:1024]:
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

    except (
        ValueError,
        RuntimeError,
        pymupdf.FileDataError,
    ) as error:

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
# Process Chat Question
# --------------------------------------------------

def process_question(
    question: str,
    source: str | None,
):

    with data_lock:

        return answer_question(
            question,
            retriever,
            source=source,
        )


# --------------------------------------------------
# Chat Endpoint
# --------------------------------------------------

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
