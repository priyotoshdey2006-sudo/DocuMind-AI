
import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from pathlib import Path

import pymupdf



def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    """Split text into overlapping chunks, preferring word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be nonnegative and smaller than chunk_size")

    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start + 1, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Move forward while keeping the requested overlap.
        start = max(end - overlap, start + 1)

    return chunks


def ingest_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract page text and return chunks with citation metadata."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    records = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_text = page.get_text("text")

            for chunk_index, chunk in enumerate(split_text(page_text)):
                records.append({
                    "id": f"{pdf_path.stem}-p{page_index + 1}-c{chunk_index + 1}",
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_index + 1,
                    },
                })

    if not records:
        raise ValueError(
            "No text found. This may be a scanned PDF that needs OCR."
        )

    return records