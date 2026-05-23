import hashlib
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from config import get_settings
from db import find_document_by_checksum, init_db, insert_chunks, insert_document
from embeddings import embed_texts, get_embedding_dimension


def file_checksum(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(file_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(file_path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"page_number": index, "text": text})
    return pages


def split_page_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            last_sentence = max(chunk.rfind(". "), chunk.rfind("\n"))
            if last_sentence > chunk_size * 0.55:
                end = start + last_sentence + 1
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)

    return chunks


def build_chunks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settings = get_settings()
    chunks = []

    for page in pages:
        for text in split_page_text(
            page["text"],
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ):
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "page_number": page["page_number"],
                    "text": text,
                    "metadata": {"page_number": page["page_number"]},
                }
            )

    return chunks


def infer_collection(file_path: Path) -> str:
    name = file_path.name.lower()
    if "monetary" in name or "mps" in name:
        return "mps"
    if "financial stability" in name or "fsr" in name:
        return "financial_stability"
    if "annual" in name:
        return "annual_report"
    return "rbnz"


def ingest_document(
    file_path: str | Path,
    *,
    source: str = "rbnz",
    collection: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path}")

    embedding_dimension = get_embedding_dimension()
    init_db(embedding_dimension)

    checksum = file_checksum(path)
    existing = find_document_by_checksum(checksum)
    if existing:
        return {
            "status": "skipped",
            "reason": "document already ingested",
            "document_id": existing["id"],
            "title": existing["title"],
        }

    pages = extract_pdf_pages(path)
    if not pages:
        raise ValueError(f"No extractable text found in PDF: {path}")

    chunks = build_chunks(pages)
    if not chunks:
        raise ValueError(f"No chunks generated from PDF: {path}")

    document_title = title or path.stem
    embeddings = embed_texts([chunk["text"] for chunk in chunks], title=document_title)
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk["embedding"] = embedding

    document_id = insert_document(
        title=document_title,
        source=source,
        collection=collection or infer_collection(path),
        file_path=str(path),
        checksum=checksum,
        page_count=len(pages),
        metadata={
            "original_file_name": path.name,
            "embedding_provider": get_settings().embedding_provider,
            "embedding_model": (
                get_settings().gemini_embedding_model
                if get_settings().embedding_provider.lower() == "gemini"
                else get_settings().embedding_model_name
            ),
        },
    )
    insert_chunks(document_id, chunks)

    return {
        "status": "ingested",
        "document_id": document_id,
        "title": document_title,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "collection": collection or infer_collection(path),
    }
