import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from config import get_settings
from db import get_index_summary
from ingest import ingest_document
from retrieval import retrieve_and_answer, retrieve_chunks

app = FastAPI(title="RBNZ Fintech RAG API")


class RetrieveRequest(BaseModel):
    query: str
    collection: str | None = None
    top_k: int | None = None
    answer: bool = True


def _pipeline_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _save_upload(file: UploadFile) -> Path:
    original_name = Path(file.filename or "upload.pdf").name
    if not original_name.lower().endswith(".pdf"):
        raise ValueError(f"Expected a PDF upload, got: {original_name}")

    settings = get_settings()
    uploads_dir = Path(settings.documents_dir) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    destination = uploads_dir / f"{uuid4().hex[:8]}_{original_name}"
    # Read directly from the underlying SpooledTemporaryFile since this is a sync
    # path function (FastAPI runs it in a threadpool); no `await file.read()` needed.
    with destination.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    return destination


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/index")
def index() -> dict:
    try:
        return get_index_summary()
    except Exception as exc:
        raise _pipeline_error(exc) from exc


@app.post("/ingest")
def ingest(
    file: UploadFile = File(...),
    collection: str | None = Form(None),
    source: str = Form("rbnz"),
    title: str | None = Form(None),
) -> dict:
    try:
        saved_path = _save_upload(file)
        return ingest_document(saved_path, source=source, collection=collection, title=title)
    except Exception as exc:
        raise _pipeline_error(exc) from exc


@app.post("/retrieve")
def retrieve(request: RetrieveRequest) -> dict:
    try:
        if request.answer:
            return retrieve_and_answer(
                request.query, collection=request.collection, top_k=request.top_k
            )
        chunks = retrieve_chunks(
            request.query, collection=request.collection, top_k=request.top_k
        )
        return {"query": request.query, "chunks": chunks}
    except Exception as exc:
        raise _pipeline_error(exc) from exc
