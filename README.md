# RBNZ Fintech RAG

A modular Retrieval-Augmented Generation (RAG) project for querying Reserve Bank of New Zealand (RBNZ) financial, monetary policy, and regulatory documents.

The project is being built with a **core-first** approach:

1. Python RAG pipeline
2. FastAPI RAG service
3. Node.js backend API
4. React + TypeScript frontend

The current repository contains the first milestone: a working Python-based ingestion and retrieval pipeline.

## What This Project Does

This project turns RBNZ PDF documents into a searchable knowledge base.

The current pipeline can:

- Read local RBNZ PDF files.
- Extract page-level text.
- Split documents into overlapping chunks.
- Generate embeddings for each chunk.
- Store document metadata and vectors in PostgreSQL with pgvector.
- Retrieve the most relevant chunks for a user question.
- Optionally send retrieved context to an LLM to generate a grounded answer.

Example questions this project is intended to support:

- "What is the OCR outlook?"
- "What risks does the RBNZ identify for financial stability?"
- "What does the annual report say about operational priorities?"
- "How has the RBNZ described inflation pressure?"

## Current Status

Implemented:

- Core Python RAG scripts.
- Local PDF ingestion.
- Chunking and embedding flow.
- PostgreSQL + pgvector storage.
- Semantic retrieval.
- Optional LLM answer generation.
- Gemini embedding support.
- Gemini, DeepSeek, and OpenRouter-compatible LLM hooks.
- CLI scripts for ingestion, querying, and index inspection.

Not implemented yet:

- FastAPI service wrapper.
- Node.js backend API.
- React frontend.
- Automated RBNZ document scraping/acquisition.
- Docker Compose for the full system.
- Deployment.

## Planned Tech Stack

### Core RAG Pipeline

- Python
- pypdf
- sentence-transformers
- Google Gemini embeddings
- PostgreSQL
- pgvector
- psycopg2
- httpx
- pydantic-settings

### LLM Layer

Planned/default:

- Google Gemini Flash-Lite

Supported hooks:

- Gemini
- DeepSeek
- OpenRouter-compatible models, including Qwen/Kimi-style workflows

### Backend API

Planned:

- FastAPI for exposing the Python RAG pipeline
- Node.js + Express as the application API layer
- Redis for optional response caching

### Frontend

Planned:

- React
- TypeScript
- Vite

### Infrastructure

Planned:

- Docker
- Docker Compose
- PostgreSQL with pgvector
- Future AWS deployment

## Repository Structure

```text
.
├── README.md
├── rag/
│   ├── config.py
│   ├── db.py
│   ├── embeddings.py
│   ├── ingest.py
│   ├── llm.py
│   ├── retrieval.py
│   ├── requirements.txt
│   ├── .env.example
│   └── scripts/
│       ├── ingest_local.py
│       ├── query_local.py
│       └── show_index.py
└── .gitignore
```

Local-only files and folders are ignored:

```text
rag/.env
rag/documents/
rag/venv/
__pycache__/
.DS_Store
```

## How The RAG Pipeline Works

### 1. Ingestion

`rag/ingest.py` handles PDF ingestion.

It:

1. Opens a PDF.
2. Extracts text from each page.
3. Cleans the text.
4. Splits it into overlapping chunks.
5. Embeds each chunk.
6. Stores the document and chunk records in PostgreSQL.

Duplicate documents are skipped using a SHA-256 checksum.

### 2. Embeddings

`rag/embeddings.py` supports two embedding modes:

```text
local
gemini
```

Local mode uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Gemini mode uses:

```text
gemini-embedding-2
```

Important: if you change embedding models, reset the database tables and re-ingest the documents. Vectors from different embedding models should not be mixed.

### 3. Storage

`rag/db.py` creates and manages two core tables:

```text
documents
chunks
```

The `chunks` table stores the vector embedding in a pgvector column.

### 4. Retrieval

`rag/retrieval.py` embeds the user query, searches pgvector for similar chunks, and returns the top matches.

It can either:

- return retrieved chunks only, or
- pass the retrieved context to an LLM for answer generation.

### 5. LLM Answering

`rag/llm.py` supports:

```text
none
gemini
deepseek
openrouter
```

For early development, retrieval can be tested with `LLM_PROVIDER=none`.

## Usage

Run commands from the `rag/` directory.

### Ingest A PDF

```bash
venv/bin/python scripts/ingest_local.py "documents/MPS_Report_Feb2026.pdf" --collection mps
```

Example collections:

```text
mps
financial_stability
annual_report
```

### Show Index Status

```bash
venv/bin/python scripts/show_index.py
```

### Retrieve Chunks Only

```bash
venv/bin/python scripts/query_local.py "What is the OCR outlook?" --chunks-only
```

### Retrieve And Generate An Answer

```bash
venv/bin/python scripts/query_local.py "What is the OCR outlook?"
```

This requires `LLM_PROVIDER` and the relevant API key to be configured.

If Gemini returns a temporary `503 Service Unavailable` or similar transient error, the LLM layer retries with exponential backoff before failing.

## Resetting The Index

Reset the database when changing embedding models.

```bash
docker exec -it pgvector-dev psql -U raguser -d ragdb
```

Then:

```sql
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS documents;
```

After resetting, re-ingest all documents.

## Use Cases

This project is designed for learning and prototyping fintech RAG workflows.

Possible use cases:

- Querying central bank publications.
- Summarizing RBNZ monetary policy documents.
- Comparing financial stability risks across reports.
- Building a regulatory research assistant.
- Creating a fintech document intelligence prototype.
- Learning how vector databases, embeddings, and LLMs fit together in a real application.

## Development Roadmap

### Phase 1: Core RAG Pipeline

Status: currently in progress.

Current focus:

- Improve retrieval quality.
- Test Gemini embeddings.
- Test Gemini-generated answers.
- Improve source citation formatting.
- Add basic evaluation questions.

### Phase 2: FastAPI RAG Service

Planned:

- Add `rag/main.py`.
- Expose `/ingest`.
- Expose `/retrieve`.
- Return structured JSON answers and source citations.

### Phase 3: Node.js API

Planned:

- Add Express API.
- Proxy upload/query requests to the FastAPI RAG service.
- Add validation, rate limiting, and optional caching.

### Phase 4: Frontend UI

Planned:

- Add React + TypeScript frontend.
- Upload PDFs.
- Ask questions.
- Display generated answers and cited sources.

### Phase 5: Docker Compose

Planned:

- Run Python RAG service, Node API, frontend, pgvector, and Redis together.

### Phase 6: Deployment

Planned:

- Deploy after local Docker Compose flow is stable.

## Notes

This repository does not include RBNZ PDFs, API keys, local environment files, or vector database data.

The current implementation is intentionally explicit and modular so the core RAG mechanics are easy to understand before adding API and frontend layers.
