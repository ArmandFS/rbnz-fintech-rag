from typing import Any

from config import get_settings
from db import init_db, search_chunks
from embeddings import embed_query, get_embedding_dimension
from llm import answer_with_llm


def retrieve_chunks(
    query: str,
    collection: str | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    settings = get_settings()
    embedding_dimension = get_embedding_dimension()
    init_db(embedding_dimension)

    query_embedding = embed_query(query)
    return search_chunks(
        query_embedding=query_embedding,
        query_text=query,
        collection=collection,
        top_k=top_k or settings.retrieval_top_k,
        lexical_weight=settings.retrieval_lexical_weight,
    )


def format_context(chunks: list[dict[str, Any]]) -> str:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk["page_number"] or "unknown"
        context_blocks.append(
            f"[{index}] {chunk['title']} | page {page} | similarity {chunk['similarity']:.3f}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_blocks)


def retrieve_and_answer(
    query: str,
    collection: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    chunks = retrieve_chunks(query=query, collection=collection, top_k=top_k)
    context = format_context(chunks)
    answer = answer_with_llm(query=query, context=context)

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "index": index,
                "title": chunk["title"],
                "page_number": chunk["page_number"],
                "similarity": float(chunk["similarity"]),
                "file_path": chunk["file_path"],
            }
            for index, chunk in enumerate(chunks, start=1)
        ],
    }
