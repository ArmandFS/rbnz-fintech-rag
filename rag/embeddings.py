from functools import lru_cache

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import get_settings
from llm import post_with_retries


@lru_cache  # avoids reloading the model on every call once this runs inside a long-lived FastAPI process
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model_name)


def get_embedding_dimension() -> int:
    settings = get_settings()
    if settings.embedding_provider.lower() == "gemini":
        return settings.gemini_embedding_dimension
    return int(get_embedding_model().get_sentence_embedding_dimension())


def prepare_gemini_document(text: str, title: str | None = None) -> str:
    return f"title: {title or 'none'} | text: {text}"


def prepare_gemini_query(query: str) -> str:
    return f"task: question answering | query: {query}"


def embed_texts(texts: list[str], *, title: str | None = None) -> list[list[float]]:
    settings = get_settings()
    if settings.embedding_provider.lower() == "gemini":
        prepared_texts = [prepare_gemini_document(text, title=title) for text in texts]
        return embed_with_gemini(prepared_texts)

    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [vector.tolist() for vector in vectors]


def embed_query(query: str) -> list[float]:
    settings = get_settings()
    if settings.embedding_provider.lower() == "gemini":
        return embed_with_gemini([prepare_gemini_query(query)])[0]
    return embed_texts([query])[0]


def embed_with_gemini(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")

    vectors: list[list[float]] = []
    for batch in tqdm(
        list(batch_texts(texts, settings.gemini_embedding_batch_size)),
        desc="Embedding with Gemini",
    ):
        vectors.extend(embed_gemini_batch(batch))
    return vectors


def batch_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    return [texts[index : index + batch_size] for index in range(0, len(texts), batch_size)]


def embed_gemini_batch(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    model_path = f"models/{settings.gemini_embedding_model}"
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_embedding_model}:batchEmbedContents"
    )
    payload = {
        "requests": [
            {
                "model": model_path,
                "content": {"parts": [{"text": text}]},
                "output_dimensionality": settings.gemini_embedding_dimension,
            }
            for text in texts
        ]
    }
    response = post_with_retries(
        url,
        headers={"x-goog-api-key": settings.gemini_api_key},
        json=payload,
        timeout=90,
        error_message="Gemini embedding request failed",
    )
    data = response.json()
    return [embedding["values"] for embedding in data["embeddings"]]
