from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


RAG_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    database_url: str = "postgresql://raguser:ragpass@localhost:5432/ragdb"
    embedding_provider: str = "local"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_embedding_dimension: int = 768
    gemini_embedding_batch_size: int = 16
    documents_dir: Path = Path("documents")
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_top_k: int = 5

    llm_provider: str = "none"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"

    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"

    openrouter_api_key: str | None = None
    openrouter_model: str = "qwen/qwen-2.5-72b-instruct"

    model_config = SettingsConfigDict(
        env_file=RAG_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
