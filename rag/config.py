from pathlib import Path
from functools import lru_cache


#use pydantic settings to manage configuration, with support for environment variables and .env file
from pydantic_settings import BaseSettings, SettingsConfigDict


#define the directory of the RAG project, which can be used to locate the .env file and other resources
RAG_DIR = Path(__file__).resolve().parent

#define the Settings class to hold all configuration parameters for the RAG system, with default values and support for environment variables
class Settings(BaseSettings):
    database_url: str = "postgresql://raguser:ragpass@localhost:5432/ragdb"
    embedding_provider: str = "local"
    #use a smaller embedding model for local testing, and a more powerful one for production
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_embedding_dimension: int = 768
    gemini_embedding_batch_size: int = 16
    documents_dir: Path = Path("documents")
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    #weight given to keyword/full-text matching vs vector similarity when reranking; 0 = pure vector search
    retrieval_lexical_weight: float = 0.3

    llm_provider: str = "none"
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 2.0
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
