"""
Configuration centralisée de l'application RAG.
Toutes les variables sont chargées depuis le fichier .env.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).parent.parent.parent.absolute()


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    BASE_DIR: Path = _BASE_DIR
    DATA_RAW_DIR: Path = _BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = _BASE_DIR / "data" / "processed"

    QDRANT_URL: str = "http://localhost:6333"
    COLLECTION_NAME: str = "french_law_hybrid"

    DENSE_MODEL_NAME: str = "BAAI/bge-m3"
    SPARSE_MODEL_NAME: str = "Qdrant/bm25"

    RERANKER_MODEL_NAME: str = "antoinelouis/crossencoder-camembert-base-mmarcoFR"

    CHILD_CHUNK_SIZE: int = 1500
    CHILD_CHUNK_OVERLAP: int = 250

    LLM_BASE_URL: str = "http://localhost:1234/v1"
    LLM_API_KEY: str = "lm-studio"
    LLM_MODEL: str = "local-model"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1500

    HYBRID_SEARCH_K: int = 60
    RERANKER_TOP_N: int = 10
    FINAL_TOP_K: int = 5

    GROQ_API_KEY: str = ""

    LEGIFRANCE_CLIENT_ID: str = ""
    LEGIFRANCE_CLIENT_SECRET: str = ""

    DEVICE: str = "mps"
    EMBEDDING_BATCH_SIZE: int = 8

    def model_post_init(self, __context):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


settings = Settings()
