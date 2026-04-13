import os
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = Path(__file__).parent.absolute()
MARKDOWN_DIR = BASE_DIR / "data" / "markdown_files"

PARENT_STORE_PATH = BASE_DIR / "data" / "parent_store"
PARENT_STORE_PATH.mkdir(parents=True, exist_ok=True)

# VECTOR DATABASE (QDRANT) CONFIG
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "french_law_hybrid"

# EMBEDDINGS CONFIG
DENSE_MODEL_NAME = "BAAI/bge-m3"
SPARSE_MODEL_NAME = "Qdrant/bm25"

# RERANKER CONFIG
# RERANKER_MODEL_NAME = "unicamp-dl/mMiniLM-L6-v2-mmarco-v2"
RERANKER_MODEL_NAME = "antoinelouis/crossencoder-camembert-base-mmarcoFR"

# CHUNKING CONFIG
CHILD_CHUNK_SIZE = 1500
CHILD_CHUNK_OVERLAP = 250

# LLM CONFIG
LLM_PROVIDER = "lmstudio"  
LLM_TEMPERATURE = 0.0
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "local-model"

GROQ_API_KEY = "your_groq_api_key"