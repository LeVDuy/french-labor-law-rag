"""
Initialisation et gestion du vector store Qdrant.
"""

import warnings
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

_vector_store = None
_dense_embeddings = None
_sparse_embeddings = None


def get_dense_embeddings():
    global _dense_embeddings
    if _dense_embeddings is None:
        import os
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info(f"Chargement des embeddings denses : {settings.DENSE_MODEL_NAME}")
        _dense_embeddings = HuggingFaceEmbeddings(
            model_name=settings.DENSE_MODEL_NAME,
            model_kwargs={"device": settings.DEVICE},
            encode_kwargs={"batch_size": settings.EMBEDDING_BATCH_SIZE},
        )
    return _dense_embeddings


def get_sparse_embeddings():
    global _sparse_embeddings
    if _sparse_embeddings is None:
        from langchain_qdrant import FastEmbedSparse
        logger.info(f"Chargement des embeddings sparse : {settings.SPARSE_MODEL_NAME}")
        _sparse_embeddings = FastEmbedSparse(model_name=settings.SPARSE_MODEL_NAME)
    return _sparse_embeddings


def get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=str(settings.QDRANT_URL))


def get_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    from qdrant_client.http import models as qmodels
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    dense_embeddings = get_dense_embeddings()
    sparse_embeddings = get_sparse_embeddings()
    client = get_qdrant_client()

    embedding_dimension = len(dense_embeddings.embed_query("test"))
    if not client.collection_exists(settings.COLLECTION_NAME):
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE,
            ),
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams()
            },
        )
        logger.info(f"Collection créée : {settings.COLLECTION_NAME}")

    _vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse",
    )
    logger.info(f"Vector store initialisé : {settings.COLLECTION_NAME}")
    return _vector_store


def reset_vector_store():
    global _vector_store
    _vector_store = None
