"""
Module de retrieval pour le système RAG juridique.
Recherche hybride (dense + sparse) avec reranking CrossEncoder
et extraction des documents parents.
"""

from typing import List
from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.core.logging import get_logger, log_execution_time

logger = get_logger(__name__)

_cross_encoder = None
_compressor = None


def _get_reranker():
    global _cross_encoder, _compressor
    if _compressor is None:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

        logger.info(f"Chargement du reranker : {settings.RERANKER_MODEL_NAME}")
        _cross_encoder = HuggingFaceCrossEncoder(
            model_name=settings.RERANKER_MODEL_NAME
        )
        _compressor = CrossEncoderReranker(
            model=_cross_encoder, top_n=settings.RERANKER_TOP_N
        )
    return _compressor


class LegalRetriever:

    def __init__(self):
        from app.rag.vectorstore import get_vector_store
        self.vector_store = get_vector_store()
        self.compressor = _get_reranker()

    @log_execution_time(logger)
    def retrieve(self, queries: List[str], doc_type_filter: str = "all") -> list:
        qdrant_filter = None
        if doc_type_filter != "all":
            qdrant_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.doc_type",
                        match=qmodels.MatchValue(value=doc_type_filter),
                    )
                ]
            )
            logger.info(f"Cible de recherche (Filtre) : {doc_type_filter}")

        all_docs = []
        for q in queries:
            initial_docs = self.vector_store.similarity_search(
                query=q,
                k=settings.HYBRID_SEARCH_K,
                filter=qdrant_filter,
            )
            if initial_docs:
                reranked_docs = self.compressor.compress_documents(initial_docs, q)
                all_docs.extend(reranked_docs)
                logger.info(
                    f"Requête '{q[:50]}...' → {len(initial_docs)} résultats → "
                    f"{len(reranked_docs)} après reranking"
                )

        unique_parents = {}
        for doc in all_docs:
            parent_content = doc.metadata.get(
                "formatted_parent_content", doc.page_content
            )
            if parent_content not in unique_parents:
                doc.page_content = parent_content
                unique_parents[parent_content] = doc

        top_docs = list(unique_parents.values())[: settings.FINAL_TOP_K]
        logger.info(
            f"Recherche terminée : {len(all_docs)} total → "
            f"{len(unique_parents)} uniques → {len(top_docs)} sélectionnés"
        )
        return top_docs
