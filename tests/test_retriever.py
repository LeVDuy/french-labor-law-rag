"""
Tests du module de retrieval.

Lancer avec :
    pytest tests/test_retriever.py -v

Note : Nécessite Qdrant actif en local.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def test_dense_retrieval(query: str = "Quel est le préavis de démission pour un cadre ?"):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    print(f"\n🔍 Question : '{query}'\n" + "-" * 50)

    embeddings = HuggingFaceEmbeddings(model_name=settings.DENSE_MODEL_NAME)
    client = QdrantClient(url=settings.QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)

    for i, doc in enumerate(results):
        print(f"\nRÉSULTAT {i + 1} :")
        print(f"  SOURCE  : {doc.metadata.get('Livre', 'N/A')}")
        print(f"  SECTION : {doc.metadata.get('Section', 'N/A')}")
        print(f"  ARTICLE : {doc.metadata.get('Article', 'N/A')}")
        print(f"  CONTENU : {doc.page_content[:300]}...")

    return results


def test_hybrid_retrieval(query: str = "Quel est le préavis de démission pour un cadre ?"):
    from app.rag.retriever import LegalRetriever

    print(f"\n🔍 Recherche hybride : '{query}'\n" + "-" * 50)

    retriever = LegalRetriever()
    docs = retriever.retrieve([query])

    for i, doc in enumerate(docs):
        print(f"\nRÉSULTAT {i + 1} :")
        print(f"  SOURCE  : {doc.metadata.get('Livre', doc.metadata.get('source', 'N/A'))}")
        print(f"  ARTICLE : {doc.metadata.get('Article', 'N/A')}")
        print(f"  EXTRAIT : {doc.page_content[:200]}...")

    return docs


if __name__ == "__main__":
    test_dense_retrieval("Quel est le préavis de démission pour un cadre ?")
    test_dense_retrieval("What are the rules for part-time work?")
    print("\n" + "=" * 60)
    test_hybrid_retrieval("Quel est le préavis de démission pour un cadre Syntec ?")
