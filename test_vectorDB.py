import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import configs

warnings.filterwarnings("ignore")

def test_retriever(query: str):
    print(f"\n Question: '{query}'\n" + "-"*50)
    embeddings = HuggingFaceEmbeddings(model_name=configs.DENSE_MODEL_NAME)
    client = QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name= configs.COLLECTION_NAME, 
        embedding=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    results = retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"\nRÉSULTAT {i+1}:")
        print(f"SOURCE: {doc.metadata.get('Livre', 'N/A')}")
        print(f"SECTION: {doc.metadata.get('Section', 'N/A')}")
        print(f"ARTICLE: {doc.metadata.get('Article', 'N/A')}")
        print(f"CONTENU:\n{doc.page_content[:300]}...")

if __name__ == "__main__":
    test_retriever("Quel est le préavis de démission pour un cadre ?")
    test_retriever("What are the rules for part-time work?")