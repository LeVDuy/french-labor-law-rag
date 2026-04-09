import os
import glob
import warnings
from pathlib import Path
import configs
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def get_vector_base():
    print("Initialisation du système de recherche vectorielle (Hybrid)...")
    
    dense_embeddings = HuggingFaceEmbeddings(model_name=configs.DENSE_MODEL_NAME)
    sparse_embeddings = FastEmbedSparse(model_name=configs.SPARSE_MODEL_NAME)
    embedding_dimension = len(dense_embeddings.embed_query("test"))
    client = QdrantClient(url=str(configs.QDRANT_URL))

    if not client.collection_exists(configs.COLLECTION_NAME):
        client.create_collection(
            collection_name=configs.COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE
            ),
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams()
            }
        )
        print(f"Collection créée : {configs.COLLECTION_NAME}")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=configs.COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse"
    )
    return vector_store

def run_ingestion():
    print("\nDÉMARRAGE DU SYSTÈME D’INGESTION ET DE VECTORISATION...\n")

    headers = [
        ("#", "Livre"), 
        ("##", "Partie"), 
        ("###", "Titre"), 
        ("####", "Section"),
        ("#####", "Article")
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=configs.CHILD_CHUNK_SIZE, 
        chunk_overlap=configs.CHILD_CHUNK_OVERLAP 
    )
    all_chunks = []
    md_files = sorted(glob.glob(os.path.join(configs.MARKDOWN_DIR, "**", "*.md"), recursive=True))

    for file_path in md_files:
        path = Path(file_path)
        print(f"Lecture et découpage du fichier : {path.name}")
        parent_folder_name = path.parent.name
        
        with open(path, "r", encoding="utf-8") as f:
            md_text = f.read()
 
        header_docs = md_splitter.split_text(md_text)
        chunks = text_splitter.split_documents(header_docs)
        for c in chunks:
            c.metadata["source"] = path.name
            c.metadata["doc_type"] = parent_folder_name         
            article_title = c.metadata.get("Article", "")
            if article_title:
                c.page_content = f"Article : {article_title}\nContenu : {c.page_content}"
            
            all_chunks.append(c)
    client = QdrantClient(url=str(configs.QDRANT_URL))
    if client.collection_exists(configs.COLLECTION_NAME):
        client.delete_collection(configs.COLLECTION_NAME)
    
    print(f"Vectorisation et stockage de {len(all_chunks)} segments dans Qdrant...")
    vector_store = get_vector_base()
    vector_store.add_documents(all_chunks)
    print("\nINGESTION TERMINÉE. LA BASE DE DONNÉES EST PRÊTE !")

if __name__ == "__main__":
    run_ingestion()