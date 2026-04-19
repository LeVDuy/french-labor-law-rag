"""
Pipeline d'ingestion des données juridiques.
Lecture des fichiers Markdown, découpage, vectorisation et indexation dans Qdrant.

Lancer avec :
    python -m app.rag.ingest
"""

from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger, log_execution_time, setup_logging

setup_logging()
logger = get_logger(__name__)


def load_documents(source_dir: Path = None) -> list:
    source_dir = source_dir or settings.DATA_PROCESSED_DIR
    md_files = sorted(source_dir.glob("**/*.md"))

    documents = []
    for file_path in md_files:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append((path, content))
        logger.info(f"Lecture du fichier : {path.name} ({len(content)} chars)")

    logger.info(f"Total documents chargés : {len(documents)}")
    return documents


def chunk_documents(documents: list) -> list:
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    headers = [
        ("#", "Livre"),
        ("##", "Partie"),
        ("###", "Titre"),
        ("####", "Section"),
        ("#####", "Article"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHILD_CHUNK_SIZE,
        chunk_overlap=settings.CHILD_CHUNK_OVERLAP,
    )

    all_chunks = []

    for file_path, content in documents:
        parent_folder_name = file_path.parent.name
        source_name = file_path.stem

        header_docs = md_splitter.split_text(content)
        for hd in header_docs:
            hd.metadata["raw_parent_content"] = hd.page_content

        chunks = text_splitter.split_documents(header_docs)

        for c in chunks:
            c.metadata["source"] = file_path.name
            c.metadata["doc_type"] = parent_folder_name

            article_title = c.metadata.get("Article", "")
            raw_parent = c.metadata.get("raw_parent_content", c.page_content)

            if article_title:
                c.page_content = (
                    f"Source : {source_name}\n"
                    f"Article : {article_title}\n"
                    f"Contenu : {c.page_content}"
                )
                c.metadata["formatted_parent_content"] = (
                    f"Source : {source_name}\n"
                    f"Article : {article_title}\n"
                    f"Contenu : {raw_parent}"
                )
            else:
                c.page_content = f"Source : {source_name}\nContenu : {c.page_content}"
                c.metadata["formatted_parent_content"] = (
                    f"Source : {source_name}\nContenu : {raw_parent}"
                )

            if "raw_parent_content" in c.metadata:
                del c.metadata["raw_parent_content"]

            all_chunks.append(c)

        logger.info(f"Découpage : {file_path.name} → {len(chunks)} segments")

    logger.info(f"Total segments créés : {len(all_chunks)}")
    return all_chunks


@log_execution_time(logger)
def embed_and_index(chunks: list) -> None:
    from app.rag.vectorstore import get_qdrant_client, get_vector_store, reset_vector_store

    client = get_qdrant_client()

    if client.collection_exists(settings.COLLECTION_NAME):
        client.delete_collection(settings.COLLECTION_NAME)
        logger.info(f"Collection supprimée : {settings.COLLECTION_NAME}")

    reset_vector_store()

    logger.info(f"Vectorisation et stockage de {len(chunks)} segments dans Qdrant...")
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    logger.info("Indexation terminée.")


@log_execution_time(logger)
def run_ingestion() -> None:
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DU PIPELINE D'INGESTION")
    logger.info("=" * 60)

    documents = load_documents()
    if not documents:
        logger.warning("Aucun document trouvé. Vérifiez le chemin DATA_PROCESSED_DIR.")
        return

    chunks = chunk_documents(documents)
    if not chunks:
        logger.warning("Aucun segment produit. Vérifiez le format des documents.")
        return

    embed_and_index(chunks)

    logger.info("=" * 60)
    logger.info("INGESTION TERMINÉE — LA BASE DE DONNÉES EST PRÊTE !")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()
