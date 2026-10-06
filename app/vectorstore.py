import logging
from functools import cache
from typing import NamedTuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

PERSIST_DIR = "./data/chroma_db"
COLLECTION_NAME = "banking_faq"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class IndexSync(NamedTuple):
    total: int
    added: int
    removed: int

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


@cache
def get_embeddings() -> HuggingFaceEmbeddings:
    # Shared by the vector store and the semantic cache, so the model is
    # loaded once per process, and only when first needed.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@cache
def get_vectorstore(
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )


def index_documents(vector_store: Chroma, data_dir: str = "sample_data") -> IndexSync:
    """Syncs the collection with data_dir: adds new chunks, deletes stale ones.
    Chunk ids are content hashes, so an unchanged corpus is a no-op and an
    edited file or chunking change is picked up without deleting the DB."""
    # Lazy: langchain_text_splitters' package init imports torch-based
    # splitters (~30s cold), which only indexing needs.
    from app.ingestion import load_and_chunk

    chunks = {c.metadata["chunk_id"]: c for c in load_and_chunk(data_dir=data_dir)}
    existing = set(vector_store.get(include=[])["ids"])

    stale = existing - chunks.keys()
    new_ids = [chunk_id for chunk_id in chunks if chunk_id not in existing]

    if stale:
        vector_store.delete(ids=list(stale))
    if new_ids:
        vector_store.add_documents(
            documents=[chunks[chunk_id] for chunk_id in new_ids], ids=new_ids
        )

    logger.info(
        f"Index sync: {len(new_ids)} added, {len(stale)} removed, {len(chunks)} total"
    )
    return IndexSync(total=len(chunks), added=len(new_ids), removed=len(stale))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    vector_store = get_vectorstore()
    sync = index_documents(vector_store)
    if sync.changed:
        # A running server's cached answers may quote the old content.
        from app.cache import clear_cache

        print(f"Content changed: cleared {clear_cache()} cached answers")
    print(f"Total vectors in collection: {vector_store._collection.count()}")

    query = "what is the daily withdrawal limit"
    results = vector_store.similarity_search_with_score(query, k=2)

    print(f"\nQuery: '{query}'\n")
    for doc, score in results:
        print(f"[Score: {score:.4f}]")
        print(f"Metadata: {doc.metadata}")
        print(doc.page_content)
        print("-" * 60)
