from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.ingestion import load_and_chunk

PERSIST_DIR = "./data/chroma_db"
COLLECTION_NAME = "banking_faq"


def get_vectorstore(
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def index_documents(vector_store: Chroma, data_dir: str = "sample_data") -> int:
    existing_count = vector_store._collection.count()
    if existing_count > 0:
        print(
            f"Collection already has {existing_count} vectors — skipping re-index. "
            f"Delete {PERSIST_DIR} first if you want to rebuild from scratch."
        )
        return existing_count

    chunks = load_and_chunk(data_dir=data_dir)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)
    return len(chunks)


if __name__ == "__main__":
    vector_store = get_vectorstore()
    index_documents(vector_store)
    print(f"Total vectors in collection: {vector_store._collection.count()}")

    query = "what is the daily withdrawal limit"
    results = vector_store.similarity_search_with_score(query, k=2)

    print(f"\nQuery: '{query}'\n")
    for doc, score in results:
        print(f"[Score: {score:.4f}]")
        print(f"Metadata: {doc.metadata}")
        print(doc.page_content)
        print("-" * 60)
