from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(
    query: str, docs: List[Document], top_n: int = 3
) -> List[Document]:
    if not docs:
        return []

    pairs = [[query, doc.page_content] for doc in docs]
    scores = _cross_encoder.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_n]]
