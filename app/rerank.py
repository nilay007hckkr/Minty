from functools import cache
from typing import List
from langchain_core.documents import Document


@cache
def _get_cross_encoder():
    # Imported lazily: sentence_transformers pulls in torch, which is slow to
    # import and not needed by code paths (or tests) that never rerank.
    from sentence_transformers import CrossEncoder

    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(
    query: str, docs: List[Document], top_n: int = 3
) -> List[Document]:
    if not docs:
        return []

    pairs = [[query, doc.page_content] for doc in docs]
    scores = _get_cross_encoder().predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_n]]
