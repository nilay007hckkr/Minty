from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    original_query: str
    query: str
    documents: List[Document]
    sources: List[str]
    answer: str
    refinement_count: int
    is_grounded: str
    classification: str
