from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    original_query: str
    query: str
    documents: List[Document]
    sources: List[str]
    answer: str
    refinement_count: int
    generation_attempts: int
    is_grounded: bool
    unsupported_claims: List[str]
    classification: str
    upstream_error: bool
