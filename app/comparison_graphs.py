from langgraph.graph import StateGraph, END

from app.state import GraphState
from app.nodes import retrieve_node, rerank_node, generate_node


def attach_sources_node(state: GraphState) -> dict:
    """Mirrors grade_node's source-extraction, without the relevance filtering
    these comparison graphs deliberately skip."""
    docs = state.get("documents", [])
    sources = list(set([doc.metadata.get("source", "sample_data") for doc in docs]))
    return {"sources": sources}


def build_naive_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("attach_sources", attach_sources_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "attach_sources")
    workflow.add_edge("attach_sources", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def build_reranked_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("attach_sources", attach_sources_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "attach_sources")
    workflow.add_edge("attach_sources", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


naive_graph = build_naive_graph()
reranked_graph = build_reranked_graph()
