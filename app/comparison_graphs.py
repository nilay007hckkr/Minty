from langgraph.graph import StateGraph, END
import dotenv
from app.state import GraphState
from app.nodes import retrieve_node, rerank_node, generate_node

dotenv.load_dotenv()

def build_naive_graph():
    """retrieve -> generate only. No grading, no reranking, no validation."""
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def build_reranked_graph():
    """retrieve -> rerank -> generate. Adds reranking, still no grading/validation."""
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


naive_graph = build_naive_graph()
reranked_graph = build_reranked_graph()
