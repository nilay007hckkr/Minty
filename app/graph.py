from dotenv import load_dotenv

load_dotenv()

import logging
from langgraph.graph import StateGraph, END

from app.state import GraphState
from app.nodes import (
    classify_node,
    retrieve_node,
    rerank_node,
    grade_node,
    refine_node,
    generate_node,
    validate_node,
    fallback_node,
)


MAX_REFINEMENTS = 2
MAX_GENERATIONS = 2
# Worst case: classify + 3x(retrieve, rerank, grade) + 2 refine
# + 2x(generate, validate) + fallback = 17 steps.
RECURSION_LIMIT = 25


def route_after_classification(state: GraphState) -> str:
    return "retrieve" if state.get("classification") == "in_scope" else "fallback"


def route_after_grading(state: GraphState) -> str:
    filtered_docs = state.get("documents", [])
    count = state.get("refinement_count", 0)

    if filtered_docs:
        return "generate"
    elif count < MAX_REFINEMENTS:
        return "refine"
    else:
        return "fallback"


def route_after_validation(state: GraphState) -> str:
    if state.get("is_grounded", False):
        return END
    # Retry once with the validator's feedback, but only when it named
    # specific claims; an API failure or bare "no" has nothing to fix.
    if (
        state.get("unsupported_claims")
        and state.get("generation_attempts", 0) < MAX_GENERATIONS
    ):
        return "generate"
    return "fallback"


workflow = StateGraph(GraphState)

workflow.add_node("classify", classify_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("grade", grade_node)
workflow.add_node("refine", refine_node)
workflow.add_node("generate", generate_node)
workflow.add_node("validate", validate_node)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    route_after_classification,
    {"retrieve": "retrieve", "fallback": "fallback"},
)

workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "grade")

workflow.add_conditional_edges(
    "grade",
    route_after_grading,
    {"generate": "generate", "refine": "refine", "fallback": "fallback"},
)

workflow.add_edge("refine", "retrieve")
workflow.add_edge("generate", "validate")

workflow.add_conditional_edges(
    "validate",
    route_after_validation,
    {END: END, "generate": "generate", "fallback": "fallback"},
)

workflow.add_edge("fallback", END)

app_graph = workflow.compile()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)s: %(message)s", force=True
    )
    while True:
        user_input = input("\nEnter your question (or type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        initial_state: GraphState = {
            "original_query": user_input,
            "query": user_input,
            "refinement_count": 0,
        }

        result = app_graph.invoke(
            initial_state, config={"recursion_limit": RECURSION_LIMIT}
        )

        print(f"\nAnswer: {result.get('answer')}")
        if result.get("sources"):
            print(f"Sources: {result.get('sources')}")
