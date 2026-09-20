import json
import re
from dotenv import load_dotenv

load_dotenv()

from app.comparison_graphs import naive_graph, reranked_graph
from app.graph import app_graph

EVAL_SET_PATH = "tests/eval_set.json"


def strip_markdown(text: str) -> str:
    return re.sub(r"\*\*|\*|_", "", text)


def run_graph(graph, query: str) -> dict:
    initial_state = {
        "original_query": query,
        "query": query,
        "refinement_count": 0,
    }
    result = graph.invoke(initial_state, config={"recursion_limit": 15})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }


def score_case(case: dict, result: dict) -> tuple[bool, str]:
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    clean_answer = strip_markdown(answer)

    if case["expect"] == "fallback":
        passed = len(sources) == 0
        reason = (
            "correctly fell back"
            if passed
            else f"expected fallback, got sources: {sources}"
        )
    else:
        has_sources = len(sources) > 0
        keyword_hit = any(kw.lower() in clean_answer.lower() for kw in case["keywords"])
        passed = has_sources and keyword_hit
        if not has_sources:
            reason = "no sources returned"
        elif not keyword_hit:
            reason = f"none of {case['keywords']} found in answer"
        else:
            reason = "correct"

    return passed, reason


def run_comparison():
    with open(EVAL_SET_PATH) as f:
        cases = json.load(f)

    systems = {
        "Naive RAG (retrieve -> generate)": naive_graph,
        "Reranked RAG (retrieve -> rerank -> generate)": reranked_graph,
        "Minty (full pipeline)": app_graph,
    }

    scores = {name: 0 for name in systems}
    details = {name: [] for name in systems}

    for case in cases:
        for name, graph in systems.items():
            try:
                result = run_graph(graph, case["query"])
                passed, reason = score_case(case, result)
            except Exception as e:
                passed, reason = False, f"crashed: {e}"

            if passed:
                scores[name] += 1
            details[name].append((case["query"], passed, reason))

    total = len(cases)
    print(f"\n{'='*70}")
    print(f"Baseline comparison — {total} queries each\n")
    for name, score in scores.items():
        print(f"{name}: {score}/{total}")
    print(f"{'='*70}\n")

    for name, rows in details.items():
        print(f"--- {name} ---")
        for query, passed, reason in rows:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {query} — {reason}")
        print()


if __name__ == "__main__":
    run_comparison()
