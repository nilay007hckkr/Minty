import json
import re
import unicodedata
from dotenv import load_dotenv

load_dotenv()

from app.comparison_graphs import naive_graph, reranked_graph
from app.graph import app_graph, RECURSION_LIMIT

EVAL_SET_PATH = "tests/eval_set.json"
RESULTS_PATH = "tests/comparison_results.json"


def strip_markdown(text: str) -> str:
    return re.sub(r"\*\*|\*|_", "", text)


def normalize(text: str) -> str:
    # gpt-oss emits typographic characters (U+202F narrow no-break space in
    # "4:00 PM", U+2011 non-breaking hyphen in "fee-free") that would make
    # plain keyword matching fail on correct answers.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return strip_markdown(text).lower()


def run_graph(graph, query: str) -> dict:
    initial_state = {
        "original_query": query,
        "query": query,
        "refinement_count": 0,
    }
    result = graph.invoke(initial_state, config={"recursion_limit": RECURSION_LIMIT})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }


# Phrases that signal the answer is admitting the KB doesn't cover the
# question. Phrase matching can be fooled: an answer can admit a gap and then
# conclude from the absence anyway ("doesn't mention a fee, so there is no
# fee"). In the first manual review, 3 of 36 hard-negative answers did that.
# So hard-negative answers are saved for manual review.
GAP_PHRASES = [
    "not specified", "doesn't specify", "does not specify", "isn't specified",
    "not mentioned", "doesn't mention", "does not mention", "isn't mentioned",
    "not stated", "doesn't state", "does not state",
    "not provided", "doesn't provide", "does not provide",
    "not included", "doesn't include", "does not include",
    "not covered", "doesn't cover", "does not cover", "isn't covered", "is not covered",
    "isn't included", "is not included", "isn't in the", "not indicated",
    "not listed", "isn't listed", "doesn't list", "does not list", "not available",
    "doesn't say", "does not say", "no information", "not enough information",
    "only covers", "only mentions", "only lists", "only provides",
    "don't have", "do not have", "can't find", "cannot find", "unable to find",
]


def hard_negative_outcome(answer: str, sources: list) -> str:
    """refused: fell back with no sources. acknowledged: answered but said the
    KB doesn't cover it. answered: neither, i.e. likely asserted or misapplied
    a fact the KB doesn't contain."""
    if not sources:
        return "refused"
    clean = normalize(answer).replace("’", "'")
    if any(phrase in clean for phrase in GAP_PHRASES):
        return "acknowledged"
    return "answered"


def score_case(case: dict, result: dict) -> tuple[bool, str]:
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    clean_answer = normalize(answer)

    if case["expect"] == "hard_negative":
        outcome = hard_negative_outcome(answer, sources)
        passed = outcome != "answered"
        reason = outcome if passed else f"answered as if covered (trap: {case['trap']})"
    elif case["expect"] == "fallback":
        passed = len(sources) == 0
        reason = (
            "correctly fell back"
            if passed
            else f"expected fallback, got sources: {sources}"
        )
    else:
        has_sources = len(sources) > 0
        keyword_hit = any(normalize(kw) in clean_answer for kw in case["keywords"])
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

    categories = sorted({c["expect"] for c in cases})
    details = {name: [] for name in systems}

    for case in cases:
        for name, graph in systems.items():
            try:
                result = run_graph(graph, case["query"])
                passed, reason = score_case(case, result)
            except Exception as e:
                result, passed, reason = {"answer": "", "sources": []}, False, f"crashed: {e}"
            details[name].append({**case, **result, "passed": passed, "reason": reason})

    total = len(cases)
    print(f"\n{'='*70}")
    print(f"Baseline comparison — {total} queries each\n")
    header = f"{'System':<48}{'Total':>8}" + "".join(f"{c:>15}" for c in categories)
    print(header)
    for name, rows in details.items():
        cells = []
        for cat in categories:
            in_cat = [r for r in rows if r["expect"] == cat]
            cells.append(f"{sum(r['passed'] for r in in_cat)}/{len(in_cat)}")
        score = f"{sum(r['passed'] for r in rows)}/{total}"
        print(f"{name:<48}{score:>8}" + "".join(f"{c:>15}" for c in cells))

    hard = [c for c in cases if c["expect"] == "hard_negative"]
    if hard:
        print("\nHard negatives by outcome (refused / acknowledged / answered):")
        for name, rows in details.items():
            outcomes = [r["reason"].split(" ")[0] for r in rows if r["expect"] == "hard_negative"]
            counts = [outcomes.count(o) for o in ("refused", "acknowledged", "answered")]
            print(f"  {name:<46} {counts[0]} / {counts[1]} / {counts[2]}")
    print(f"{'='*70}\n")

    for name, rows in details.items():
        print(f"--- {name} ---")
        for r in rows:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] ({r['expect']}) {r['query']} — {r['reason']}")
        print()

    # Full answers for manual review (gitignored). Hard-negative "acknowledged"
    # answers in particular should be read: phrase matching can't tell an
    # honest gap from a gap admission next to an invented figure.
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"Full answers written to {RESULTS_PATH}")


if __name__ == "__main__":
    run_comparison()
