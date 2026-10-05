import json
from dotenv import load_dotenv

load_dotenv()

from app.graph import app_graph
from tests.run_comparison_eval import run_graph, score_case

EVAL_SET_PATH = "tests/eval_set.json"

# Invokes the graph directly rather than POSTing to /chat: the API's semantic
# cache is global with a 1h TTL, so repeated eval runs would score cached
# answers instead of the pipeline.


def run_eval():
    with open(EVAL_SET_PATH) as f:
        cases = json.load(f)

    results = []

    for case in cases:
        try:
            result = run_graph(app_graph, case["query"])
            passed, reason = score_case(case, result)
        except Exception as e:
            results.append({**case, "passed": False, "reason": f"crashed: {e}"})
            continue

        results.append(
            {**case, "passed": passed, "reason": reason, "answer": result["answer"]}
        )

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\n{'='*60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['query']}")
        print(f"       {r['reason']}")
    print(f"{'='*60}")
    print(f"\nScore: {passed_count}/{total}")

    with open("tests/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_eval()
