import json
import re
import time
import requests

API_URL = "http://localhost:8000/chat"
EVAL_SET_PATH = "tests/eval_set.json"


def strip_markdown(text: str) -> str:
    return re.sub(r"\*\*|\*|_", "", text)


def run_eval():
    with open(EVAL_SET_PATH) as f:
        cases = json.load(f)

    results = []

    for i, case in enumerate(cases):
        session_id = f"eval-{i}-{int(time.time())}"
        try:
            response = requests.post(
                API_URL,
                json={"query": case["query"], "session_id": session_id},
                timeout=60,
            )
            data = response.json()
        except Exception as e:
            results.append({**case, "passed": False, "reason": f"request failed: {e}"})
            continue

        answer = data.get("answer", "")
        sources = data.get("sources", [])
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
            keyword_hit = any(
                kw.lower() in clean_answer.lower() for kw in case["keywords"]
            )
            passed = has_sources and keyword_hit
            if not has_sources:
                reason = "expected an answer, got empty sources (fell back)"
            elif not keyword_hit:
                reason = f"got sources, but none of {case['keywords']} found in answer"
            else:
                reason = "correctly answered with expected content"

        results.append({**case, "passed": passed, "reason": reason, "answer": answer})

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
