import json

import pytest

from tests.run_comparison_eval import EVAL_SET_PATH, hard_negative_outcome, score_case

HARD = {
    "query": "what is the fee for an outgoing international wire transfer",
    "expect": "hard_negative",
    "keywords": [],
    "trap": "Only the DOMESTIC outgoing wire fee ($25) is in the KB.",
}
SOURCES = ["sample_data/FAQ.md"]


def test_hard_negative_refusal_passes():
    passed, reason = score_case(HARD, {"answer": "I don't have enough information...", "sources": []})
    assert passed and reason == "refused"


@pytest.mark.parametrize(
    "answer",
    [
        "The documents only list the domestic wire fee ($25); the international fee is not specified.",
        "The provided information doesn’t mention international wire fees.",  # curly apostrophe
        "The FAQ **does not mention** an international wire fee.",  # markdown
        # phrasings the first live run used that the original list missed:
        "I’m sorry, but the fee for an outgoing international wire transfer isn’t covered in the information you provided.",
        "I’m sorry, but the information you’re looking for isn’t included in the provided context.",
        "It does **not** list a separate daily ATM withdrawal limit for a savings account.",
    ],
)
def test_hard_negative_gap_acknowledgement_passes(answer):
    passed, reason = score_case(HARD, {"answer": answer, "sources": SOURCES})
    assert passed and reason == "acknowledged"


def test_hard_negative_misapplied_fact_fails():
    answer = "The fee for an outgoing international wire transfer is $25."
    passed, reason = score_case(HARD, {"answer": answer, "sources": SOURCES})
    assert not passed
    assert reason.startswith("answered") and "DOMESTIC" in reason


def test_outcome_requires_sources_for_acknowledged():
    # An empty-sources reply is a refusal even if it happens to contain a gap phrase.
    assert hard_negative_outcome("not specified", []) == "refused"


def test_eval_set_is_well_formed():
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    assert len({c["query"].lower() for c in cases}) == len(cases), "duplicate queries"
    for c in cases:
        assert c["expect"] in {"answerable", "fallback", "hard_negative"}
        if c["expect"] == "answerable":
            assert c["keywords"], c["query"]
        if c["expect"] == "hard_negative":
            assert c["trap"] and not c["keywords"], c["query"]
    assert sum(c["expect"] == "hard_negative" for c in cases) >= 10
