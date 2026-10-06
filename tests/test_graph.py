from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from app.graph import (
    route_after_grading,
    route_after_validation,
    route_after_classification,
)
from app.schemas import GradeDecision, RouteDecision, ValidationResult


def test_route_after_grading_documents_found_generates():
    assert route_after_grading({"documents": [1], "refinement_count": 0}) == "generate"


def test_route_after_grading_empty_under_cap_refines():
    assert route_after_grading({"documents": [], "refinement_count": 0}) == "refine"


def test_route_after_grading_empty_at_cap_falls_back():
    assert route_after_grading({"documents": [], "refinement_count": 2}) == "fallback"


def test_route_after_validation_grounded_ends():
    from langgraph.graph import END

    assert route_after_validation({"is_grounded": True}) == END


def test_route_after_validation_ungrounded_with_claims_regenerates_once():
    state = {
        "is_grounded": False,
        "unsupported_claims": ["fee is $10"],
        "generation_attempts": 1,
    }
    assert route_after_validation(state) == "generate"


def test_route_after_validation_ungrounded_after_retry_falls_back():
    state = {
        "is_grounded": False,
        "unsupported_claims": ["fee is $10"],
        "generation_attempts": 2,
    }
    assert route_after_validation(state) == "fallback"


def test_route_after_validation_no_claims_falls_back_without_retry():
    # e.g. validator API failure: nothing to fix, so don't spend a retry.
    state = {"is_grounded": False, "unsupported_claims": [], "generation_attempts": 1}
    assert route_after_validation(state) == "fallback"


def test_route_after_classification_in_scope():
    assert route_after_classification({"classification": "in_scope"}) == "retrieve"


def test_route_after_classification_out_of_scope():
    assert route_after_classification({"classification": "out_of_scope"}) == "fallback"


@patch("app.nodes.get_classify_chain")
def test_classify_node_uses_structured_label(mock_get_chain):
    from app.nodes import classify_node

    mock_get_chain.return_value.invoke.return_value = RouteDecision(label="out_of_scope")

    result = classify_node({"original_query": "write me a poem"})
    assert result["classification"] == "out_of_scope"


@patch("app.nodes.get_grade_chain")
def test_grade_node_filters_irrelevant_documents(mock_get_chain):
    from app.nodes import grade_node

    mock_get_chain.return_value.invoke.side_effect = [
        GradeDecision(relevant=True),
        GradeDecision(relevant=False),
    ]

    state = {
        "query": "test query",
        "documents": [
            Document(page_content="relevant", metadata={"source": "a.md"}),
            Document(page_content="irrelevant", metadata={"source": "b.md"}),
        ],
    }

    result = grade_node(state)

    assert len(result["documents"]) == 1
    assert result["documents"][0].metadata["source"] == "a.md"
    assert result["sources"] == ["a.md"]


@patch("app.nodes.get_validate_chain")
def test_validate_node_catches_ungrounded_answer(mock_get_chain):
    from app.nodes import validate_node

    mock_get_chain.return_value.invoke.return_value = ValidationResult(
        unsupported_claims=["monthly fee is $10"], grounded=False
    )

    result = validate_node({"answer": "some answer", "documents": []})

    assert result["is_grounded"] is False
    assert result["unsupported_claims"] == ["monthly fee is $10"]


@patch("app.nodes.get_validate_chain")
def test_validate_node_fails_closed_on_contradictory_output(mock_get_chain):
    from app.nodes import validate_node

    mock_get_chain.return_value.invoke.return_value = ValidationResult(
        unsupported_claims=["invented term"], grounded=True
    )

    result = validate_node({"answer": "some answer", "documents": []})
    assert result["is_grounded"] is False


@patch("app.nodes.get_revise_chain")
@patch("app.nodes.get_generate_chain")
def test_generate_node_revises_with_validator_feedback(mock_gen, mock_rev):
    from app.nodes import generate_node

    mock_rev.return_value.invoke.return_value = "revised"

    result = generate_node(
        {
            "original_query": "q",
            "documents": [],
            "answer": "first draft",
            "generation_attempts": 1,
            "unsupported_claims": ["fee is $10"],
        }
    )

    assert result == {"answer": "revised", "generation_attempts": 2}
    mock_gen.return_value.invoke.assert_not_called()
    sent = mock_rev.return_value.invoke.call_args.args[0]
    assert sent["answer"] == "first draft"
    assert "fee is $10" in sent["unsupported_claims"]


def test_fallback_node_no_documents():
    from app.nodes import fallback_node

    result = fallback_node({"documents": []})
    assert "sources" in result
    assert result["sources"] == []


@patch("app.nodes.rerank_documents")
def test_rerank_node_reorders_documents(mock_rerank):
    from app.nodes import rerank_node, RERANK_TOP_N

    docs = [Document(page_content=f"doc {i}") for i in range(3)]
    mock_rerank.return_value = [docs[2], docs[0]]

    result = rerank_node({"query": "q", "documents": docs})

    assert result["documents"] == [docs[2], docs[0]]
    mock_rerank.assert_called_once_with("q", docs, top_n=RERANK_TOP_N)


@patch("app.nodes.rerank_documents", side_effect=RuntimeError("model missing"))
def test_rerank_node_falls_back_to_vector_order(mock_rerank):
    from app.nodes import rerank_node, RERANK_TOP_N

    docs = [Document(page_content=f"doc {i}") for i in range(6)]
    result = rerank_node({"query": "q", "documents": docs})
    assert result["documents"] == docs[:RERANK_TOP_N]


@patch("app.nodes.get_condense_chain")
def test_condense_query_skips_llm_without_history(mock_get_chain):
    from app.nodes import condense_query

    assert condense_query([], "what is the ATM limit") == "what is the ATM limit"
    mock_get_chain.assert_not_called()


@patch("app.nodes.get_condense_chain")
def test_condense_query_rewrites_follow_up(mock_get_chain):
    from app.nodes import condense_query

    mock_get_chain.return_value.invoke.return_value = "  What is the ATM limit for Premium accounts?\n"
    history = [
        {"role": "user", "content": "what is the ATM limit"},
        {"role": "assistant", "content": "$1,000 per day."},
    ]

    assert (
        condense_query(history, "what about premium?")
        == "What is the ATM limit for Premium accounts?"
    )


def test_source_of_normalizes_windows_paths():
    from app.nodes import source_of

    doc = Document(page_content="x", metadata={"source": r"sample_data\FAQ.md"})
    assert source_of(doc) == "sample_data/FAQ.md"


def test_chunks_keep_faq_heading_and_posix_source():
    from app.ingestion import load_and_chunk

    chunks = load_and_chunk()
    opening = [c for c in chunks if "photo ID" in c.page_content]

    assert len(opening) == 1
    assert "What documents do I need to open a checking account?" in opening[0].page_content
    assert opening[0].metadata["source"] == "sample_data/account_opening.md"
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_importing_app_builds_no_clients_or_models():
    # Lazy init: importing the API must not construct Groq clients or load
    # models, so unit tests and CI run without GROQ_API_KEY. Runs in a fresh
    # interpreter because other tests in this process may warm the caches.
    import os
    import subprocess
    import sys

    code = (
        "import app.main, app.nodes as n, app.vectorstore as v, app.rerank as r\n"
        "assert n._generate_llm.cache_info().currsize == 0\n"
        "assert n._small_llm.cache_info().currsize == 0\n"
        "assert v.get_embeddings.cache_info().currsize == 0\n"
        "assert r._get_cross_encoder.cache_info().currsize == 0\n"
    )
    env = {**os.environ, "GROQ_API_KEY": ""}
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_cache_key_is_stable_across_processes():
    from app.cache import _cache_key

    assert _cache_key("hello") == _cache_key("hello")
    assert _cache_key("hello").startswith("qcache:")
    assert len(_cache_key("hello")) == len("qcache:") + 32


# --- Groq outage: stop early instead of amplifying failing calls -------------

@patch("app.nodes.get_grade_chain")
def test_grade_node_stops_after_double_failure(mock_get_chain):
    import groq
    from app.nodes import grade_node

    err = groq.APIConnectionError(request=MagicMock())
    mock_get_chain.return_value.invoke.side_effect = err
    docs = [Document(page_content=f"doc {i}", metadata={"source": "a.md"}) for i in range(3)]

    result = grade_node({"query": "q", "documents": docs})

    # 2 attempts on the first doc, then stop: not 2 x 3 docs.
    assert mock_get_chain.return_value.invoke.call_count == 2
    assert result["upstream_error"] is True
    assert result["documents"] == []


def test_route_after_grading_upstream_error_skips_refine():
    state = {"documents": [], "refinement_count": 0, "upstream_error": True}
    assert route_after_grading(state) == "fallback"


@patch("app.nodes.get_generate_chain")
def test_generate_node_flags_upstream_error(mock_get_chain):
    import groq
    from app.nodes import generate_node

    mock_get_chain.return_value.invoke.side_effect = groq.APIConnectionError(
        request=MagicMock()
    )
    result = generate_node({"original_query": "q", "documents": []})
    assert result["upstream_error"] is True


def test_fallback_message_distinguishes_outage_from_unknown():
    from app.nodes import fallback_node

    outage = fallback_node({"documents": [], "upstream_error": True})["answer"]
    unknown = fallback_node({"documents": []})["answer"]

    assert "temporarily unable" in outage
    assert "don't have enough information" in unknown


@patch("app.nodes.get_grade_chain")
def test_grade_node_judges_against_original_question_not_refined_query(mock_get_chain):
    from app.nodes import grade_node

    mock_get_chain.return_value.invoke.return_value = GradeDecision(relevant=True)
    state = {
        "original_query": "what is the ATM limit",
        "query": "daily automated teller machine cash withdrawal maximum",
        "documents": [Document(page_content="doc", metadata={"source": "a.md"})],
    }

    grade_node(state)

    sent = mock_get_chain.return_value.invoke.call_args.args[0]
    assert sent["query"] == "what is the ATM limit"
