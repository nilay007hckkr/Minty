from unittest.mock import patch, MagicMock

from app.graph import (
    route_after_grading,
    route_after_validation,
    route_after_classification,
)


def test_route_after_grading_documents_found_generates():
    assert route_after_grading({"documents": [1], "refinement_count": 0}) == "generate"


def test_route_after_grading_empty_under_cap_refines():
    assert route_after_grading({"documents": [], "refinement_count": 0}) == "refine"


def test_route_after_grading_empty_at_cap_falls_back():
    assert route_after_grading({"documents": [], "refinement_count": 2}) == "fallback"


def test_route_after_validation_grounded_ends():
    from langgraph.graph import END

    assert route_after_validation({"is_grounded": "yes"}) == END


def test_route_after_validation_ungrounded_falls_back():
    assert route_after_validation({"is_grounded": "no"}) == "fallback"


def test_route_after_classification_in_scope():
    assert route_after_classification({"classification": "in_scope"}) == "retrieve"


def test_route_after_classification_out_of_scope():
    assert route_after_classification({"classification": "out_of_scope"}) == "fallback"


@patch("app.nodes.grade_chain")
def test_grade_node_filters_irrelevant_documents(mock_chain):
    from app.nodes import grade_node
    from langchain_core.documents import Document

    mock_chain.invoke.side_effect = ["yes", "no"]

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


@patch("app.nodes.validate_chain")
def test_validate_node_catches_ungrounded_answer(mock_chain):
    from app.nodes import validate_node

    mock_chain.invoke.return_value = "some reasoning...\nverdict: no"

    state = {"answer": "some answer", "documents": []}
    result = validate_node(state)

    assert result["is_grounded"] == "no"


def test_fallback_node_no_documents():
    from app.nodes import fallback_node

    result = fallback_node({"documents": []})
    assert "sources" in result
    assert result["sources"] == []


@patch("app.nodes.rerank_documents")
def test_rerank_node_reorders_documents(mock_rerank):
    from app.nodes import rerank_node
    from langchain_core.documents import Document


def test_first_word_takes_leading_label_only():
    from app.nodes import first_word

    assert first_word("Yes.") == "yes"
    assert first_word("  'in_scope'\n") == "in_scope"
    assert first_word("no, though it mentions yes") == "no"
    assert first_word("not in_scope") == "not"
    assert first_word("") == ""


def test_final_verdict_reads_last_line_only():
    from app.nodes import final_verdict

    assert final_verdict("1. claim ok\nVerdict: yes") == "yes"
    assert final_verdict("**Verdict: yes**") == "yes"
    assert final_verdict("If grounded I'd say Verdict: yes\nVerdict: no") == "no"
    assert final_verdict("Verdict: yes\n2. extra unsupported claim") == "no"
    assert final_verdict("") == "no"


def test_source_of_normalizes_windows_paths():
    from app.nodes import source_of
    from langchain_core.documents import Document

    doc = Document(page_content="x", metadata={"source": "sample_data\FAQ.md"})
    assert source_of(doc) == "sample_data/FAQ.md"
