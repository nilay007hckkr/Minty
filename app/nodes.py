import logging
from functools import cache
from typing import List

import groq
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pydantic import ValidationError

from app.state import GraphState
from app.prompts import (
    classify_prompt,
    condense_prompt,
    generate_prompt,
    grade_prompt,
    refine_prompt,
    revise_prompt,
    validate_prompt,
)
from app.schemas import GradeDecision, RouteDecision, ValidationResult
from app.vectorstore import get_vectorstore
from app.rerank import rerank_documents

logger = logging.getLogger(__name__)

RETRIEVE_K = 6
# 3 is enough once chunks carry their FAQ heading (see ingestion.py): every
# answerable eval question's target chunk then reranks #1. Raising this only
# adds grading calls.
RERANK_TOP_N = 3

# Upstream API failures, plus a structured response that fails to parse.
LLM_ERRORS = (groq.GroqError, OutputParserException, ValidationError)


# Clients and chains are built on first use, not at import: importing the
# graph (e.g. in unit tests or CI) should not need GROQ_API_KEY or load models.
@cache
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": RETRIEVE_K})


@cache
def _generate_llm() -> ChatGroq:
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)


@cache
def _small_llm() -> ChatGroq:
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0)


def _structured(llm: ChatGroq, schema):
    return llm.with_structured_output(schema, method="json_schema", strict=True)


@cache
def get_classify_chain():
    return classify_prompt | _structured(_small_llm(), RouteDecision)


@cache
def get_grade_chain():
    return grade_prompt | _structured(_small_llm(), GradeDecision)


@cache
def get_refine_chain():
    return refine_prompt | _small_llm() | StrOutputParser()


@cache
def get_condense_chain():
    return condense_prompt | _small_llm() | StrOutputParser()


@cache
def get_generate_chain():
    return generate_prompt | _generate_llm() | StrOutputParser()


@cache
def get_revise_chain():
    return revise_prompt | _generate_llm() | StrOutputParser()


@cache
def get_validate_chain():
    return validate_prompt | _structured(_generate_llm(), ValidationResult)


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


def source_of(doc: Document) -> str:
    # Chroma stores whatever path separator the indexing OS used; normalize so
    # sources look the same whether indexed on Windows or in Docker.
    return doc.metadata.get("source", "sample_data").replace("\\", "/")


def format_history(history: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def condense_query(history: list[dict], query: str) -> str:
    """Rewrites a follow-up ("what about premium accounts?") into a standalone
    question using recent chat history, so retrieval and the semantic cache
    see a self-contained query. Returns the query unchanged on failure."""
    if not history:
        return query
    try:
        standalone = get_condense_chain().invoke(
            {"history": format_history(history), "query": query}
        ).strip()
    except LLM_ERRORS as e:
        logger.error(f"Query condensation failed. Using raw query. Error: {e}")
        return query
    return standalone or query


def classify_node(state: GraphState) -> dict:
    query = state.get("original_query", "")
    try:
        label = get_classify_chain().invoke({"query": query}).label
    except LLM_ERRORS as e:
        logger.error(f"Classification API failed. Defaulting to in_scope. Error: {e}")
        label = "in_scope"

    return {"classification": label, "query": query, "refinement_count": 0}


def retrieve_node(state: GraphState) -> dict:
    query = state.get("query", "")
    try:
        docs = get_retriever().invoke(query)
        logger.debug(
            f"Retrieved (vector order): {[d.metadata.get('source') for d in docs]}"
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        docs = []
    return {"documents": docs}


def rerank_node(state: GraphState) -> dict:
    query = state.get("query", "")
    docs = state.get("documents", [])

    try:
        reranked = rerank_documents(query, docs, top_n=RERANK_TOP_N)
        logger.debug(
            f"Reranked (cross-encoder order): {[d.metadata.get('source') for d in reranked]}"
        )
    except Exception as e:
        logger.error(f"Reranking failed, falling back to vector order: {e}")
        reranked = docs[:RERANK_TOP_N]

    return {"documents": reranked}


def grade_node(state: GraphState) -> dict:
    # Relevance is judged against what the user asked, not the refined search
    # query: refinement adds synonyms to help retrieval and can drift, and
    # generate answers original_query, so that's what the docs must support.
    query = state.get("original_query") or state.get("query", "")
    docs = state.get("documents", [])
    filtered_docs = []
    api_failed = False

    for doc in docs:
        for attempt in range(2):
            try:
                decision = get_grade_chain().invoke(
                    {"query": query, "context": doc.page_content}
                )
                logger.debug(f"Grade {doc.metadata.get('source')} -> {decision.relevant}")
                if decision.relevant:
                    filtered_docs.append(doc)
                break
            except LLM_ERRORS as e:
                if attempt == 0:
                    logger.warning(
                        f"Grading API error for {doc.metadata.get('source')}. Retrying... Error: {e}"
                    )
                else:
                    logger.error(
                        f"Grading API failed twice for {doc.metadata.get('source')}. "
                        f"Stopping grading. Error: {e}"
                    )
                    api_failed = True
        if api_failed:
            # Two consecutive failures means the API is very likely down.
            # Grading the remaining docs, then refining and re-grading, would
            # only add more doomed calls (each with ChatGroq's own retries).
            break

    sources = sorted(set(source_of(doc) for doc in filtered_docs))
    return {"documents": filtered_docs, "sources": sources, "upstream_error": api_failed}


def refine_node(state: GraphState) -> dict:
    query = state.get("query", "")
    original_query = state.get("original_query", query)
    count = state.get("refinement_count", 0)

    try:
        rewritten_query = get_refine_chain().invoke(
            {"original_query": original_query, "query": query}
        ).strip()
        logger.debug(f"Refine Cycle {count + 1}: '{query}' -> '{rewritten_query}'")
    except LLM_ERRORS as e:
        logger.error(
            f"Refinement API failed. Falling back to original query. Error: {e}"
        )
        rewritten_query = query

    return {"query": rewritten_query or query, "refinement_count": count + 1}


def generate_node(state: GraphState) -> dict:
    original_query = state.get("original_query", "")
    docs = state.get("documents", [])
    context = format_docs(docs)
    attempts = state.get("generation_attempts", 0)
    unsupported = state.get("unsupported_claims") or []

    try:
        if attempts and unsupported:
            # Second pass: tell the model exactly which claims the validator
            # rejected. A plain re-run at temperature 0 would mostly reproduce
            # the same answer.
            answer = get_revise_chain().invoke(
                {
                    "original_query": original_query,
                    "context": context,
                    "answer": state.get("answer", ""),
                    "unsupported_claims": "\n".join(f"- {c}" for c in unsupported),
                }
            )
        else:
            answer = get_generate_chain().invoke(
                {"original_query": original_query, "context": context}
            )
    except LLM_ERRORS as e:
        logger.error(f"Generation API failed: {e}")
        return {
            "answer": "Error: Upstream API failure during generation.",
            "generation_attempts": attempts + 1,
            "upstream_error": True,
        }

    return {"answer": answer, "generation_attempts": attempts + 1}


def validate_node(state: GraphState) -> dict:
    answer = state.get("answer", "")
    docs = state.get("documents", [])
    context = format_docs(docs)

    if answer.startswith("Error:"):
        logger.debug("Bypassing validation due to upstream API error.")
        return {"is_grounded": False, "unsupported_claims": []}

    try:
        result = get_validate_chain().invoke({"context": context, "answer": answer})
    except LLM_ERRORS as e:
        logger.error(f"Validation API failed. Failing closed. Error: {e}")
        return {"is_grounded": False, "unsupported_claims": []}

    logger.debug(f"Validation: {result}")
    # Fail closed if the two fields disagree (grounded=true but claims listed).
    grounded = result.grounded and not result.unsupported_claims
    return {"is_grounded": grounded, "unsupported_claims": result.unsupported_claims}


def fallback_node(state: GraphState) -> dict:
    docs = state.get("documents", [])

    if state.get("upstream_error"):
        # Not "I don't know": the KB may well cover this, the model API failed.
        message = "I'm temporarily unable to answer because an upstream service is unavailable. Please try again in a moment."
    elif not docs:
        message = "I don't have enough information in the provided banking database to answer that confidently."
    else:
        message = "I found related information in our banking database, but I couldn't generate a fully verified answer based strictly on those guidelines."

    return {"answer": message, "sources": []}
