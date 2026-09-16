import logging
import groq
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.state import GraphState
from app.prompts import (
    classify_prompt,
    generate_prompt,
    grade_prompt,
    refine_prompt,
    validate_prompt,
)
from app.vectorstore import get_vectorstore
from app.rerank import rerank_documents

logger = logging.getLogger(__name__)

vector_store = get_vectorstore()
retriever = vector_store.as_retriever(search_kwargs={"k": 6})

generate_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
grade_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

classify_chain = classify_prompt | grade_llm | StrOutputParser()
generate_chain = generate_prompt | generate_llm | StrOutputParser()
grade_chain = grade_prompt | grade_llm | StrOutputParser()
refine_chain = refine_prompt | grade_llm | StrOutputParser()
validate_chain = validate_prompt | generate_llm | StrOutputParser()


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


def classify_node(state: GraphState) -> dict:
    query = state.get("original_query", "")
    try:
        label = classify_chain.invoke({"query": query}).strip().lower()
    except groq.GroqError as e:
        logger.error(f"Classification API failed. Defaulting to in_scope. Error: {e}")
        label = "in_scope"

    classification = "in_scope" if "in_scope" in label else "out_of_scope"
    return {"classification": classification, "query": query, "refinement_count": 0}


def retrieve_node(state: GraphState) -> dict:
    query = state.get("query", "")
    try:
        docs = retriever.invoke(query)
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
        reranked = rerank_documents(query, docs, top_n=3)
        logger.debug(
            f"Reranked (cross-encoder order): {[d.metadata.get('source') for d in reranked]}"
        )
    except Exception as e:
        logger.error(f"Reranking failed, falling back to vector order: {e}")
        reranked = docs[:3]

    return {"documents": reranked}


def grade_node(state: GraphState) -> dict:
    query = state.get("query", "")
    docs = state.get("documents", [])
    filtered_docs = []

    for doc in docs:
        for attempt in range(2):
            try:
                verdict = (
                    grade_chain.invoke({"query": query, "context": doc.page_content})
                    .strip()
                    .lower()
                )
                logger.debug(f"Grade {doc.metadata.get('source')} -> {verdict}")
                if "yes" in verdict:
                    filtered_docs.append(doc)
                break
            except groq.GroqError as e:
                if attempt == 0:
                    logger.warning(
                        f"Grading API error for {doc.metadata.get('source')}. Retrying... Error: {e}"
                    )
                else:
                    logger.error(
                        f"Grading API failed twice for {doc.metadata.get('source')}. Skipping. Error: {e}"
                    )

    sources = list(
        set([doc.metadata.get("source", "sample_data") for doc in filtered_docs])
    )
    return {"documents": filtered_docs, "sources": sources}


def refine_node(state: GraphState) -> dict:
    query = state.get("query", "")
    original_query = state.get("original_query", query)
    count = state.get("refinement_count", 0)

    try:
        rewritten_query = refine_chain.invoke(
            {"original_query": original_query, "query": query}
        ).strip()
        logger.debug(f"Refine Cycle {count + 1}: '{query}' -> '{rewritten_query}'")
    except groq.GroqError as e:
        logger.error(
            f"Refinement API failed. Falling back to original query. Error: {e}"
        )
        rewritten_query = query

    return {"query": rewritten_query, "refinement_count": count + 1}


def generate_node(state: GraphState) -> dict:
    original_query = state.get("original_query", "")
    docs = state.get("documents", [])
    context = format_docs(docs)

    try:
        answer = generate_chain.invoke(
            {"original_query": original_query, "context": context}
        )
    except groq.GroqError as e:
        logger.error(f"Generation API failed: {e}")
        answer = "Error: Upstream API failure during generation."

    return {"answer": answer}


def validate_node(state: GraphState) -> dict:
    answer = state.get("answer", "")
    docs = state.get("documents", [])
    context = format_docs(docs)

    if answer.startswith("Error:"):
        logger.debug("Bypassing validation due to upstream API error.")
        return {"is_grounded": "no"}

    try:
        analysis = (
            validate_chain.invoke({"context": context, "answer": answer})
            .strip()
            .lower()
        )
        logger.debug(f"Validate Analysis:\n{analysis}")

        if "verdict: yes" in analysis:
            return {"is_grounded": "yes"}
        return {"is_grounded": "no"}
    except groq.GroqError as e:
        logger.error(f"Validation API failed. Failing closed. Error: {e}")
        return {"is_grounded": "no"}


def fallback_node(state: GraphState) -> dict:
    docs = state.get("documents", [])

    if not docs:
        message = "I don't have enough information in the provided banking database to answer that confidently."
    else:
        message = "I found related information in our banking database, but I couldn't generate a fully verified answer based strictly on those guidelines."

    return {"answer": message, "sources": []}
