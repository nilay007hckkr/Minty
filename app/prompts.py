from langchain_core.prompts import ChatPromptTemplate

generate_prompt = ChatPromptTemplate.from_template(
    """You are an expert banking assistant. Answer the user's question using only the provided context.

Context:
{context}

Question:
{original_query}

Answer:"""
)

revise_prompt = ChatPromptTemplate.from_template(
    """You are an expert banking assistant. A fact-checker found that your previous answer contained claims not supported by the context.

Rewrite the answer using only the provided context. Remove or correct every unsupported claim listed below. Do not add any new facts that are not in the context.

Context:
{context}

Question:
{original_query}

Previous answer:
{answer}

Unsupported claims:
{unsupported_claims}

Revised answer:"""
)

grade_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing whether a retrieved document helps answer a user's question.
    Mark it relevant if it contains facts, policies, or instructions that answer the question or part of it.
    Mark it not relevant if it only shares loose keywords with the question but would not help answer it.

Retrieved Document:
{context}

User Question:
{query}"""
)

refine_prompt = ChatPromptTemplate.from_template(
    """You are an expert search query optimizer for a banking database.
    The following user query did not return any relevant results in our vector search.
    Rewrite the query to improve retrieval. Expand abbreviations, add likely financial synonyms, and be more specific.

    CRITICAL INSTRUCTION: NEVER output an empty string. If you cannot improve the query, return the original query exactly as it was.
    Output ONLY the rewritten search query, nothing else.

    Original User Question: {original_query}
    Current Search Query: {query}

    Rewritten Search Query:"""
)

validate_prompt = ChatPromptTemplate.from_template(
    """You are a fact-checker validating an AI-generated banking answer against source documents.

    Check FACTUAL CLAIMS only: numbers, amounts, fees, rates, limits, times, deadlines,
    eligibility rules, product names, defined terms or abbreviations, and procedures
    (where or how to do something). Each one must be stated in, or directly implied by,
    the source documents. Treat a claim as unsupported if it adds a detail, condition,
    or definition the sources do not contain, even if it sounds plausible.

    Do NOT flag:
    - wording or paraphrase differences (e.g. "open a new checking account" vs
      "open a checking account"),
    - restating the user's question,
    - formatting, greetings, or general connective phrasing that asserts no fact.

    Source Documents:
    {context}

    Generated Answer:
    {answer}

    List every unsupported factual claim. Set grounded to true only if that list is empty."""
)

classify_prompt = ChatPromptTemplate.from_template(
    """You are a query router for a banking information assistant.

    Classify the user's query into exactly one label:
    - in_scope: a factual question about banking products, fees, rates, hours,
      account types, or how-to instructions.
    - out_of_scope: anything else - small talk, unrelated topics, requests for
      personal financial advice, or requests to perform account actions.

    User query: {query}"""
)

condense_prompt = ChatPromptTemplate.from_template(
    """Given a conversation and a follow-up message, rewrite the follow-up as a standalone question that can be understood without the conversation.

    Rules:
    - Resolve pronouns and references ("it", "that card", "what about premium?") using the conversation.
    - If the follow-up is already standalone, or unrelated to the conversation, return it unchanged.
    - Do NOT answer the question. Do NOT add facts.
    - Output ONLY the standalone question.

    Conversation:
    {history}

    Follow-up: {query}

    Standalone question:"""
)
