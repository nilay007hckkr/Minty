from langchain_core.prompts import ChatPromptTemplate

generate_prompt = ChatPromptTemplate.from_template(
    """You are an expert banking assistant. Answer the user's question using only the provided context. 

Context:
{context}

Question:
{original_query}

Answer:"""
)

grade_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing whether a retrieved document contains enough relevant information to answer a user's question. 
    If the document contains relevant keywords, concepts, or direct answers related to the query, output 'yes'. 
    If it does not contain helpful information to answer the question, output 'no'. 
    Output exactly one word: 'yes' or 'no'.

Retrieved Document:
{context}

User Question:
{query}

Verdict:"""
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
    """You are a strict fact-checker validating an AI-generated answer against source documents.
    
    Step 1: Extract every specific claim (numbers, fees, abbreviations, defined terms, policies) made in the generated answer as a numbered list.
    Step 2: For each claim, check if it is explicitly stated in the source documents. 
    Step 3: If even ONE claim (including implied definitions or abbreviations like '(OD)') is not explicitly in the source, the answer is NOT grounded.
    
    Source Documents:
    {context}

    Generated Answer:
    {answer}

    Output your analysis step-by-step. 
    You MUST end your response with EXACTLY one of these two phrases on a new line:
    "Verdict: yes" (if fully grounded) OR "Verdict: no" (if any claim is unsupported)."""
)

classify_prompt = ChatPromptTemplate.from_template(
    """You are a query router for a banking information assistant.

    Classify the user's query into exactly one label:
    - in_scope: a factual question about banking products, fees, rates, hours,
      account types, or how-to instructions.
    - out_of_scope: anything else - small talk, unrelated topics, requests for
      personal financial advice, or requests to perform account actions.

    Respond with exactly one word: in_scope or out_of_scope.

    User query: {query}"""
)
