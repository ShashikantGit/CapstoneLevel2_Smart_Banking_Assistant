from __future__ import annotations

import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from src.api.v1.schemas.query_schema import AIResponse
from src.api.v1.states.rag_state import RAGState
from src.core.db import get_sql_database
from src.api.v1.agents.retriever import retrieve_documents


load_dotenv()


MAX_SQL_RETRIES = 3


# ============================================================
# LLM
# ============================================================

def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    return ChatOpenAI(
        model=os.getenv(
            "OPENAI_CHAT_MODEL",
            "gpt-5.5",
        ),
        api_key=api_key,
        temperature=0,
    )


# ============================================================
# ROUTER
# ============================================================

class RouteDecision(BaseModel):
    route: Literal[
        "VECTOR_DB",
        "RDBMS",
    ]
    reason: str


def router_node(
    state: RAGState,
) -> RAGState:

    query = state["query"].strip()

    structured_llm = (
        _get_llm()
        .with_structured_output(
            RouteDecision
        )
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the routing controller for a Smart Banking Assistant.

Your job is to decide whether the user's question should be
answered from:

1. VECTOR_DB
2. RDBMS

============================================================
VECTOR_DB
============================================================

Choose VECTOR_DB when the user is asking for GENERAL banking
knowledge contained in uploaded PDFs or the banking knowledge base.

Examples:

- What is the home loan interest rate?
- What are the home loan processing charges?
- What documents are required for a home loan?
- What is the eligibility for a personal loan?
- What is the fixed deposit interest rate?
- What are NEFT charges?
- What are RTGS charges?
- What is the credit card annual fee?
- What is the loan processing fee?
- What is the bank's loan policy?
- What are the rules for closing an account?
- How can I open an account?
- What is the minimum balance requirement?

These are general banking questions.

============================================================
RDBMS
============================================================

Choose RDBMS when the question asks for actual
customer/account-specific information.

Available tables:

accounts
transactions
loan_accounts
fixed_deposits
credit_cards
card_transactions

Examples:

- What is my account balance?
- Show my account details.
- What are my recent transactions?
- What was my last transaction?
- Show my transaction history.
- How much did I spend?
- Show my deposits.
- Show my withdrawals.
- What loans do I have?
- What is my outstanding loan balance?
- Show my fixed deposits.
- What is my credit card balance?
- Show my credit card transactions.
- What was my latest card transaction?

============================================================
IMPORTANT DISTINCTION
============================================================

General banking information -> VECTOR_DB

Customer-specific banking information -> RDBMS

Examples:

"What is the home loan interest rate?"
-> VECTOR_DB

"What is the interest rate on my home loan?"
-> RDBMS

"What are the charges for a home loan?"
-> VECTOR_DB

"What charges were applied to my account?"
-> RDBMS

"What is the transaction limit?"
-> VECTOR_DB

"What is my account balance?"
-> RDBMS

"How can I check my account balance?"
-> VECTOR_DB

Never choose RDBMS merely because the question contains
words such as loan, account, transaction, or card.

Use the actual meaning of the question.

Return exactly one route and one short reason.
""",
            ),
            (
                "human",
                """
User question:

{query}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    decision = chain.invoke(
        {
            "query": query,
        }
    )

    print(
        f"[ROUTER] route={decision.route} "
        f"reason={decision.reason}"
    )

    return {
        **state,
        "route": decision.route,
    }


# ============================================================
# ROUTER DECISION
# ============================================================

def route_after_router(
    state: RAGState,
) -> str:

    route = state.get(
        "route",
        "VECTOR_DB",
    )

    if route == "RDBMS":
        print("[ROUTER] Sending query to RDBMS")
        return "sql_generator"

    print("[ROUTER] Sending query to VECTOR_DB")
    return "vector_retrieval"


# ============================================================
# SQL GENERATOR
# ============================================================

def sql_generator_node(
    state: RAGState,
) -> RAGState:

    print("[RDBMS] Generating SQL...")

    query = state["query"]

    feedback = state.get(
        "query_feedback",
        "",
    )

    db = get_sql_database()

    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a PostgreSQL expert for a Smart Banking Assistant.

Generate ONE safe SELECT query that answers the user's question.

Use ONLY tables and columns available in the supplied schema.

Rules:

1. Return ONLY raw SQL.
2. Do not use markdown.
3. Do not use ```sql.
4. Do not generate INSERT.
5. Do not generate UPDATE.
6. Do not generate DELETE.
7. Do not generate DROP.
8. Do not generate ALTER.
9. Do not generate CREATE.
10. Only SELECT statements are allowed.
11. Add LIMIT 50 for non-aggregate queries.
12. Never invent tables.
13. Never invent columns.
14. Use the exact database schema.
15. For account questions use accounts.
16. For transaction questions use transactions.
17. For loan questions use loan_accounts.
18. For fixed deposit questions use fixed_deposits.
19. For credit card questions use credit_cards.
20. For credit card transaction questions use card_transactions.

Database schema:

{schema}
""",
            ),
            (
                "human",
                """
User question:

{question}

Previous SQL execution feedback:

{feedback}
""",
            ),
        ]
    )

    chain = sql_prompt | _get_llm()

    response = chain.invoke(
        {
            "schema": schema_info,
            "question": query,
            "feedback": feedback,
        }
    )

    generated_sql = str(
        response.content
    ).strip()

    generated_sql = (
        generated_sql
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )

    print(
        "========== GENERATED SQL =========="
    )
    print(generated_sql)

    return {
        **state,
        "generated_sql": generated_sql,
    }


# ============================================================
# SQL EXECUTOR
# ============================================================

def sql_executor_node(
    state: RAGState,
) -> RAGState:

    print("[RDBMS] Executing SQL...")

    generated_sql = state.get(
        "generated_sql",
        "",
    )

    attempts = state.get(
        "attempts",
        0,
    )

    db = get_sql_database()

    try:

        result = db.run(
            generated_sql
        )

        print(
            "[RDBMS] SQL execution successful."
        )

        return {
            **state,
            "response": result,
            "sql_result": str(result),
            "attempts": attempts + 1,
            "query_feedback": "",
        }

    except Exception as exc:

        error_message = str(exc)

        print(
            "[RDBMS] SQL execution failed:",
            error_message,
        )

        return {
            **state,
            "response": (
                "Unable to execute the generated SQL."
            ),
            "sql_result": "",
            "attempts": attempts + 1,
            "query_feedback": error_message,
        }


# ============================================================
# SQL RETRY ROUTER
# ============================================================

def route_after_sql_execution(
    state: RAGState,
) -> str:

    feedback = state.get(
        "query_feedback",
        "",
    )

    attempts = state.get(
        "attempts",
        0,
    )

    if feedback and attempts < MAX_SQL_RETRIES:

        print(
            "[RDBMS] SQL failed. Retrying..."
        )

        return "sql_generator"

    return "rdbms_response"


# ============================================================
# RDBMS RESPONSE
# ============================================================

def response_generator_node(
    state: RAGState,
) -> RAGState:

    print(
        "[RDBMS] Generating final response..."
    )

    query = state["query"]

    sql = state.get(
        "generated_sql",
        "",
    )

    result = state.get(
        "sql_result",
        "",
    )

    structured_llm = (
        _get_llm()
        .with_structured_output(
            AIResponse
        )
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful Smart Banking Assistant.

Answer the user's question using ONLY the SQL result.

Rules:

- Do not invent information.
- Do not use PDF knowledge.
- Do not add policy information.
- If the result is empty, say that no matching record was found.
- Be concise.
- Format numbers and lists clearly.

For this RDBMS response:

policy_citations = "N/A"
page_no = "N/A"
document_name = "smart_banking_assistant_DB"
""",
            ),
            (
                "human",
                """
Question:

{query}

SQL:

{sql}

SQL Result:

{result}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    answer = chain.invoke(
        {
            "query": query,
            "sql": sql,
            "result": result,
        }
    )

    response = answer.model_dump()

    response["policy_citations"] = "N/A"
    response["page_no"] = "N/A"
    response["document_name"] = (
        "smart_banking_assistant_DB"
    )
    response["sql_query_executed"] = sql

    return {
        **state,
        "response": response,
    }


# ============================================================
# VECTOR RETRIEVAL
# ============================================================

def vector_retrieval_node(
    state: RAGState,
) -> RAGState:

    query = state["query"]

    print(
        "[VECTOR_DB] Retrieving documents..."
    )

    try:

        documents = retrieve_documents(
            query=query,
            top_k=5,
        )

        print(
            f"[VECTOR_DB] Retrieved {len(documents)} documents."
        )

        return {
            **state,
            "retrieved_docs": documents,
            "reranked_docs": documents,
        }

    except Exception as exc:

        print(
            "[VECTOR_DB] Retrieval failed:",
            exc,
        )

        return {
            **state,
            "retrieved_docs": [],
            "reranked_docs": [],
        }


# ============================================================
# VECTOR RESPONSE
# ============================================================

def vector_response_generator_node(
    state: RAGState,
) -> RAGState:

    print(
        "[VECTOR_DB] Generating final response..."
    )

    query = state["query"]

    documents = state.get(
        "reranked_docs",
        [],
    )

    if not documents:

        response = {
            "answer": (
                "I could not find relevant information "
                "in the available banking documents."
            ),
            "policy_citations": "N/A",
            "page_no": "N/A",
            "document_name": "N/A",
        }

        return {
            **state,
            "response": response,
        }

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        content = str(
            document.get(
                "content",
                "",
            )
        ).strip()

        metadata = (
            document.get(
                "metadata",
                {},
            )
            or {}
        )

        document_name = metadata.get(
            "document_name",
            "Unknown document",
        )

        page_number = metadata.get(
            "page_number",
            "N/A",
        )

        context_parts.append(
            f"""
Document {index}

Document name:
{document_name}

Page:
{page_number}

Content:
{content}
"""
        )

    context = "\n".join(
        context_parts
    )

    structured_llm = (
        _get_llm()
        .with_structured_output(
            AIResponse
        )
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a Smart Banking Assistant.

Answer the user's question using ONLY the supplied
PDF/document context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent rates.
3. Do not invent fees.
4. Do not invent dates.
5. Do not invent policies.
6. Do not invent eligibility criteria.
7. If the exact answer is available in the context,
   answer it directly.
8. If the context does not contain the answer,
   clearly say that the information was not found.
9. Prefer the most relevant document chunk.
10. Do not mention irrelevant retrieved chunks.
11. Keep the answer concise.
12. Include the document name and page number
    when available.

For PDF responses, populate:

policy_citations:
The relevant document/page citation.

page_no:
The page number containing the answer.

document_name:
The document containing the answer.
""",
            ),
            (
                "human",
                """
User question:

{query}

Retrieved PDF context:

{context}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    answer = chain.invoke(
        {
            "query": query,
            "context": context,
        }
    )

    response = answer.model_dump()

    return {
        **state,
        "response": response,
    }


# ============================================================
# BUILD GRAPH
# ============================================================

def build_rag_graph():

    workflow = StateGraph(
        RAGState
    )

    # -------------------------------
    # Nodes
    # -------------------------------

    workflow.add_node(
        "router",
        router_node,
    )

    workflow.add_node(
        "sql_generator",
        sql_generator_node,
    )

    workflow.add_node(
        "sql_execution",
        sql_executor_node,
    )

    workflow.add_node(
        "rdbms_response",
        response_generator_node,
    )

    workflow.add_node(
        "vector_retrieval",
        vector_retrieval_node,
    )

    workflow.add_node(
        "vector_response",
        vector_response_generator_node,
    )

    # -------------------------------
    # Entry
    # -------------------------------

    workflow.set_entry_point(
        "router"
    )

    # -------------------------------
    # Router
    # -------------------------------

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "sql_generator": "sql_generator",
            "vector_retrieval": "vector_retrieval",
        },
    )

    # -------------------------------
    # RDBMS
    # -------------------------------

    workflow.add_edge(
        "sql_generator",
        "sql_execution",
    )

    workflow.add_conditional_edges(
        "sql_execution",
        route_after_sql_execution,
        {
            "sql_generator": "sql_generator",
            "rdbms_response": "rdbms_response",
        },
    )

    workflow.add_edge(
        "rdbms_response",
        END,
    )

    # -------------------------------
    # VECTOR DB
    # -------------------------------

    workflow.add_edge(
        "vector_retrieval",
        "vector_response",
    )

    workflow.add_edge(
        "vector_response",
        END,
    )

    return workflow.compile()


# ============================================================
# COMPILE
# ============================================================

rag_graph = build_rag_graph()


# ============================================================
# RUN AGENT
# ============================================================

def run_search_agent(
    query: str,
):

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    print(
        "============ INSIDE run_search_agent ============"
    )

    initial_state = {
        "query": query.strip(),
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "attempts": 0,
        "query_feedback": "",
    }

    final_state = rag_graph.invoke(
        initial_state
    )

    return final_state["response"]


# ============================================================
# STREAMING
# ============================================================

async def run_search_agent_stream(
    query: str,
):

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    initial_state = {
        "query": query.strip(),
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "attempts": 0,
        "query_feedback": "",
    }

    async for event in rag_graph.astream_events(
        initial_state,
        version="v1",
    ):

        kind = event["event"]

        if kind == "on_chat_model_stream":

            content = (
                event["data"]["chunk"].content
            )

            if content:

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "token": content
                        }
                    )
                    + "\n\n"
                )

    yield "data: [DONE]\n\n"