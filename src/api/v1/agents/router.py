from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.api.v1.states.rag_state import RAGState


load_dotenv()


class RouteDecision(BaseModel):
    route: Literal[
        "VECTOR_DB",
        "RDBMS",
    ]
    reason: str


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
You are the query router for a Smart Banking Assistant.

Your job is to select exactly ONE route:

1. VECTOR_DB
2. RDBMS

==================================================
VECTOR_DB
==================================================

Choose VECTOR_DB when the question asks for GENERAL
banking information available in PDFs or the knowledge base.

This includes:

- interest rates
- loan rates
- processing fees
- foreclosure charges
- banking charges
- eligibility
- required documents
- procedures
- policies
- rules
- limits
- general product information

Examples:

"What is the home loan interest rate?"
-> VECTOR_DB

"What are the home loan processing charges?"
-> VECTOR_DB

"What documents are required for a home loan?"
-> VECTOR_DB

"What is the fixed deposit interest rate?"
-> VECTOR_DB

"What is the credit card annual fee?"
-> VECTOR_DB

"What is the foreclosure charge for a home loan?"
-> VECTOR_DB

==================================================
RDBMS
==================================================

Choose RDBMS when the question asks about
CUSTOMER-SPECIFIC or ACCOUNT-SPECIFIC data.

Available tables:

accounts
transactions
loan_accounts
fixed_deposits
credit_cards
card_transactions

Examples:

"What is my account balance?"
-> RDBMS

"Show my transactions."
-> RDBMS

"What was my latest transaction?"
-> RDBMS

"What loans do I have?"
-> RDBMS

"What is my outstanding loan amount?"
-> RDBMS

"What is my credit card balance?"
-> RDBMS

"Show my fixed deposits."
-> RDBMS

==================================================
IMPORTANT DISTINCTION
==================================================

GENERAL INFORMATION -> VECTOR_DB

CUSTOMER-SPECIFIC INFORMATION -> RDBMS

Examples:

"What is the home loan interest rate?"
-> VECTOR_DB

"What is the interest rate on my home loan?"
-> RDBMS

"What is the transaction limit?"
-> VECTOR_DB

"What is my transaction limit?"
-> RDBMS

"What are the account charges?"
-> VECTOR_DB

"What charges were applied to my account?"
-> RDBMS

"How can I check my account balance?"
-> VECTOR_DB

"What is my account balance?"
-> RDBMS

Do not select RDBMS just because the question contains
words such as account, loan, transaction, or card.

Use the meaning of the question.

Return exactly one route and a short reason.
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
            "query": query
        }
    )

    print(
        "[ROUTER]"
        f" route={decision.route}"
        f" reason={decision.reason}"
    )

    return {
        **state,
        "route": decision.route,
    }