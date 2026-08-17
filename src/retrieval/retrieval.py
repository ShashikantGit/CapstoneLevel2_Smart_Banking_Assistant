from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI

from src.core.db import (
    fts_search,
    similarity_search,
)
from src.retrieval.reranker import rerank


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VECTOR_TOP_K = 20
FTS_TOP_K = 20
RRF_TOP_K = 20
FINAL_TOP_K = 5

RRF_K = 60

MAX_QUERY_RETRIES = 2


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def _vector_search(
    query: str,
    top_k: int = VECTOR_TOP_K,
) -> list[dict[str, Any]]:
    """
    Perform vector similarity search using the existing
    multimodal_chunks table.
    """

    try:
        documents = similarity_search(
            query=query,
            k=top_k,
        )

    except Exception as exc:
        print(
            "[retrieval] Vector search failed: "
            f"{exc}"
        )
        return []

    results: list[dict[str, Any]] = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        metadata = dict(
            document.get(
                "metadata",
                {},
            )
            or {}
        )

        document_name = str(
            metadata.get(
                "document_name",
                document.get(
                    "source_file",
                    "",
                ),
            )
        ).strip()

        document_id = str(
            metadata.get(
                "document_id",
                document.get(
                    "doc_id",
                    "",
                ),
            )
        ).strip()

        metadata["document_name"] = document_name
        metadata["document_id"] = document_id
        metadata["page_number"] = document.get(
            "page_number"
        )
        metadata["chunk_type"] = document.get(
            "chunk_type"
        )

        results.append(
            {
                "content": document.get(
                    "content",
                    "",
                ),
                "metadata": metadata,
                "vector_rank": index,
                "similarity": document.get(
                    "similarity"
                ),
            }
        )

    print(
        f"[retrieval] Vector results: "
        f"{len(results)}"
    )

    return results


# ---------------------------------------------------------------------------
# Stable duplicate key
# ---------------------------------------------------------------------------

def _create_result_key(
    document: dict[str, Any],
) -> str:
    """
    Create a stable key for deduplication.
    """

    metadata = document.get(
        "metadata",
        {},
    ) or {}

    document_id = str(
        metadata.get(
            "document_id",
            "",
        )
    ).strip()

    page_number = str(
        metadata.get(
            "page_number",
            "",
        )
    ).strip()

    content = str(
        document.get(
            "content",
            "",
        )
    ).strip()

    if document_id and page_number:
        return (
            f"{document_id}|"
            f"{page_number}|"
            f"{content}"
        )

    return content


# ---------------------------------------------------------------------------
# Remove duplicates
# ---------------------------------------------------------------------------

def _deduplicate(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove duplicate chunks while preserving order.
    """

    seen: set[str] = set()

    unique: list[
        dict[str, Any]
    ] = []

    for document in documents:

        key = _create_result_key(
            document
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            document
        )

    return unique


# ---------------------------------------------------------------------------
# Banking relevance boost
# ---------------------------------------------------------------------------

def _apply_banking_relevance_boost(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Give additional ranking weight to chunks containing
    banking rate/table information.
    """

    rate_keywords = {
        "interest rate",
        "interest rates",
        "rate",
        "rates",
        "repo rate",
        "%",
        "percent",
        "percentage",
        "floating rate",
        "fixed rate",
        "home loan",
        "housing loan",
        "mortgage",
    }

    table_keywords = {
        "loan amount",
        "tenure",
        "applicable rate",
        "annual rate",
        "per annum",
        "p.a.",
        "minimum",
        "maximum",
        "range",
    }

    for document in documents:

        content = str(
            document.get(
                "content",
                "",
            )
        ).lower()

        boost = 0.0

        for keyword in rate_keywords:
            if keyword in content:
                boost += 0.5

        for keyword in table_keywords:
            if keyword in content:
                boost += 1.0

        if "%" in content:
            boost += 3.0

        numeric_count = sum(
            character.isdigit()
            for character in content
        )

        if numeric_count >= 3:
            boost += 1.5

        document[
            "banking_boost"
        ] = boost

    return documents


# ---------------------------------------------------------------------------
# RRF merge
# ---------------------------------------------------------------------------

def _rrf_merge(
    vector_results: list[dict[str, Any]],
    fts_results: list[dict[str, Any]],
    top_k: int = RRF_TOP_K,
) -> list[dict[str, Any]]:
    """
    Merge vector and FTS results using
    Reciprocal Rank Fusion.
    """

    merged: dict[
        str,
        dict[str, Any],
    ] = {}

    # -----------------------------------------------------------------------
    # Vector results
    # -----------------------------------------------------------------------

    for rank, document in enumerate(
        vector_results,
        start=1,
    ):

        key = _create_result_key(
            document
        )

        if key not in merged:

            merged[key] = {
                "content": document[
                    "content"
                ],
                "metadata": document[
                    "metadata"
                ],
                "vector_rank": rank,
                "fts_rank": None,
                "rrf_score": 0.0,
            }

        merged[key]["rrf_score"] += (
            1.0 /
            (RRF_K + rank)
        )

    # -----------------------------------------------------------------------
    # FTS results
    # -----------------------------------------------------------------------

    for rank, document in enumerate(
        fts_results,
        start=1,
    ):

        key = _create_result_key(
            document
        )

        if key not in merged:

            merged[key] = {
                "content": document[
                    "content"
                ],
                "metadata": document[
                    "metadata"
                ],
                "vector_rank": None,
                "fts_rank": rank,
                "rrf_score": 0.0,
            }

        else:

            merged[key]["fts_rank"] = rank

        merged[key]["rrf_score"] += (
            1.0 /
            (RRF_K + rank)
        )

    results = list(
        merged.values()
    )

    results = _deduplicate(
        results
    )

    results = _apply_banking_relevance_boost(
        results
    )

    results.sort(
        key=lambda item: (
            item.get(
                "banking_boost",
                0.0,
            ),
            item.get(
                "rrf_score",
                0.0,
            ),
        ),
        reverse=True,
    )

    return results[:top_k]


# ---------------------------------------------------------------------------
# Alternate query generation
# ---------------------------------------------------------------------------

def _generate_alternate_queries(
    query: str,
) -> list[str]:
    """
    Generate alternate banking search queries.
    """

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        return []

    try:

        model_name = os.getenv(
            "OPENAI_CHAT_MODEL",
            "gpt-5.5",
        )

        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key,
        )

        prompt = f"""
Generate exactly two alternate search queries
for the following Smart Banking knowledge-base question.

Original question:
{query}

Requirements:

- Preserve the original meaning.
- Focus on exact banking terminology.
- If the question asks for a rate, search for:
  rate, percentage, applicable rate, interest rate,
  rate table, current rate.
- If the question asks for a fee, search for:
  processing fee, charges, applicable charges.
- Make each query suitable for vector search
  and PostgreSQL full-text search.
- Return exactly two lines.
- Do not number the lines.
- Do not add explanations.
"""

        response = llm.invoke(
            prompt
        )

        text = str(
            response.content
        ).strip()

        alternatives = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return alternatives[:2]

    except Exception as exc:

        print(
            "[retrieval] Alternate query "
            f"generation failed: {exc}"
        )

        return []


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

def _hybrid_search(
    query: str,
) -> list[dict[str, Any]]:
    """
    Perform hybrid retrieval:

        Vector Search
             +
        PostgreSQL FTS
             ↓
            RRF
             ↓
       Deduplication
             ↓
    Banking relevance boost
             ↓
          Top 20
    """

    print(
        f"[retrieval] Searching for: {query}"
    )

    # -----------------------------------------------------------------------
    # Vector search
    # -----------------------------------------------------------------------

    vector_results = _vector_search(
        query,
        VECTOR_TOP_K,
    )

    # -----------------------------------------------------------------------
    # FTS search
    # -----------------------------------------------------------------------

    try:

        fts_results = fts_search(
            query,
            FTS_TOP_K,
        )

    except Exception as exc:

        print(
            "[retrieval] FTS search failed: "
            f"{exc}"
        )

        fts_results = []

    print(
        f"[retrieval] FTS results: "
        f"{len(fts_results)}"
    )

    # -----------------------------------------------------------------------
    # RRF
    # -----------------------------------------------------------------------

    merged = _rrf_merge(
        vector_results,
        fts_results,
        RRF_TOP_K,
    )

    print(
        f"[retrieval] Hybrid results: "
        f"{len(merged)}"
    )

    return merged


# ---------------------------------------------------------------------------
# Complete retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = FINAL_TOP_K,
) -> dict[str, Any]:
    """
    Complete Smart Banking retrieval pipeline.

    Flow:

        User Query
             ↓
        Vector Search
             +
        PostgreSQL FTS
             ↓
            RRF
             ↓
       Deduplication
             ↓
    Banking relevance boost
             ↓
      Cross Encoder
             ↓
          Final Top K
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    original_query = query.strip()

    queries = [
        original_query
    ]

    # -----------------------------------------------------------------------
    # First search
    # -----------------------------------------------------------------------

    candidates = _hybrid_search(
        original_query
    )

    # -----------------------------------------------------------------------
    # Retry with alternate queries
    # -----------------------------------------------------------------------

    if not candidates:

        alternate_queries = (
            _generate_alternate_queries(
                original_query
            )
        )

        queries.extend(
            alternate_queries
        )

        for alternate_query in (
            alternate_queries
        ):

            print(
                "[retrieval] Retrying with: "
                f"{alternate_query}"
            )

            candidates = _hybrid_search(
                alternate_query
            )

            if candidates:
                break

    # -----------------------------------------------------------------------
    # No results
    # -----------------------------------------------------------------------

    if not candidates:

        return {
            "query": original_query,
            "results": [],
            "count": 0,
            "searched_queries": queries,
            "message": (
                "No relevant documents found."
            ),
        }

    # -----------------------------------------------------------------------
    # Cross encoder reranking
    # -----------------------------------------------------------------------

    try:

        results = rerank(
            query=original_query,
            documents=candidates,
            top_k=top_k,
        )

    except Exception as exc:

        print(
            "[retrieval] Reranker failed: "
            f"{exc}"
        )

        results = candidates[:top_k]

    # -----------------------------------------------------------------------
    # Final duplicate protection
    # -----------------------------------------------------------------------

    results = _deduplicate(
        results
    )

    results = results[:top_k]

    # -----------------------------------------------------------------------
    # Final rank
    # -----------------------------------------------------------------------

    for index, document in enumerate(
        results,
        start=1,
    ):
        document["rank"] = index

    return {
        "query": original_query,
        "results": results,
        "count": len(results),
        "searched_queries": queries,
    }