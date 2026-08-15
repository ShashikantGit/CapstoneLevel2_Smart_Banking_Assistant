from __future__ import annotations

from typing import Any

from src.retrieval.retrieval import retrieve


def retrieve_documents(
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant PDF knowledge-base documents.

    Retrieval pipeline:

        Vector Search
             +
        PostgreSQL FTS
             ↓
            RRF
             ↓
       Banking Boost
             ↓
        Cross Encoder
             ↓
          Top K
    """

    if not query or not query.strip():
        return []

    result = retrieve(
        query=query.strip(),
        top_k=top_k,
    )

    return result.get(
        "results",
        [],
    )