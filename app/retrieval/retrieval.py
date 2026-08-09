import os
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from sentence_transformers import CrossEncoder

from app.core.db import get_vector_store

load_dotenv()

COLLECTION_NAME = "smart_banking_kb"
PG_CONNECTION_FTS = os.getenv("PG_CONNECTION_STRING_FTS") or os.getenv(
    "PG_CONNECTION_STRING"
)

VECTOR_K = 5
FTS_K = 5
RRF_K = 60
RERANK_INPUT_K = 10
FINAL_K = 5

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def vector_search(
    query: str,
    k: int = VECTOR_K,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]]:
    store = get_vector_store(collection_name)
    documents = store.similarity_search(query, k=k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]


def fts_search(
    query: str,
    k: int = FTS_K,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]]:
    if not PG_CONNECTION_FTS:
        raise ValueError("PostgreSQL connection is not configured.")

    sql = """
        SELECT
            e.document AS content,
            e.cmetadata AS metadata,
            ts_rank(
                to_tsvector('english', e.document),
                plainto_tsquery('english', %(query)s)
            ) AS fts_rank
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c
          ON c.uuid = e.collection_id
        WHERE c.name = %(collection)s
          AND to_tsvector('english', e.document)
              @@ plainto_tsquery('english', %(query)s)
        ORDER BY fts_rank DESC
        LIMIT %(k)s;
    """

    with psycopg.connect(PG_CONNECTION_FTS, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                {
                    "query": query,
                    "collection": collection_name,
                    "k": k,
                },
            )
            rows = cur.fetchall()

    return [
        {
            "content": row["content"],
            "metadata": row["metadata"] or {},
            "fts_rank": float(row["fts_rank"] or 0),
        }
        for row in rows
    ]


def hybrid_search(
    query: str,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]]:
    vector_results = vector_search(query, VECTOR_K, collection_name)
    fts_results = fts_search(query, FTS_K, collection_name)

    scores: dict[str, float] = {}
    documents: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(vector_results, start=1):
        key = result["content"]
        scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)
        documents[key] = result

    for rank, result in enumerate(fts_results, start=1):
        key = result["content"]
        scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)
        documents[key] = result

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    output = []
    for content, score in ranked[:RERANK_INPUT_K]:
        item = dict(documents[content])
        item["rrf_score"] = score
        output.append(item)

    return output


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    k: int = FINAL_K,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, item["content"]) for item in candidates]
    scores = model.predict(pairs)

    results = []
    for item, score in zip(candidates, scores):
        result = dict(item)
        result["reranker_score"] = float(score)
        results.append(result)

    results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )
    return results[:k]


def search(
    query: str,
    k: int = FINAL_K,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]]:
    if not query or not query.strip():
        return []

    candidates = hybrid_search(
        query.strip(),
        collection_name,
    )

    return rerank(
        query.strip(),
        candidates,
        k,
    )


def retrieve(
    query: str,
    alternate_queries: list[str] | None = None,
    k: int = FINAL_K,
) -> dict[str, Any]:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    queries = [query.strip()]

    if alternate_queries:
        queries.extend(
            item.strip()
            for item in alternate_queries[:2]
            if item and item.strip()
        )

    for attempt, search_query in enumerate(queries):
        results = search(search_query, k)

        if results:
            return {
                "query": query,
                "search_query": search_query,
                "results": results,
                "retry_count": attempt,
                "status": "success",
            }

    return {
        "query": query,
        "search_query": None,
        "results": [],
        "retry_count": min(len(queries) - 1, 2),
        "status": "no_relevant_documents",
    }
