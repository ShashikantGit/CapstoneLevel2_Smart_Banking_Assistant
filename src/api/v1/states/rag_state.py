from typing import Any, TypedDict


class RAGState(TypedDict, total=False):

    query: str

    route: str

    retrieved_docs: list[dict[str, Any]]

    reranked_docs: list[dict[str, Any]]

    sql_query: str

    sql_result: Any

    query_feedback: str

    attempts: int

    response: dict[str, Any]