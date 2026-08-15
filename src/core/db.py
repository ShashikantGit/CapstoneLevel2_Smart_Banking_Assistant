from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import re

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from langchain_community.utilities import SQLDatabase
from langchain_openai import OpenAIEmbeddings


# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

_API_KEY = os.getenv("OPENAI_API_KEY")

_PG_DSN = os.getenv("PG_CONNECTION_STRING_FTS")

_EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

pg_rdbms_connection = os.getenv(
    "PG_RDBMS_CONNECTION_STRING"
)


# ---------------------------------------------------------------------------
# Validate required configuration
# ---------------------------------------------------------------------------

if not _API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. Check your .env file."
    )

if not _PG_DSN:
    raise ValueError(
        "PG_CONNECTION_STRING_FTS is not set. Check your .env file."
    )


# ---------------------------------------------------------------------------
# OpenAI embeddings
# ---------------------------------------------------------------------------

_embeddings = OpenAIEmbeddings(
    model=_EMBED_MODEL,
    api_key=_API_KEY,
)


# ---------------------------------------------------------------------------
# SQL database connection
# ---------------------------------------------------------------------------

def get_sql_database() -> SQLDatabase:
    """
    Create a SQLDatabase connection for the banking RDBMS.

    This connection is separate from the document/vector database.
    """

    if not pg_rdbms_connection:
        raise ValueError(
            "PG_RDBMS_CONNECTION_STRING is not set. "
            "Check your .env file."
        )

    return SQLDatabase.from_uri(
        pg_rdbms_connection,
        include_tables=[
            "accounts",
            "transactions",
            "loan_accounts",
            "fixed_deposits",
            "credit_cards",
            "card_transactions",
        ],
    )


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed_texts(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings for a list of text strings.
    """

    if not texts:
        return []

    return _embeddings.embed_documents(texts)


# ---------------------------------------------------------------------------
# PostgreSQL connection pool
# ---------------------------------------------------------------------------

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """
    Return the module-level PostgreSQL connection pool.

    The pool is created lazily.
    """

    global _pool

    if _pool is None:
        _pool = ConnectionPool(
            _PG_DSN,
            min_size=2,
            max_size=10,
            kwargs={
                "row_factory": dict_row,
            },
        )

    return _pool


def get_db_conn():
    """
    Return a pooled PostgreSQL connection context manager.

    Example:

        with get_db_conn() as conn:
            with conn.cursor() as cur:
                ...
    """

    return _get_pool().connection()


# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------

def upsert_document(
    document_name: str,
    file_path: str,
) -> str:
    """
    Insert or update a document in the documents table.

    Returns:
        Document UUID as a string.
    """

    with get_db_conn() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO documents (
                    document_name,
                    file_path
                )
                VALUES (
                    %s,
                    %s
                )
                ON CONFLICT (document_name) DO UPDATE
                    SET
                        file_path = EXCLUDED.file_path,
                        updated_at = now()
                RETURNING id
                """,
                (
                    document_name,
                    file_path,
                ),
            )

            row = cur.fetchone()

        conn.commit()

    if not row:
        raise RuntimeError(
            "Unable to create or update document record."
        )

    return str(row["id"])


# ---------------------------------------------------------------------------
# Chunk storage
# ---------------------------------------------------------------------------

def store_chunks(
    chunks: list[dict],
    doc_id: str,
) -> int:
    """
    Embed and store document chunks in multimodal_chunks.

    Supported chunk types:

        text
        table
        image
    """

    if not chunks:
        return 0

    # -----------------------------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------------------------

    all_embeddings = _embed_texts(
        [
            str(chunk.get("content", ""))
            for chunk in chunks
        ]
    )

    # -----------------------------------------------------------------------
    # Metadata fields stored in dedicated columns
    # -----------------------------------------------------------------------

    dedicated_columns = {
        "content_type",
        "element_type",
        "section",
        "page_number",
        "source_file",
        "position",
        "image_base64",
    }

    rows_inserted = 0

    with get_db_conn() as conn:

        with conn.cursor() as cur:

            # ---------------------------------------------------------------
            # Delete old chunks for this document.
            #
            # This makes re-ingestion safe.
            # ---------------------------------------------------------------

            cur.execute(
                """
                DELETE FROM multimodal_chunks
                WHERE doc_id = %s::uuid
                """,
                (doc_id,),
            )

            # ---------------------------------------------------------------
            # Insert new chunks
            # ---------------------------------------------------------------

            for chunk, embedding in zip(
                chunks,
                all_embeddings,
            ):

                meta = chunk.get(
                    "metadata",
                    {},
                ) or {}

                # -----------------------------------------------------------
                # Image handling
                # -----------------------------------------------------------

                img_b64 = meta.get(
                    "image_base64"
                )

                image_path: str | None = None
                mime_type: str | None = None

                if img_b64:

                    image_bytes = base64.b64decode(
                        img_b64
                    )

                    img_dir = pathlib.Path(
                        "data/images"
                    )

                    img_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    img_hash = hashlib.sha256(
                        image_bytes
                    ).hexdigest()[:16]

                    img_file = (
                        img_dir
                        / f"{doc_id}_{img_hash}.png"
                    )

                    img_file.write_bytes(
                        image_bytes
                    )

                    image_path = str(
                        img_file
                    )

                    mime_type = "image/png"

                # -----------------------------------------------------------
                # Convert embedding to pgvector format
                # -----------------------------------------------------------

                embedding_str = (
                    "["
                    + ",".join(
                        str(value)
                        for value in embedding
                    )
                    + "]"
                )

                # -----------------------------------------------------------
                # Remaining metadata → JSONB
                # -----------------------------------------------------------

                clean_meta = {
                    key: value
                    for key, value in meta.items()
                    if key not in dedicated_columns
                }

                # -----------------------------------------------------------
                # Insert chunk
                # -----------------------------------------------------------

                cur.execute(
                    """
                    INSERT INTO multimodal_chunks (
                        doc_id,
                        chunk_type,
                        element_type,
                        content,
                        image_path,
                        mime_type,
                        page_number,
                        section,
                        source_file,
                        position,
                        embedding,
                        metadata
                    )
                    VALUES (
                        %s::uuid,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s::jsonb,
                        %s::vector,
                        %s::jsonb
                    )
                    """,
                    (
                        doc_id,
                        chunk.get(
                            "content_type",
                            "text",
                        ),
                        meta.get(
                            "element_type"
                        ),
                        chunk.get(
                            "content",
                            "",
                        ),
                        image_path,
                        mime_type,
                        meta.get(
                            "page_number"
                        ),
                        meta.get(
                            "section"
                        ),
                        meta.get(
                            "source_file"
                        ),
                        (
                            json.dumps(
                                meta.get(
                                    "position"
                                )
                            )
                            if meta.get(
                                "position"
                            )
                            else None
                        ),
                        embedding_str,
                        json.dumps(
                            clean_meta
                        ),
                    ),
                )

                rows_inserted += 1

        conn.commit()

    return rows_inserted


# ---------------------------------------------------------------------------
# Helper: convert stored image path to base64
# ---------------------------------------------------------------------------

def _attach_image_base64(
    row: dict,
) -> dict:
    """
    Convert image_path into image_base64 for API/retrieval responses.
    """

    img_path = row.pop(
        "image_path",
        None,
    )

    if (
        img_path
        and os.path.exists(img_path)
    ):

        row["image_base64"] = (
            base64.b64encode(
                pathlib.Path(
                    img_path
                ).read_bytes()
            ).decode()
        )

    else:

        row["image_base64"] = None

    return row


# ---------------------------------------------------------------------------
# Vector similarity search
# ---------------------------------------------------------------------------

def similarity_search(
    query: str,
    k: int = 5,
    chunk_type: str | None = None,
) -> list[dict]:
    """
    Find the k most similar chunks using pgvector cosine similarity.
    """

    if not query or not query.strip():
        return []

    if k <= 0:
        return []

    # -----------------------------------------------------------------------
    # Generate query embedding
    # -----------------------------------------------------------------------

    query_vec = _embed_texts(
        [query.strip()]
    )[0]

    embedding_str = (
        "["
        + ",".join(
            str(value)
            for value in query_vec
        )
        + "]"
    )

    # -----------------------------------------------------------------------
    # Optional chunk type filter
    # -----------------------------------------------------------------------

    type_clause = (
        "AND chunk_type = %(chunk_type)s"
        if chunk_type
        else ""
    )

    sql = f"""
        SELECT
            id,
            doc_id,
            content,
            chunk_type,
            page_number,
            section,
            source_file,
            element_type,
            image_path,
            mime_type,
            position,
            metadata,
            1 - (
                embedding <=> %(vec)s::vector
            ) AS similarity
        FROM multimodal_chunks
        WHERE 1=1
        {type_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """

    with get_db_conn() as conn:

        with conn.cursor() as cur:

            cur.execute(
                sql,
                {
                    "vec": embedding_str,
                    "chunk_type": chunk_type,
                    "k": k,
                },
            )

            rows = cur.fetchall()

    results = []

    for row in rows:

        row = dict(row)

        row = _attach_image_base64(
            row
        )

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# PostgreSQL Full Text Search
# ---------------------------------------------------------------------------

def fts_search(
    query: str,
    k: int = 20,
) -> list[dict]:
    """
    Perform PostgreSQL full-text search over multimodal_chunks.content.

    Uses PostgreSQL's:

        plainto_tsquery
        websearch_to_tsquery
        ts_rank

    The function returns the same general document structure
    expected by the hybrid retrieval pipeline.
    """

    if not query or not query.strip():
        return []

    if k <= 0:
        return []

    search_query = query.strip()

    # -----------------------------------------------------------------------
    # PostgreSQL FTS query
    #
    # websearch_to_tsquery is user-friendly because it handles
    # normal natural-language search terms.
    # -----------------------------------------------------------------------

    sql = """
        SELECT
            id,
            doc_id,
            content,
            chunk_type,
            page_number,
            section,
            source_file,
            element_type,
            image_path,
            mime_type,
            position,
            metadata,

            ts_rank(
                to_tsvector(
                    'english',
                    COALESCE(content, '')
                ),
                websearch_to_tsquery(
                    'english',
                    %(query)s
                )
            ) AS fts_score

        FROM multimodal_chunks

        WHERE to_tsvector(
            'english',
            COALESCE(content, '')
        ) @@ websearch_to_tsquery(
            'english',
            %(query)s
        )

        ORDER BY fts_score DESC

        LIMIT %(k)s
    """

    try:

        with get_db_conn() as conn:

            with conn.cursor() as cur:

                cur.execute(
                    sql,
                    {
                        "query": search_query,
                        "k": k,
                    },
                )

                rows = cur.fetchall()

    except Exception as exc:

        print(
            "[db] FTS search failed: "
            f"{exc}"
        )

        return []

    results = []

    for row in rows:

        row = dict(row)

        # ---------------------------------------------------------------
        # Convert database metadata into a normal dictionary.
        # ---------------------------------------------------------------

        metadata = row.get(
            "metadata",
            {},
        ) or {}

        if isinstance(
            metadata,
            str,
        ):
            try:
                metadata = json.loads(
                    metadata
                )
            except Exception:
                metadata = {}

        # ---------------------------------------------------------------
        # Add standard retrieval metadata.
        # ---------------------------------------------------------------

        metadata = dict(
            metadata
        )

        metadata["document_id"] = str(
            row.get(
                "doc_id",
                "",
            )
        )

        metadata["page_number"] = row.get(
            "page_number"
        )

        metadata["document_name"] = (
            metadata.get(
                "document_name",
                row.get(
                    "source_file",
                    "",
                ),
            )
        )

        metadata["chunk_type"] = row.get(
            "chunk_type"
        )

        # ---------------------------------------------------------------
        # Remove internal fields.
        # ---------------------------------------------------------------

        row.pop(
            "metadata",
            None,
        )

        row.pop(
            "image_path",
            None,
        )

        # ---------------------------------------------------------------
        # Build retrieval document.
        # ---------------------------------------------------------------

        results.append(
            {
                "content": row.get(
                    "content",
                    "",
                ),
                "metadata": metadata,
                "fts_score": float(
                    row.get(
                        "fts_score",
                        0.0,
                    )
                    or 0.0
                ),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Get all chunks
# ---------------------------------------------------------------------------

def get_all_chunks(
    chunk_type: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """
    Return stored chunks for preview/debugging.
    """

    if limit <= 0:
        return []

    type_clause = (
        "WHERE chunk_type = %(chunk_type)s"
        if chunk_type
        else ""
    )

    sql = f"""
        SELECT
            id,
            doc_id,
            content,
            chunk_type,
            page_number,
            section,
            source_file,
            element_type,
            image_path,
            mime_type,
            position,
            metadata
        FROM multimodal_chunks
        {type_clause}
        ORDER BY
            page_number ASC NULLS LAST,
            id ASC
        LIMIT %(limit)s
    """

    with get_db_conn() as conn:

        with conn.cursor() as cur:

            cur.execute(
                sql,
                {
                    "chunk_type": chunk_type,
                    "limit": limit,
                },
            )

            rows = cur.fetchall()

    results = []

    for row in rows:

        row = dict(row)

        row = _attach_image_base64(
            row
        )

        results.append(row)

    return results