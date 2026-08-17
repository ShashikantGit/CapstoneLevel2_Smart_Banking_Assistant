from __future__ import annotations

import os
from typing import Any


# ============================================================
# IMPORTANT:
# Disable PyTorch / TorchDynamo / TorchInductor BEFORE
# importing sentence_transformers or transformers.
#
# This prevents Windows from trying to use cl.exe
# (Microsoft C++ compiler) through TorchInductor.
# ============================================================

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from sentence_transformers import CrossEncoder


# ============================================================
# Configuration
# ============================================================

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

RERANKER_TOP_K = 5


# ============================================================
# Cached CrossEncoder
# ============================================================

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """
    Load the CrossEncoder reranker once and reuse it.

    The model runs on CPU and Torch compilation is disabled
    to avoid the Windows cl.exe / TorchInductor issue.
    """

    global _reranker

    if _reranker is None:

        print(
            "[RERANKER] Loading CrossEncoder model..."
        )

        _reranker = CrossEncoder(
            RERANKER_MODEL,
            trust_remote_code=False,
            device="cpu",
        )

        print(
            "[RERANKER] CrossEncoder model loaded."
        )

    return _reranker


# ============================================================
# Reranking
# ============================================================

def rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = RERANKER_TOP_K,
) -> list[dict[str, Any]]:
    """
    Rerank hybrid retrieval results using a CrossEncoder.

    Pipeline:

        Vector Search
             +
        PostgreSQL FTS
             ↓
            RRF
             ↓
        CrossEncoder
             ↓
          Top K
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    # --------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------

    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than zero."
        )

    # --------------------------------------------------------
    # No documents
    # --------------------------------------------------------

    if not documents:

        return []

    # --------------------------------------------------------
    # Prepare valid documents
    # --------------------------------------------------------

    valid_documents: list[
        dict[str, Any]
    ] = []

    pairs: list[
        tuple[str, str]
    ] = []

    clean_query = query.strip()

    for document in documents:

        content = str(
            document.get(
                "content",
                "",
            )
        ).strip()

        # Ignore documents without content
        if not content:

            continue

        valid_documents.append(
            document
        )

        pairs.append(
            (
                clean_query,
                content,
            )
        )

    # --------------------------------------------------------
    # No valid documents
    # --------------------------------------------------------

    if not pairs:

        return []

    # --------------------------------------------------------
    # Limit top_k to available documents
    # --------------------------------------------------------

    top_k = min(
        top_k,
        len(valid_documents),
    )

    # --------------------------------------------------------
    # Load CrossEncoder
    # --------------------------------------------------------

    model = get_reranker()

    # --------------------------------------------------------
    # Calculate relevance scores
    # --------------------------------------------------------

    print(
        "[RERANKER] Calculating relevance scores..."
    )

    scores = model.predict(
        pairs,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # --------------------------------------------------------
    # Attach scores
    # --------------------------------------------------------

    scored_documents: list[
        dict[str, Any]
    ] = []

    for document, score in zip(
        valid_documents,
        scores,
    ):

        result = dict(document)

        result["rerank_score"] = float(
            score
        )

        scored_documents.append(
            result
        )

    # --------------------------------------------------------
    # Sort by highest relevance
    # --------------------------------------------------------

    scored_documents.sort(
        key=lambda item: item[
            "rerank_score"
        ],
        reverse=True,
    )

    # --------------------------------------------------------
    # Select final Top K
    # --------------------------------------------------------

    final_results = scored_documents[
        :top_k
    ]

    # --------------------------------------------------------
    # Assign final ranking
    # --------------------------------------------------------

    for index, document in enumerate(
        final_results,
        start=1,
    ):

        document["rank"] = index

    print(
        f"[RERANKER] Final results: "
        f"{len(final_results)}"
    )

    return final_results