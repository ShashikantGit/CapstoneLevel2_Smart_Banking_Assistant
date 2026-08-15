from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from src.retrieval.retrieval import retrieve


router = APIRouter(
    prefix="/query",
    tags=["Retrieval"],
)


@router.get("/")
def query_documents(
    query: str = Query(
        ...,
        min_length=1,
        description=(
            "Banking question to search "
            "in the knowledge base"
        ),
        examples=[
            "What is the home loan interest rate?"
        ],
    ),
    top_k: int = Query(
        5,
        ge=1,
        le=20,
        description=(
            "Number of final chunks to return."
        ),
    ),
):
    """
    Retrieve relevant banking documents.
    """

    try:

        return retrieve(
            query=query,
            top_k=top_k,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print(
            f"[retrieval] Error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve "
                "relevant documents."
            ),
        ) from exc