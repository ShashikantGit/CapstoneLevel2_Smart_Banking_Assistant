from fastapi import APIRouter, HTTPException, Query

from app.retrieval.retrieval import retrieve

router = APIRouter(
    prefix="/query",
    tags=["Retrieval"],
)


@router.get("/")
def query_documents(
    query: str = Query(
        ...,
        min_length=1,
        description="Banking question to search in the knowledge base",
        examples=["What is the home loan interest rate?"],
    )
):
    """
    Retrieve relevant banking documents for the given question.
    """

    try:
        result = retrieve(query)

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve relevant documents.",
        )