from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from src.api.v1.agents.rag_agent import (
    run_search_agent,
)


router = APIRouter(
    prefix="/query",
    tags=["Agent"],
)


from src.api.v1.schemas.query_schema import (
    QueryRequest,
)


@router.post("")
def query_agent(
    request: QueryRequest,
):

    try:

        return run_search_agent(
            query=request.query,
            top_k=request.top_k,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print(
            f"[agent] Error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process "
                "banking query."
            ),
        ) from exc
    """
    Run the Smart Banking LangGraph agent.
    """

    try:

        return run_search_agent(
            query=query,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print(
            f"[agent] Error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process banking query.",
        ) from exc