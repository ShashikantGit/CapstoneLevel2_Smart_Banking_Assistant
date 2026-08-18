from fastapi import APIRouter, HTTPException

from src.api.v1.agents.rag_agent import run_search_agent
from src.api.v1.schemas.query_schema import QueryRequest


router = APIRouter(
    prefix="/query",
    tags=["Agent"],
)


@router.post("")
def query_agent(
    request: QueryRequest,
):
    """
    Run the Smart Banking LangGraph agent.
    """

    try:
        result = run_search_agent(
            query=request.query,
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        import traceback

        print("\n========== AGENT ERROR ==========")
        print(f"Exception: {exc}")
        traceback.print_exc()
        print("=================================\n")

        raise HTTPException(
            status_code=500,
            detail="Unable to process banking query.",
        ) from exc