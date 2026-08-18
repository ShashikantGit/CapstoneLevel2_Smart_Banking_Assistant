from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile


router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
}


@router.post("")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload a document for ingestion.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                f"Allowed types: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    try:
        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        with NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            temp_file.write(file_content)

            temp_file_path = Path(
                temp_file.name
            )

        try:

            # TODO:
            # Connect this to your existing
            # ingestion pipeline.
            #
            # Example:
            #
            # result = run_ingestion(
            #     str(temp_file_path)
            # )

            result = {
                "status": "uploaded",
                "filename": file.filename,
                "message": (
                    "File uploaded successfully."
                ),
            }

            return result

        finally:

            try:
                temp_file_path.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

    except HTTPException:
        raise

    except Exception as exc:

        print(
            f"[upload] Error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process uploaded document."
            ),
        ) from exc