from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile


router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)


# ============================================================
# Project directories
# ============================================================

# upload.py is located at:
#
# src/
#   api/
#     v1/
#       routes/
#         upload.py
#
# parents[0] = routes
# parents[1] = v1
# parents[2] = api
# parents[3] = src
# parents[4] = project root
#
BASE_DIR = Path(__file__).resolve().parents[4]

DATA_DIR = BASE_DIR / "data"


# Create data directory automatically if it doesn't exist.
DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Allowed file types
# ============================================================

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
}


# ============================================================
# Upload endpoint
# ============================================================

@router.post("")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload a document and permanently store it
    inside the project's data directory.
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    # --------------------------------------------------------
    # Get file extension
    # --------------------------------------------------------

    original_filename = Path(
        file.filename
    ).name

    extension = Path(
        original_filename
    ).suffix.lower()

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                f"Allowed types: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # --------------------------------------------------------
    # Create destination path
    # --------------------------------------------------------

    file_path = DATA_DIR / original_filename

    try:

        # ----------------------------------------------------
        # Read uploaded file
        # ----------------------------------------------------

        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # ----------------------------------------------------
        # Save file permanently
        # ----------------------------------------------------

        with file_path.open("wb") as buffer:

            buffer.write(file_content)

        # ----------------------------------------------------
        # Return success response
        # ----------------------------------------------------

        return {
            "status": "uploaded",
            "filename": original_filename,
            "location": str(file_path),
            "message": (
                "File uploaded successfully "
                "and saved to the data folder."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:

        print(
            f"[upload] Error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to save uploaded document."
            ),
        ) from exc