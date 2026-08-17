from pathlib import Path

from src.services.ingestion_service import (
    IngestionService
)


BASE_DIR = Path(__file__).resolve().parent.parent


DOCUMENT_DIR = (
    BASE_DIR /
    "data" /
    "documents"
)


def main():

    pdf_files = list(
        DOCUMENT_DIR.glob("*.pdf")
    )


    if not pdf_files:

        print(
            "No PDF files found"
        )

        return


    service = IngestionService()


    for pdf in pdf_files:

        result = api.v1.service.ingest_document(
            str(pdf)
        )


        print()
        print(
            "Ingestion Result"
        )
        print(
            result
        )


if __name__ == "__main__":

    main()