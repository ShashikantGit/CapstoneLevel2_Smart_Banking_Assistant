from pathlib import Path

import pytest

from src.ingestion.ingestion import DocumentIngestion


@pytest.mark.skipif(
    not list(
        Path("data/documents").glob("*.pdf")
    ),
    reason="No PDF available for ingestion test",
)
def test_pdf_ingestion():

    pdf_file = next(
        Path("data/documents").glob("*.pdf")
    )

    ingestion = DocumentIngestion()

    result = ingestion.process(pdf_file)

    assert result["filename"] == pdf_file.name

    assert result["text"]

    assert result["chunks"]

    assert result["chunk_count"] > 0