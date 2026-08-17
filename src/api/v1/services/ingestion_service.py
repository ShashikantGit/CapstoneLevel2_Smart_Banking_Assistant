from pathlib import Path

from src.ingestion.docling_parser import parse_document
from src.database.vector_repository import VectorRepository


class IngestionService:

    def __init__(self):
        self.vector_repository = VectorRepository()


    def ingest_document(
        self,
        file_path: str
    ) -> dict:

        file = Path(file_path)

        if not file.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )


        print(
            f"Starting ingestion: {file.name}"
        )


        # Step 1:
        # PDF -> Docling -> chunks

        chunks = parse_document(
            str(file)
        )


        if not chunks:
            raise RuntimeError(
                "No chunks generated from document"
            )


        print(
            f"Generated chunks: {len(chunks)}"
        )


        # Step 2:
        # Create document record

        document_id = (
            self.vector_repository.save_document(
                document_name=file.name,
                file_path=str(file)
            )
        )


        print(
            f"Document ID: {document_id}"
        )


        # Step 3:
        # Store chunks + embeddings

        stored_chunks = (
            self.vector_repository.save_chunks(
                chunks=chunks,
                document_id=document_id
            )
        )


        print(
            f"Stored chunks: {stored_chunks}"
        )


        return {

            "document_id": document_id,

            "filename": file.name,

            "total_chunks": len(chunks),

            "stored_chunks": stored_chunks

        }