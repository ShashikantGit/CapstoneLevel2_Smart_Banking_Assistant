from src.core import db


class VectorRepository:
    """
    Repository layer for vector database operations.

    This class delegates actual database operations
    to trainer's db.py implementation.
    """

    def save_document(
        self,
        document_name: str,
        file_path: str,
    ) -> str:

        return db.upsert_document(
            document_name=document_name,
            file_path=file_path,
        )


    def save_chunks(
        self,
        chunks: list[dict],
        document_id: str,
    ) -> int:

        return db.store_chunks(
            chunks=chunks,
            doc_id=document_id,
        )


    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict]:

        return db.similarity_search(
            query=query,
            k=k,
        )


    def full_text_search(
        self,
        query: str,
        k: int = 20,
    ) -> list[dict]:

        return db.fts_search(
            query=query,
            k=k,
        )


    def get_all_chunks(
        self,
        limit: int = 200,
    ) -> list[dict]:

        return db.get_all_chunks(
            limit=limit
        )