import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
 
load_dotenv()
 
PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
 
 
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=1536,
    )
 
 
def get_vector_store(
    collection_name: str = "smart_banking_kb",
    pre_delete_collection: bool = False,
) -> PGVector:
    if not PG_CONNECTION:
        raise ValueError("PG_CONNECTION_STRING is not configured.")
 
    return PGVector(
        collection_name=collection_name,
        connection=PG_CONNECTION,
        embeddings=get_embeddings(),
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )