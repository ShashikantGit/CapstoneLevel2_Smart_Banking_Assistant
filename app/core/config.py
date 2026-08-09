import os
from dotenv import load_dotenv
 
load_dotenv()
 
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")
PG_CONNECTION_STRING_FTS = os.getenv("PG_CONNECTION_STRING_FTS", PG_CONNECTION_STRING)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "smart_banking_kb")