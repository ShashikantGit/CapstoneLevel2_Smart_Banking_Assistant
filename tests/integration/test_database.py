from sqlalchemy import text

from src.core.db import engine


def test_database_connection():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))

        assert result.scalar() == 1


def test_pgvector_extension():
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM pg_extension
                WHERE extname = 'vector'
            """)
        )

        assert result.scalar() == 1


def test_required_tables_exist():
    required_tables = {
        "documents",
        "document_chunks",
        "customers",
        "accounts",
        "transactions",
        "beneficiaries",
    }

    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
        )

        existing_tables = {row[0] for row in result}

    assert required_tables.issubset(existing_tables)


def test_seed_customers():
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM customers
                WHERE customer_number IN (
                    'CUST10001',
                    'CUST10002'
                )
            """)
        )

        assert result.scalar() == 2


def test_embedding_column():
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c
                    ON a.attrelid = c.oid
                WHERE c.relname = 'document_chunks'
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
            """)
        )

        assert result.scalar() == "vector(1536)"