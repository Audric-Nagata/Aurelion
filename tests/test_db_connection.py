import psycopg2
from psycopg2 import errors as pg_errors
from config import database_url
from db.vector_store import VectorStore


def test_database_connection():
    try:
        conn = psycopg2.connect(database_url)
    except psycopg2.OperationalError as e:
        raise AssertionError(
            "Could not connect to the database. "
            "Is PostgreSQL running on localhost:5432? "
            f"Try 'docker compose up db -d'.\nOriginal error: {e}"
        ) from e

    assert not conn.closed, "Connection was closed immediately after opening"
    cur = conn.cursor()
    cur.execute("SELECT 1")
    assert cur.fetchone() == (1,), "Sanity query SELECT 1 failed"
    cur.close()
    conn.close()


def test_vector_store_search():
    try:
        store = VectorStore()
        dummy_embedding = [0.0] * 1024
        results = store.search(dummy_embedding, top_k=2)
    except pg_errors.UndefinedTable as e:
        raise AssertionError(
            "The 'papers' table does not exist. "
            "Run the schema: 'psql -U postgres -d aurelion -f db/schema.sql' "
            "or restart the Docker container so the init script runs.\n"
            f"Original error: {e}"
        ) from e
    except psycopg2.OperationalError as e:
        raise AssertionError(
            "Could not connect to the database for search. "
            "Is PostgreSQL running on localhost:5432? "
            f"Try 'docker compose up db -d'.\nOriginal error: {e}"
        ) from e

    assert isinstance(results, list), (
        f"search() should return a list, got {type(results).__name__}"
    )
