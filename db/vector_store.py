import json
from typing import Any

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PsycopgConnection
from config import database_url


class VectorStore:
    def __init__(self, connection_string: str | None = None):
        self.connection_string = connection_string or database_url
        self._conn: PsycopgConnection | None = None

    @property
    def conn(self) -> PsycopgConnection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.connection_string)
        return self._conn

    def store_paper(
        self, paper: dict[str, Any], embedding: list[float]
    ) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO papers (title, authors, abstract, content, url, metadata, embedding)
                VALUES (%(title)s, %(authors)s, %(abstract)s, %(content)s, %(url)s, %(metadata)s, %(embedding)s)
                RETURNING id
                """,
                {
                    "title": paper["title"],
                    "authors": paper.get("authors"),
                    "abstract": paper.get("abstract"),
                    "content": paper.get("content"),
                    "url": paper.get("url"),
                    "metadata": json.dumps(paper.get("metadata", {})),
                    "embedding": embedding,
                },
            )
            self.conn.commit()
            return cur.fetchone()[0]

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        with self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(
                """
                SELECT id, title, authors, abstract, content, url, metadata,
                       1 - (embedding <=> %(embedding)s::vector) AS similarity
                FROM papers
                ORDER BY similarity DESC
                LIMIT %(top_k)s
                """,
                {"embedding": query_embedding, "top_k": top_k},
            )
            return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
