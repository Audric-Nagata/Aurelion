import logging

from models.embedding_client import EmbeddingClient
from db.vector_store import VectorStore

logger = logging.getLogger(__name__)


def search_research_papers(query: str, top_k: int = 5) -> list[dict]:
    """Search for relevant academic papers using RAG."""
    try:
        embedder = EmbeddingClient()
        vector_store = VectorStore()

        embedding = embedder.embed(query)
        papers = vector_store.search(embedding, top_k=top_k)

        return papers
    except Exception as e:
        logger.warning("RAG search failed (pipeline will continue without papers): %s", e)
        return []
