from models.embedding_client import EmbeddingClient
from db.vector_store import VectorStore


def search_research_papers(query: str, top_k: int = 5) -> list[dict]:
    """Search for relevant academic papers using RAG."""
    embedder = EmbeddingClient()
    vector_store = VectorStore()

    embedding = embedder.embed(query)
    papers = vector_store.search(embedding, top_k=top_k)

    return papers
