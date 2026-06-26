from models.embedding_client import EmbeddingClient
from db.vector_store import VectorStore


def embed_and_store(paper: dict) -> int:
    """Generate embedding for a paper and store in pgvector."""
    embedder = EmbeddingClient()
    store = VectorStore()

    text_to_embed = (
        f"{paper.get('title', '')}\n"
        f"{paper.get('abstract', '')}\n"
        f"{paper.get('content', '')}"
    )
    embedding = embedder.embed(text_to_embed)
    paper_id = store.store_paper(paper, embedding)
    return paper_id


def chunk_and_store(
    paper: dict, chunk_size: int = 1000, overlap: int = 200
) -> list[int]:
    """Split a paper into chunks, embed each, and store."""
    embedder = EmbeddingClient()
    store = VectorStore()

    content = paper.get("content", "")
    chunks = _split_chunks(content, chunk_size, overlap)

    ids = []
    for i, chunk in enumerate(chunks):
        chunk_paper = dict(paper)
        chunk_paper["content"] = chunk
        chunk_paper["metadata"] = {
            **paper.get("metadata", {}),
            "chunk_index": i,
        }
        embedding = embedder.embed(chunk)
        paper_id = store.store_paper(chunk_paper, embedding)
        ids.append(paper_id)

    return ids


def _split_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
