from typing import Any


def ingest_document(
    file_path: str, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Ingest a document (PDF or text) for RAG pipeline."""
    import os

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        content = _extract_pdf_text(file_path)
    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    paper = {
        "title": os.path.basename(file_path),
        "content": content,
        "metadata": metadata or {},
    }

    return paper


def _extract_pdf_text(file_path: str) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages)
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF ingestion. Install: pip install pypdf"
        )
