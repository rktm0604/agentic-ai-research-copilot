"""
rag.py — Retrieval-Augmented Generation pipeline.

Handles the full document lifecycle:
  1. Load PDFs (native text extraction + OCR fallback)
  2. Smart chunking with page-number metadata
  3. Embed and store in ChromaDB
  4. Semantic search to retrieve relevant context

Inspired by the RAG Study Assistant project, simplified for clarity.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from utils import (
    get_logger, EMBEDDING_MODEL, CHROMA_DB_PATH,
    COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, UPLOAD_DIR,
)

logger = get_logger("rag")

# ---------------------------------------------------------------------------
# Optional OCR support
# ---------------------------------------------------------------------------
OCR_AVAILABLE = True
try:
    from pdf2image import convert_from_path
    import pytesseract
except ImportError:
    OCR_AVAILABLE = False
    logger.info("OCR not available — install pytesseract + pdf2image for scanned PDF support")

# ---------------------------------------------------------------------------
# Embedding function (singleton)
# ---------------------------------------------------------------------------
_embedding_fn = None

def _get_embedding_function():
    """Return a cached BGE embedding function."""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL, device="cpu"
        )
        logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)
    return _embedding_fn


# ═══════════════════════════════════════════════════════════════════════════
# 1. DOCUMENT LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text from a PDF, preserving page numbers.

    Tries PyPDF text extraction first. If the PDF is scanned/image-based
    and no text is found, falls back to OCR via pytesseract + pdf2image.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of (page_number, page_text) tuples (1-indexed).

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If no text could be extracted.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: '{pdf_path}'")

    # --- Attempt 1: native text extraction ---
    reader = PdfReader(pdf_path)
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append((idx, text))

    if pages:
        total_chars = sum(len(t) for _, t in pages)
        logger.info("Loaded '%s' (text mode): %d pages, %d chars", path.name, len(pages), total_chars)
        return pages

    # --- Attempt 2: OCR fallback ---
    if not OCR_AVAILABLE:
        raise ValueError(
            f"No text found in '{path.name}'. PDF appears to be scanned.\n"
            "Install OCR: pip install pytesseract pdf2image"
        )

    logger.info("No text via PyPDF — falling back to OCR for '%s'", path.name)
    try:
        images = convert_from_path(str(path))
    except Exception as e:
        raise ValueError(f"OCR failed for '{path.name}': {e}") from e

    pages = []
    for idx, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image)
        if text and text.strip():
            pages.append((idx, text))

    if not pages:
        raise ValueError(f"No text extracted from '{path.name}' even with OCR.")

    logger.info("Loaded '%s' (OCR mode): %d pages", path.name, len(pages))
    return pages


# ═══════════════════════════════════════════════════════════════════════════
# 2. CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    if not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        if end < len(text):
            search_start = max(start, end - 200)
            region = text[search_start:end]
            for sep in [". ", ".\n", "? ", "! ", "\n\n"]:
                last_break = region.rfind(sep)
                if last_break != -1:
                    end = search_start + last_break + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        step = max(1, chunk_size - overlap)
        start = start + step if end >= start + chunk_size else end

    return chunks


def chunk_with_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Chunk page-annotated text, tracking which pages each chunk spans.

    Returns:
        List of dicts: {"text": str, "pages": list[int]}
    """
    if not pages:
        return []

    # Build full text with character→page mapping
    full_text = ""
    char_to_page: list[int] = []
    for page_num, page_text in pages:
        segment = page_text + "\n"
        char_to_page.extend([page_num] * len(segment))
        full_text += segment

    raw_chunks = _chunk_text(full_text, chunk_size, overlap)

    # Map each chunk back to its source pages
    result = []
    search_start = 0
    for chunk in raw_chunks:
        idx = full_text.find(chunk, search_start)
        if idx == -1:
            idx = full_text.find(chunk)

        if idx != -1:
            end_idx = min(idx + len(chunk), len(char_to_page))
            chunk_pages = sorted({char_to_page[ci] for ci in range(idx, end_idx)})
            result.append({"text": chunk, "pages": chunk_pages})
            search_start = idx + 1
        else:
            result.append({"text": chunk, "pages": []})

    logger.info("Created %d chunks (size=%d, overlap=%d)", len(result), chunk_size, overlap)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3. VECTOR STORE
# ═══════════════════════════════════════════════════════════════════════════

_client = None

def _get_client():
    """Get/create persistent ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def _get_collection():
    """Get or create the document collection."""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_function(),
    )


def add_document(pdf_path: str) -> dict:
    """Process a PDF and add it to the knowledge base.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with status info: {"filename", "pages", "chunks", "status"}
    """
    path = Path(pdf_path)
    filename = path.name

    try:
        pages = load_pdf(pdf_path)
        chunks = chunk_with_pages(pages)

        if not chunks:
            return {"filename": filename, "pages": 0, "chunks": 0, "status": "error: no text extracted"}

        collection = _get_collection()

        # Use filename + chunk index as unique IDs (allows re-uploading)
        doc_prefix = filename.replace(" ", "_").replace(".", "_")
        ids = [f"{doc_prefix}_chunk_{i}" for i in range(len(chunks))]

        # Delete old chunks from same file (if re-uploaded)
        try:
            existing = collection.get(where={"source": filename})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
                logger.info("Removed %d old chunks for '%s'", len(existing["ids"]), filename)
        except Exception:
            pass

        collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[{
                "pages": ",".join(str(p) for p in c["pages"]),
                "source": filename,
            } for c in chunks],
            ids=ids,
        )

        total_pages = max(p for _, p_list in [(None, c["pages"]) for c in chunks] for p in (p_list or [0]))
        logger.info("Added '%s': %d pages, %d chunks", filename, total_pages, len(chunks))

        return {
            "filename": filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "status": "success",
        }

    except Exception as e:
        logger.error("Failed to process '%s': %s", filename, e)
        return {"filename": filename, "pages": 0, "chunks": 0, "status": f"error: {e}"}


def retrieve_context(query: str, top_k: int = TOP_K_RESULTS) -> tuple[str, list[str]]:
    """Retrieve relevant context from the knowledge base.

    This is the single entry point for the agent to get document context.

    Args:
        query: The user's question.
        top_k: Number of top results to return.

    Returns:
        Tuple of (context_text, citations) where citations are like
        ["document.pdf (p. 3, 5)", "notes.pdf (p. 12)"]
    """
    collection = _get_collection()

    if collection.count() == 0:
        return "", []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        return "", []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "", []

    # Build context text
    context = "\n\n---\n\n".join(documents)

    # Build citation strings grouped by source file
    source_pages: dict[str, set[int]] = {}
    for meta in metadatas:
        source = meta.get("source", "unknown")
        pages_str = meta.get("pages", "")
        if source not in source_pages:
            source_pages[source] = set()
        if pages_str:
            for p in pages_str.split(","):
                p = p.strip()
                if p:
                    source_pages[source].add(int(p))

    citations = []
    for source, pages in source_pages.items():
        if pages:
            page_list = ", ".join(str(p) for p in sorted(pages))
            citations.append(f"{source} (p. {page_list})")
        else:
            citations.append(source)

    logger.info("Retrieved %d chunks for: '%s'", len(documents), query[:60])
    return context, citations


def get_document_list() -> list[dict]:
    """Return a list of all documents in the knowledge base."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    try:
        all_data = collection.get()
        sources: dict[str, int] = {}
        for meta in all_data.get("metadatas", []):
            source = meta.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1

        return [{"name": name, "chunks": count} for name, count in sources.items()]
    except Exception:
        return []


def reset_knowledge_base():
    """Delete all documents and start fresh."""
    client = _get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("Knowledge base reset")
    except Exception:
        pass
    # Also clear the upload directory
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                f.unlink()


def handle_upload(files) -> str:
    """Process uploaded files and return a status message.

    Args:
        files: List of file paths from Gradio upload component.

    Returns:
        Human-readable status string.
    """
    if not files:
        return "No files uploaded."

    results = []
    for file_path in files:
        path = Path(file_path)
        if path.suffix.lower() != ".pdf":
            results.append(f"⚠️ Skipped '{path.name}' — only PDFs are supported.")
            continue

        # Copy to uploads directory
        dest = UPLOAD_DIR / path.name
        shutil.copy2(str(path), str(dest))

        # Process the document
        info = add_document(str(dest))
        if info["status"] == "success":
            results.append(f"✅ **{info['filename']}** — {info['pages']} pages, {info['chunks']} chunks indexed")
        else:
            results.append(f"❌ **{info['filename']}** — {info['status']}")

    return "\n".join(results)
