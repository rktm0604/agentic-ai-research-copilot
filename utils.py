"""
utils.py — Shared configuration, logging, and constants.

Loads environment variables from .env and provides centralized settings
for the entire Agentic AI Research Copilot system.
"""

import os
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the project's default format."""
    return logging.getLogger(name)

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# RAG Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "research_docs"
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.environ.get("TOP_K_RESULTS", "5"))

# ---------------------------------------------------------------------------
# Memory Configuration
# ---------------------------------------------------------------------------
MEMORY_SIZE = int(os.environ.get("MEMORY_SIZE", "3"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
