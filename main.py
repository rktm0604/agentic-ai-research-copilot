"""
main.py — FastAPI Backend for Agentic AI Research Copilot.

Serves a custom HTML frontend and exposes API endpoints for:
  - POST /chat   : Send a query to the agent
  - POST /upload : Upload PDF documents
  - GET  /status : Get knowledge base & memory stats
  - POST /clear  : Wipe memory + vector DB

Run with: python main.py
"""

import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agent import ResearchAgent
from rag import handle_upload, reset_knowledge_base, get_document_list
from utils import get_logger, UPLOAD_DIR

logger = get_logger("app")

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL AGENT
# ═══════════════════════════════════════════════════════════════════════════
agent = ResearchAgent()

app = FastAPI(title="Agentic AI Research Copilot")

STATIC_DIR = Path(__file__).parent / "static"


# ═══════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════

class ChatQuery(BaseModel):
    query: str


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the custom HTML frontend."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>UI not found — ensure static/index.html exists</h1>"


@app.get("/status")
async def get_status():
    """Return knowledge base documents and memory count."""
    docs = get_document_list()
    return {
        "documents": docs,
        "memory_turns": len(agent.memory),
        "max_turns": agent.memory.max_turns,
    }


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Accept PDF uploads, save to disk, and index into ChromaDB."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            continue
        dest = UPLOAD_DIR / file.filename
        with dest.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
        saved_paths.append(str(dest))

    if not saved_paths:
        return JSONResponse(
            status_code=400,
            content={"message": "No valid PDF files found in upload."},
        )

    status_str = handle_upload(saved_paths)
    return {"message": status_str}


@app.post("/chat")
async def process_chat(body: ChatQuery):
    """Run a query through the full agentic pipeline."""
    try:
        result = agent.process(body.query)
        return result
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/clear")
async def clear_data():
    """Wipe conversation memory and the vector store."""
    agent.memory.clear()
    reset_knowledge_base()
    return {"message": "Knowledge base and memory cleared."}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Agentic AI Research Copilot (FastAPI)...")
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
