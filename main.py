"""
main.py — FastAPI Backend for Agentic AI Research Copilot.

Replaces the Gradio UI with a modern, fast API serving a custom static HTML template.
Endpoints:
- POST /chat : Query the agent
- POST /upload: Upload raw PDFs
- GET /status : Retrieve ChromaDB statistics
- POST /clear : Wipe Memory + Vector DB
"""

import os
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agent import ResearchAgent
from rag import handle_upload, reset_knowledge_base, get_document_list
from utils import get_logger, UPLOAD_DIR

logger = get_logger("app")

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL AGENT INSTANCE
# ═══════════════════════════════════════════════════════════════════════════
agent = ResearchAgent()

app = FastAPI(title="Agentic Copilot API")

# Mount static directory
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
    """Serve the static index.html template on root."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>UI not found</h1>"


@app.get("/status")
async def get_status():
    """Return memory state and available knowledge base documents."""
    docs = get_document_list()
    return {"documents": docs, "memory_turns": len(agent.memory)}


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Accept PDF files from frontend, save locally, and pass to RAG pipeline."""
    saved_paths = []
    
    for file in files:
        if file.filename and file.filename.lower().endswith('.pdf'):
            dest = UPLOAD_DIR / file.filename
            with dest.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(str(dest))
    
    if not saved_paths:
        return JSONResponse(status_code=400, content={"message": "No valid PDFs found."})

    # Note: handle_upload directly expects paths, we modify its behavior since it copies internally,
    # but since we already dumped it to dest, we can bypass its internal copying or just pass dest.
    status_str = handle_upload(saved_paths)
    return {"message": status_str}


@app.post("/chat")
async def process_chat(query_data: ChatQuery):
    """Process a user question through the Agent."""
    try:
        result = agent.process(query_data.query)
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/clear")
async def clear_data():
    """Clear agent memory and vector DB."""
    agent.memory.clear()
    reset_knowledge_base()
    return {"message": "Data cleared"}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Agentic AI Research Copilot on FastAPI...")
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
