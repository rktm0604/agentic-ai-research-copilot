"""
app.py — Gradio UI for the Agentic AI Research Copilot.

Main entry point. Run with: python app.py
Opens at http://localhost:7860

Features:
  - PDF upload panel
  - Chat interface (Gradio messages format)
  - Agent reflection status badges
  - Sliding-window memory sidebar
"""

import os
import gradio as gr

from agent import ResearchAgent
from rag import handle_upload, reset_knowledge_base, get_document_list
from utils import get_logger, OLLAMA_MODEL, GEMINI_API_KEY, GEMINI_MODEL

logger = get_logger("app")

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL AGENT INSTANCE
# ═══════════════════════════════════════════════════════════════════════════
agent = ResearchAgent()

# ═══════════════════════════════════════════════════════════════════════════
# EVENT HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

def chat_handler(message: str, history: list) -> tuple:
    """Process a chat message through the agentic pipeline.

    Uses Gradio's messages format (list of dicts with 'role' + 'content').
    """
    if not message.strip():
        return history, "", ""

    # Add user message — dict format required for Gradio 5/6
    history.append({"role": "user", "content": message})

    # Run through the full agentic pipeline (RAG → Draft → Reflect → Improve)
    result = agent.process(message)
    answer = result["answer"]

    # Build reflection status badge text
    reflection = result.get("reflection")
    if reflection:
        if reflection["improved"]:
            badge = "🔄 **Self-Corrected** — Agent improved its answer after reflection"
            if reflection.get("critique"):
                badge += f"\n> *Critique: {reflection['critique']}*"
        elif reflection["passed"]:
            badge = "✅ **Quality Verified** — Response passed self-review"
        else:
            badge = "⚠️ Reflection ran but no improvement was needed"
    else:
        badge = ""

    # Add assistant response — dict format
    history.append({"role": "assistant", "content": answer})

    return history, "", badge


def upload_handler(files):
    """Handle PDF file uploads."""
    try:
        if not files:
            return "No files selected.", update_doc_list()
        status = handle_upload(files)
        return status, update_doc_list()
    except Exception as e:
        import traceback
        return f"❌ **Upload error:** {e}\n```\n{traceback.format_exc()}\n```", update_doc_list()


def clear_handler():
    """Clear chat, memory, and knowledge base."""
    agent.memory.clear()
    reset_knowledge_base()
    return [], "", "Knowledge base cleared. Upload new documents to begin.", update_doc_list()


def update_doc_list() -> str:
    """Generate a formatted document list."""
    docs = get_document_list()
    if not docs:
        return "📂 No documents uploaded yet"
    lines = [f"📂 **{len(docs)} document(s) indexed:**"]
    for d in docs:
        lines.append(f"  📄 {d['name']} — {d['chunks']} chunks")
    return "\n".join(lines)


def get_agent_status() -> str:
    """Get current agent status string."""
    return agent.get_status()


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
.header-section {
    text-align: center;
    padding: 20px 10px 10px;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    margin-bottom: 16px;
    border: 1px solid rgba(255,255,255,0.08);
}
.header-section h1 {
    font-size: 1.8em !important;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px !important;
    font-weight: 800 !important;
}
.header-section p { color: #94a3b8 !important; font-size: 0.95em !important; margin: 0 !important; }
.feature-pills { display: flex; justify-content: center; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
.pill {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px; padding: 5px 14px; font-size: 0.8em; color: #cbd5e1;
}
.reflection-badge {
    font-size: 0.85em; padding: 8px 12px; border-radius: 8px;
    border-left: 3px solid #a78bfa; background: rgba(167, 139, 250, 0.08); margin-top: 4px;
}
.footer {
    text-align: center; color: #64748b; font-size: 0.78em;
    padding: 10px; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 8px;
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# BUILD UI
# ═══════════════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    """Construct the Gradio interface."""
    llm_info = f"Gemini ({GEMINI_MODEL})" if GEMINI_API_KEY else f"Ollama ({OLLAMA_MODEL})"

    with gr.Blocks(title="Agentic AI Research Copilot") as demo:

        # Header
        gr.HTML(f"""
        <div class="header-section">
            <h1>🧠 Agentic AI Research Copilot</h1>
            <p>Upload research documents • Ask questions • Get AI-powered answers with citations</p>
            <div class="feature-pills">
                <span class="pill">🔍 RAG Retrieval</span>
                <span class="pill">🔄 Self-Reflection</span>
                <span class="pill">💬 Memory ({agent.memory.max_turns} turns)</span>
                <span class="pill">🤖 {llm_info}</span>
            </div>
        </div>
        """)

        with gr.Row():
            # ── Left Panel ─────────────────────────────────────
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📄 Document Upload")
                file_upload = gr.File(
                    label="Upload PDFs",
                    file_count="multiple",
                    file_types=[".pdf"],
                )
                upload_btn = gr.Button("📥 Process Documents", variant="primary")
                upload_status = gr.Markdown(value="Upload PDFs to build your knowledge base.")

                gr.Markdown("---")

                gr.Markdown("### 📊 System Status")
                doc_list = gr.Markdown(value=update_doc_list())
                status_btn = gr.Button("🔄 Refresh Status", size="sm")
                agent_status = gr.Markdown(value=get_agent_status())

                gr.Markdown("---")

                clear_btn = gr.Button("🗑️ Clear Everything", variant="stop", size="sm")

            # ── Right Panel ────────────────────────────────────
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("### 💬 Research Chat")

                # CRITICAL: initial value=[] (not None), removed type="messages" for Gradio 6 compatibility
                chatbot = gr.Chatbot(
                    label="Research Assistant",
                    value=[],
                    height=480,
                    show_label=False,
                    placeholder="Upload a document and ask a question...",
                )

                reflection_display = gr.Markdown(
                    value="",
                    elem_classes=["reflection-badge"],
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about your documents...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send 🚀", variant="primary", scale=1)

                gr.Markdown("""
                <div class="footer">
                    🧠 <b>Agentic AI Research Copilot</b> — RAG + Reflection + Memory<br/>
                    Powered by Google Gemini + ChromaDB
                </div>
                """)

        # ── Event Bindings ─────────────────────────────────────
        send_btn.click(
            fn=chat_handler,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, reflection_display],
        )
        msg_input.submit(
            fn=chat_handler,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, reflection_display],
        )
        upload_btn.click(
            fn=upload_handler,
            inputs=[file_upload],
            outputs=[upload_status, doc_list],
        )
        status_btn.click(fn=get_agent_status, outputs=[agent_status])
        clear_btn.click(
            fn=clear_handler,
            outputs=[chatbot, msg_input, upload_status, doc_list],
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

if __name__ == "__main__":
    logger.info("Starting Agentic AI Research Copilot...")
    demo = build_app()

    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=THEME,
    )
