# 🧠 Agentic AI Research Copilot

> Upload research documents • Ask questions • Get self-improving AI answers with citations

An **agentic AI system** that goes beyond simple chatbots. It uses **RAG** (Retrieval-Augmented Generation), a **Reflection pattern** for self-correcting answers, and **conversation memory** to provide accurate, context-grounded responses from your research documents.

---

## ✨ What Makes This Agentic?

This is **NOT a basic chatbot**. The system implements three key agentic AI patterns:

### 🔄 Reflection Pattern (Self-Correction)
The agent doesn't just generate an answer — it **critiques its own response** and improves it:

```
Query → Draft Response → Self-Review → Critique → Improved Response
```

If the reflection finds issues (missing citations, incomplete answers, inaccuracies), the agent automatically regenerates a better response.

### 🔍 RAG (Retrieval-Augmented Generation)
Answers are **grounded in your actual documents**, not hallucinated:

```
Query → Semantic Search → Retrieve Top Chunks → Generate Grounded Answer
```

- PDF loading with OCR fallback for scanned documents
- Smart sentence-boundary chunking with page metadata
- BGE embeddings stored in ChromaDB
- Citations with page numbers in every answer

### 💬 Conversation Memory
The agent remembers the **last 3 exchanges** for context-aware follow-ups:

```
Turn 1: "What is RAG?" → Explains RAG
Turn 2: "How does it work?" → Knows "it" refers to RAG
Turn 3: "Give me an example" → Provides RAG example
```

---

## 🏗️ Architecture

```
agentic-ai-research-copilot/
├── main.py             # FastAPI server — main entry point
├── static/index.html   # Custom HTML/CSS/JS frontend
├── agent.py            # Agent logic + Reflection pattern
├── rag.py              # RAG pipeline (PDF → chunks → vectors → search)
├── memory.py           # Conversation memory (last 3 turns)
├── utils.py            # Config, logging, constants
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # This file
```

### System Flow

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│  User Input  │────▶│              agent.py (Brain)                │
└─────────────┘     │                                              │
                    │  1. Understand query (classify intent)       │
                    │  2. Retrieve context ──▶ rag.py ──▶ ChromaDB │
                    │  3. Generate draft response (LLM)            │
                    │  4. Reflect: "Is this good enough?"          │
                    │  5. If NO → Improve based on critique        │
                    │  6. Store in memory ──▶ memory.py            │
                    └──────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────────┐
                    │           static/index.html                  │
                    │  • Chat & PDF upload interface                │
                    │  • Reflection status badges                   │
                    │  • Memory sidebar                             │
                    │  • Connected via fetch() APIs                 │
                    └──────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** installed from [ollama.ai](https://ollama.ai) (for local LLM)
- OR a **Google Gemini API key** (free tier available)

### 1. Clone & Setup

```bash
git clone https://github.com/rktm0604/agentic-ai-research-copilot.git
cd agentic-ai-research-copilot

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy the environment template
cp .env.example .env

# Edit .env to set your preferences (optional)
# Default: Ollama with llama3.2:3b
```

**Option A — Ollama (Local, Free):**
```bash
ollama pull llama3.2:3b
ollama serve   # Keep running in background
```

**Option B — Google Gemini:**
```bash
# In .env, set:
GEMINI_API_KEY=your-api-key-here
```

### 3. Run

```bash
python main.py
```

Open **http://localhost:7860** in your browser 🚀

---

## 📖 Usage

### Upload Documents
1. Click "Upload PDFs" in the left panel
2. Select one or more PDF files
3. Click "Process Documents"
4. Wait for indexing to complete

### Ask Questions
Type your question in the chat box. The agent will:
1. Search your documents for relevant context
2. Generate a draft answer
3. Self-review and improve the answer
4. Display the response with citations

### Example Queries
```
"What are the key findings in this paper?"
"Explain the methodology used in chapter 3"
"Summarize the conclusions"
"What does the author say about [topic]?"
"Compare the approaches discussed on pages 5 and 12"
```

---

## 🧠 Agentic Patterns Explained

### Pattern 1: Reflection

```python
# In agent.py — the core loop:
draft = generate_draft(query, context)     # Step 1: Generate
passed, critique = reflect(draft)           # Step 2: Self-review
if not passed:
    answer = improve(draft, critique)       # Step 3: Self-correct
```

The agent asks itself: *"Does this answer fully address the question? Are citations correct? Is it clear?"*

If the answer fails review, it regenerates using the critique as guidance.

### Pattern 2: RAG Retrieval

```python
# In rag.py — grounded answers:
context, citations = retrieve_context(query)  # Semantic search
answer = generate(query + context)             # Grounded generation
```

### Pattern 3: Memory

```python
# In memory.py — contextual follow-ups:
memory.add(query, response)                    # Store
history = memory.get_context_string()           # Retrieve for next turn
```

---

## ⚙️ Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `GEMINI_API_KEY` | *(empty)* | Google Gemini API key (enables Gemini) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence transformer model |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Number of search results |
| `MEMORY_SIZE` | `3` | Conversation turns to remember |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **UI/Backend** | FastAPI + Custom HTML/CSS/JS |
| **LLM** | Ollama (local) / Google Gemini |
| **Embeddings** | BGE-small-en-v1.5 (SentenceTransformers) |
| **Vector Store** | ChromaDB (persistent) |
| **PDF Parsing** | PyPDF + pytesseract (OCR) |
| **Config** | python-dotenv |

---

## 👨‍💻 Author

**Raktim Banerjee** — Computer Science Engineering Student

Built as part of exploring agentic AI patterns for research automation.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
