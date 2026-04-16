# 🧠 Agentic AI Research Copilot

**Vertical:** AI-powered Research & Document Assistant

---

## 📋 Description

An **agentic AI system** that goes beyond basic chatbots. Upload research papers, ask questions, and receive **self-correcting, citation-grounded answers** powered by RAG retrieval, the Reflection pattern, and conversation memory.

This is not a simple LLM wrapper — it implements autonomous reasoning, self-critique, and iterative improvement before delivering a response.

---

## ✨ Features

- **PDF Upload** — Drag and drop research papers, textbooks, or reports
- **RAG Retrieval** — Semantic search across uploaded documents using ChromaDB
- **Self-Correcting AI** — Reflection pattern catches and fixes errors before you see them
- **Page Citations** — Every answer includes exact source page numbers
- **Conversation Memory** — Remembers the last 3 exchanges for contextual follow-ups
- **Google Gemini Integration** — Primary LLM with local Ollama fallback
- **OCR Support** — Handles scanned PDFs via pytesseract (optional)
- **Clean Gradio UI** — Modern dark-themed interface with reflection badges

---

## 🏗️ Architecture

```
agentic-ai-research-copilot/
├── app.py              # Gradio UI — main entry point
├── agent.py            # Agent logic + Reflection pattern
├── rag.py              # RAG pipeline (PDF → chunks → vectors → search)
├── memory.py           # Conversation memory (last 3 turns)
├── utils.py            # Config, logging, constants
├── requirements.txt    # Minimal dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Keeps repo < 1 MB
└── README.md           # This file
```

### System Flow

```
User uploads PDF  ──→  PyPDF extracts text  ──→  Smart chunking  ──→  ChromaDB (vectors)
                                                                            │
User asks question ──→ agent.py ──→ retrieve_context() ──→ Draft response   │
                          │                                      │           │
                          │              ┌───────────────────────┘           │
                          │              ▼                                   │
                          │     Reflection: "Is this accurate?"             │
                          │         │ PASS → return answer                   │
                          │         │ FAIL → regenerate with critique        │
                          │              ▼                                   │
                          └──→ Final answer with citations ──→ User
```

---

## 🧠 Agentic Patterns

### 1. 🔄 Reflection Pattern (Self-Correction)

The core differentiator. After generating a draft response, the agent acts as its own reviewer:

```
Draft → LLM Review (Accuracy, Completeness, Citations, Clarity) → PASS/FAIL
    └─ If FAIL → critique → improved answer → return
    └─ If PASS → return original
```

**Implementation:** `agent.py` → `_reflect()` and `_improve_response()`

### 2. 🔍 RAG (Retrieval-Augmented Generation)

Grounds all answers in uploaded document content — eliminates hallucinations:

```
Query → BGE embeddings → ChromaDB semantic search → Top-5 chunks → LLM
```

**Implementation:** `rag.py` → `retrieve_context(query)`

### 3. 💬 Memory (Short-Term Context)

Sliding window of last 3 conversation turns, providing continuity:

```python
memory = ConversationMemory(max_turns=3)
memory.add(query, response, citations)
context = memory.get_context_string()  # Injected into every LLM prompt
```

**Implementation:** `memory.py` → `ConversationMemory`

---

## 🔗 Google Services Used

| Service | Usage |
|---------|-------|
| **Google Gemini API** | Primary LLM for reasoning, reflection, and response generation |
| **Model: gemini-2.0-flash** | Fast, high-quality inference for agentic workflows |

Gemini is the **primary** LLM provider. If `GEMINI_API_KEY` is set, all agent calls go through Gemini. If not set (or if Gemini fails), the system automatically falls back to local Ollama.

---

## 🚀 How to Run

### 1. Clone

```bash
git clone https://github.com/rktm0604/agentic-ai-research-copilot.git
cd agentic-ai-research-copilot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure (choose one)

**Option A — Google Gemini (recommended):**
```bash
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=your_key_here
```

**Option B — Local Ollama (free, no API key):**
```bash
ollama pull llama3.2:3b
ollama serve
```

### 4. Run

```bash
python app.py
```

Open **http://localhost:7860** in your browser 🚀

---

## 📖 Example Usage

### Step 1: Upload a Document
Click **"Upload PDFs"** → select a research paper → click **"📥 Process Documents"**

### Step 2: Ask a Question
```
User: What are the main findings of this study?
```

### Step 3: Get a Self-Corrected Answer
```
Agent: The study identifies three key findings:
1. ...  [Source: paper.pdf (p. 4)]
2. ...  [Source: paper.pdf (p. 7)]
3. ...  [Source: paper.pdf (p. 12)]

🔄 Self-Corrected — Agent improved its answer after reflection
> Critique: "Initial draft missed citation for finding #2"
```

### Step 4: Follow Up (Memory-Aware)
```
User: Can you elaborate on the second point?
Agent: Building on the previous answer, finding #2 specifically...
       [Source: paper.pdf (p. 7, 8)]

✅ Quality Verified — Response passed self-review
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **UI** | Gradio 5+ |
| **Primary LLM** | Google Gemini API (gemini-2.0-flash) |
| **Fallback LLM** | Ollama (llama3.2:3b, local) |
| **Embeddings** | BGE-small-en-v1.5 (SentenceTransformers) |
| **Vector Store** | ChromaDB (persistent) |
| **PDF Parsing** | PyPDF + OCR fallback |

---

## 👤 Built By

**Raktim Banerjee** — [GitHub](https://github.com/rktm0604)
