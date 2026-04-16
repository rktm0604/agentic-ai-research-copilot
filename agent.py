"""
agent.py — Research Agent with Reflection Pattern.

The brain of the system. Orchestrates:
  1. Query understanding
  2. Context retrieval (via rag.py)
  3. Draft response generation
  4. Self-reflection & critique
  5. Improved response generation

Uses Ollama (local) by default, with Gemini API as fallback.
Inspired by RKTM83's AgentBrain, simplified for research use.
"""

import os
import requests

from utils import (
    get_logger, OLLAMA_MODEL, OLLAMA_URL, GEMINI_API_KEY,
    GEMINI_MODEL, LLM_TIMEOUT,
)
from rag import retrieve_context, get_document_list
from memory import ConversationMemory

logger = get_logger("agent")


class ResearchAgent:
    """Agentic research assistant with RAG + Reflection."""

    def __init__(self):
        self.memory = ConversationMemory()
        self._llm_provider = "gemini" if GEMINI_API_KEY else "ollama"
        logger.info(
            "ResearchAgent initialized (provider=%s, model=%s)",
            self._llm_provider,
            GEMINI_MODEL if self._llm_provider == "gemini" else OLLAMA_MODEL,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # LLM INFERENCE
    # ═══════════════════════════════════════════════════════════════════════

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama API."""
        try:
            url = f"{OLLAMA_URL}/api/generate"
            resp = requests.post(
                url,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=LLM_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            else:
                logger.error("Ollama returned status %d", resp.status_code)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama at %s — is it running?", OLLAMA_URL)
        except Exception as e:
            logger.error("Ollama error: %s", e)
        return ""

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API with Ollama fallback."""
        if not GEMINI_API_KEY:
            return self._call_ollama(prompt)

        try:
            import google.genai as genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            return response.text.strip() if response.text else ""
        except Exception as e:
            logger.warning("Gemini failed (%s), falling back to Ollama", e)
            return self._call_ollama(prompt)

    def generate(self, prompt: str) -> str:
        """Unified LLM call — tries the configured provider."""
        if self._llm_provider == "gemini":
            return self._call_gemini(prompt)
        return self._call_ollama(prompt)

    # ═══════════════════════════════════════════════════════════════════════
    # QUERY UNDERSTANDING
    # ═══════════════════════════════════════════════════════════════════════

    def _understand_query(self, query: str) -> dict:
        """Analyze the query to determine intent and whether RAG is needed.

        Returns:
            dict with keys: "needs_rag", "refined_query", "intent"
        """
        docs = get_document_list()
        has_docs = len(docs) > 0

        # Simple heuristic first — if no documents uploaded, skip RAG
        if not has_docs:
            return {
                "needs_rag": False,
                "refined_query": query,
                "intent": "general_question",
            }

        # Use LLM to classify the query
        prompt = f"""You are a query classifier. Analyze this user query and determine:
1. Does it require searching through uploaded documents? (yes/no)
2. What is the core intent? (document_question / general_question / greeting / followup)
3. Rewrite the query to be more specific for document search (if needed).

User query: "{query}"

Available documents: {[d["name"] for d in docs]}

Reply in this EXACT format (3 lines only):
NEEDS_RAG: yes or no
INTENT: document_question or general_question or greeting or followup
REFINED: the refined query text"""

        response = self.generate(prompt)

        # Parse the classification
        needs_rag = has_docs  # default: search docs if they exist
        intent = "document_question"
        refined = query

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("NEEDS_RAG:"):
                needs_rag = "yes" in line.lower()
            elif line.startswith("INTENT:"):
                intent = line.split(":", 1)[1].strip().lower()
            elif line.startswith("REFINED:"):
                refined = line.split(":", 1)[1].strip()

        logger.info("Query analysis — needs_rag=%s, intent=%s", needs_rag, intent)
        return {"needs_rag": needs_rag, "refined_query": refined, "intent": intent}

    # ═══════════════════════════════════════════════════════════════════════
    # DRAFT GENERATION
    # ═══════════════════════════════════════════════════════════════════════

    def _generate_draft(self, query: str, context: str, citations: list[str]) -> str:
        """Generate an initial response using retrieved context."""
        memory_ctx = self.memory.get_context_string()

        if context:
            prompt = f"""You are an expert AI Research Copilot. Answer the user's question using ONLY the provided context from their uploaded documents.

CONVERSATION HISTORY:
{memory_ctx}

RETRIEVED CONTEXT FROM DOCUMENTS:
{context}

SOURCE DOCUMENTS: {', '.join(citations) if citations else 'N/A'}

USER QUESTION: {query}

INSTRUCTIONS:
- Answer based on the provided context
- Be thorough but concise
- Cite specific page numbers when possible using format: [Source: filename (p. X)]
- If the context doesn't contain enough information, say so clearly
- Use markdown formatting for readability
- Structure your answer with bullet points or numbered lists where appropriate"""
        else:
            prompt = f"""You are an expert AI Research Copilot. Answer the user's question to the best of your ability.

CONVERSATION HISTORY:
{memory_ctx}

USER QUESTION: {query}

INSTRUCTIONS:
- Provide a helpful, accurate response
- Be concise but thorough
- Note that no documents are currently uploaded, so you're answering from general knowledge
- Use markdown formatting for readability"""

        return self.generate(prompt)

    # ═══════════════════════════════════════════════════════════════════════
    # REFLECTION PATTERN (Self-Correction)
    # ═══════════════════════════════════════════════════════════════════════

    def _reflect(self, query: str, draft: str, context: str) -> tuple[bool, str]:
        """Evaluate the draft response and provide critique.

        Returns:
            Tuple of (is_good_enough, critique_text)
            - is_good_enough: True if the draft adequately answers the query
            - critique_text: Specific feedback for improvement
        """
        prompt = f"""You are a strict quality reviewer for an AI research assistant.

ORIGINAL QUESTION: {query}

DRAFT RESPONSE:
{draft}

{"AVAILABLE CONTEXT: " + context[:2000] if context else "No document context available."}

Evaluate the draft response on these criteria:
1. ACCURACY: Does it correctly use the provided context (if any)?
2. COMPLETENESS: Does it fully address the question?
3. CITATIONS: Are sources properly cited (if context was provided)?
4. CLARITY: Is it well-structured and easy to understand?

Reply in this EXACT format:
VERDICT: PASS or FAIL
CRITIQUE: Your specific feedback for improvement (1-3 sentences)"""

        response = self.generate(prompt)

        # Parse the reflection
        verdict = "PASS"
        critique = ""

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().upper()
            elif line.startswith("CRITIQUE:"):
                critique = line.split(":", 1)[1].strip()

        is_good = "PASS" in verdict
        logger.info("Reflection verdict: %s", verdict)
        if critique:
            logger.info("Critique: %s", critique[:100])

        return is_good, critique

    def _improve_response(self, query: str, draft: str, critique: str, context: str, citations: list[str]) -> str:
        """Generate an improved response based on the reflection critique."""
        prompt = f"""You are an expert AI Research Copilot. Your previous response was reviewed and needs improvement.

ORIGINAL QUESTION: {query}

YOUR PREVIOUS RESPONSE:
{draft}

REVIEWER CRITIQUE:
{critique}

{"DOCUMENT CONTEXT: " + context if context else "No document context available."}
{"SOURCES: " + ', '.join(citations) if citations else ""}

INSTRUCTIONS:
- Rewrite your response addressing the critique
- Fix any identified issues (accuracy, completeness, citations, clarity)
- Maintain a professional, research-assistant tone
- Use markdown formatting
- Cite sources with [Source: filename (p. X)] format when applicable
- Be thorough but concise"""

        return self.generate(prompt)

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN ORCHESTRATION
    # ═══════════════════════════════════════════════════════════════════════

    def process(self, query: str) -> dict:
        """Process a user query through the full agentic pipeline.

        Flow:
          1. Understand the query
          2. Retrieve context (if needed)
          3. Generate draft response
          4. Reflect on the draft
          5. If reflection fails → improve and return
          6. Store in memory

        Args:
            query: The user's question.

        Returns:
            Dict with keys: "answer", "citations", "reflection", "improved"
        """
        if not query.strip():
            return {
                "answer": "Please enter a question.",
                "citations": [],
                "reflection": None,
                "improved": False,
            }

        logger.info("═══ Processing query: '%s' ═══", query[:80])

        # Step 1: Understand the query
        analysis = self._understand_query(query)

        # Step 2: Retrieve context
        context = ""
        citations = []
        if analysis["needs_rag"]:
            search_query = analysis["refined_query"]
            context, citations = retrieve_context(search_query)
            logger.info("Retrieved context: %d chars, %d citations", len(context), len(citations))

        # Step 3: Generate draft
        logger.info("Generating draft response...")
        draft = self._generate_draft(query, context, citations)

        if not draft:
            return {
                "answer": "I couldn't generate a response. Please check that your LLM (Ollama/Gemini) is running.",
                "citations": citations,
                "reflection": None,
                "improved": False,
            }

        # Step 4: Reflect
        logger.info("Reflecting on draft...")
        is_good, critique = self._reflect(query, draft, context)

        # Step 5: Improve if needed
        final_answer = draft
        improved = False
        if not is_good and critique:
            logger.info("Improving response based on critique...")
            improved_answer = self._improve_response(query, draft, critique, context, citations)
            if improved_answer:
                final_answer = improved_answer
                improved = True
                logger.info("Response improved after reflection")
            else:
                logger.warning("Improvement generation failed, using original draft")

        # Step 6: Add citation footer
        if citations:
            citation_footer = "\n\n---\n📚 **Sources:** " + " | ".join(citations)
            final_answer += citation_footer

        # Step 7: Store in memory
        self.memory.add(query, final_answer, citations)

        result = {
            "answer": final_answer,
            "citations": citations,
            "reflection": {
                "passed": is_good,
                "critique": critique,
                "improved": improved,
            },
            "improved": improved,
        }

        logger.info("═══ Query processed (improved=%s) ═══", improved)
        return result

    def get_status(self) -> str:
        """Return a status string for the UI."""
        docs = get_document_list()
        memory_len = len(self.memory)
        provider = self._llm_provider.upper()
        model = GEMINI_MODEL if self._llm_provider == "gemini" else OLLAMA_MODEL

        lines = [
            f"🤖 **LLM Provider:** {provider} ({model})",
            f"📄 **Documents:** {len(docs)} loaded",
            f"💬 **Memory:** {memory_len}/{self.memory.max_turns} turns",
        ]
        if docs:
            lines.append("📁 **Files:**")
            for d in docs:
                lines.append(f"  - {d['name']} ({d['chunks']} chunks)")

        return "\n".join(lines)
