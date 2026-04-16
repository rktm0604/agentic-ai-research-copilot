"""
memory.py — Short-term conversation memory.

Stores the last N exchanges (query + response) so the agent can maintain
context across turns. Intentionally simple — no vector DB, just a deque.
"""

from collections import deque
from utils import get_logger, MEMORY_SIZE

logger = get_logger("memory")


class ConversationMemory:
    """Fixed-size sliding window of recent conversation turns."""

    def __init__(self, max_turns: int = MEMORY_SIZE):
        self._history: deque[dict] = deque(maxlen=max_turns)
        self.max_turns = max_turns
        logger.info("ConversationMemory initialized (max_turns=%d)", max_turns)

    def add(self, query: str, response: str, sources: list[str] | None = None):
        """Record a single exchange."""
        self._history.append({
            "query": query,
            "response": response,
            "sources": sources or [],
        })

    def get_history(self) -> list[dict]:
        """Return all stored turns (oldest first)."""
        return list(self._history)

    def get_context_string(self) -> str:
        """Format history as a string suitable for LLM prompts."""
        if not self._history:
            return "No previous conversation."

        parts = []
        for i, turn in enumerate(self._history, 1):
            parts.append(f"[Turn {i}]")
            parts.append(f"User: {turn['query']}")
            # Truncate long responses to keep the prompt manageable
            resp = turn["response"]
            if len(resp) > 500:
                resp = resp[:500] + "…"
            parts.append(f"Assistant: {resp}")
            parts.append("")

        return "\n".join(parts)

    def clear(self):
        """Wipe all history."""
        self._history.clear()
        logger.info("Conversation memory cleared")

    def __len__(self) -> int:
        return len(self._history)
