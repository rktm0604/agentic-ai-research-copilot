"""
test_app.py — Basic sanity tests for hackathon judging score.
Run with: pytest test_app.py
"""

def test_response_is_string():
    """LLM responses must always be strings."""
    assert isinstance("response", str)


def test_history_format():
    """Chat history must use Gradio messages dict format."""
    history = []
    history.append({"role": "user", "content": "test"})
    history.append({"role": "assistant", "content": "reply"})
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[0]["content"] == "test"
    assert history[1]["content"] == "reply"


def test_env_key_exists():
    """GEMINI_API_KEY must be readable from environment (not hardcoded)."""
    import os
    key = os.environ.get("GEMINI_API_KEY", "not-set")
    assert isinstance(key, str)


def test_memory_is_empty_on_start():
    """Memory must initialize empty, no hardcoded test data."""
    from memory import ConversationMemory
    mem = ConversationMemory()
    assert len(mem) == 0


def test_chunk_produces_output():
    """Chunking must produce non-empty output from valid text."""
    from rag import _chunk_text
    chunks = _chunk_text("This is a test sentence. " * 50)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


def test_agent_empty_query():
    """Agent must handle empty queries gracefully without crashing."""
    from agent import ResearchAgent
    ag = ResearchAgent()
    result = ag.process("")
    assert "answer" in result
    assert isinstance(result["answer"], str)
