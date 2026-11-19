"""Tests for ConversationMemoryProvider."""

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from plugins.memory.conversation_memory import ConversationMemoryProvider


class TestConversationMemoryProvider:
    """Test the ConversationMemoryProvider implementation."""

    @pytest.fixture
    def temp_home(self):
        """Create a temporary HERMES_HOME for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def provider(self, temp_home):
        """Create a provider instance with test config."""
        config = {
            "max_context_chars": 2000,
            "max_turns_per_session": 100,
            "fts_enabled": True,
            "auto_extract": True,
        }
        p = ConversationMemoryProvider(config)
        p.initialize(session_id="test-session", hermes_home=str(temp_home))
        yield p
        p.shutdown()

    def test_provider_name(self, provider):
        assert provider.name == "conversation_memory"

    def test_is_available(self, provider):
        assert provider.is_available() is True

    def test_initialize_creates_db(self, provider, temp_home):
        db_path = temp_home / "conversation_memory" / "conversation_memory.db"
        assert db_path.exists()

    def test_system_prompt_block(self, provider):
        block = provider.system_prompt_block()
        assert "Conversation Memory" in block
        assert "conversation_memory_search" in block

    def test_sync_turn_persists_turns(self, provider):
        provider.sync_turn("Hello", "Hi there!", session_id="test-session")
        provider._db.flush(timeout=2.0)

        # Verify turns were written
        results = provider._db.recent("test-session", limit=10)
        assert len(results) == 2
        assert results[0].role == "user"
        assert results[0].content == "Hello"
        assert results[1].role == "assistant"
        assert results[1].content == "Hi there!"

    def test_sync_turn_increments_turn_count(self, provider):
        assert provider._turn_count == 0
        provider.sync_turn("User 1", "Assistant 1", session_id="test-session")
        assert provider._turn_count == 2  # user + assistant
        provider.sync_turn("User 2", "Assistant 2", session_id="test-session")
        assert provider._turn_count == 4

    def test_search_returns_ranked_results(self, provider):
        # Add some test data
        provider.sync_turn("I like apples", "Apples are great", session_id="test-session")
        provider.sync_turn("Bananas are yellow", "Yes, bananas are yellow fruits", session_id="test-session")
        provider.sync_turn("Apple pie recipe", "Here is an apple pie recipe", session_id="test-session")
        provider._db.flush(timeout=2.0)

        # Search for "apple"
        results = provider._db.search("apple", session_id="test-session", limit=5)
        assert len(results) >= 2
        # Should find both apple-related turns
        contents = [r.content.lower() for r in results]
        assert any("apple" in c for c in contents)

    def test_search_respects_session_filter(self, provider):
        provider.sync_turn("Session A content", "Response A", session_id="session-a")
        provider.sync_turn("Session B content", "Response B", session_id="session-b")
        provider._db.flush(timeout=2.0)

        results_a = provider._db.search("content", session_id="session-a", limit=5)
        results_b = provider._db.search("content", session_id="session-b", limit=5)

        assert all(r.session_id == "session-a" for r in results_a)
        assert all(r.session_id == "session-b" for r in results_b)

    def test_recent_returns_recent_turns(self, provider):
        provider.sync_turn("First", "Response 1", session_id="test-session")
        provider.sync_turn("Second", "Response 2", session_id="test-session")
        provider.sync_turn("Third", "Response 3", session_id="test-session")
        provider._db.flush(timeout=2.0)

        results = provider._db.recent("test-session", limit=2)
        assert len(results) == 2
        # Should be in chronological order (recent returns reversed)
        assert results[0].content == "Second"
        assert results[1].content == "Third"

    def test_on_memory_write_mirrors_builtin(self, provider):
        provider.on_memory_write("add", "user", "User prefers dark mode")
        provider.on_memory_write("add", "memory", "Agent should be concise")
        provider._db.flush(timeout=2.0)

        results = provider._db.recent("test-session", limit=10)
        # Should have the mirrored writes
        tool_turns = [r for r in results if r.role == "tool"]
        assert len(tool_turns) >= 2
        contents = [t.content for t in tool_turns]
        assert any("User prefers dark mode" in c for c in contents)
        assert any("Agent should be concise" in c for c in contents)

    def test_on_pre_compress_skips_compressed_messages(self, provider):
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2", "_compressed_summary": True},  # Should be skipped
            {"role": "assistant", "content": "Response 2"},
        ]
        provider.on_pre_compress(messages)
        provider._db.flush(timeout=2.0)

        results = provider._db.recent("test-session", limit=10)
        tool_turns = [r for r in results if r.role == "tool" and "Pre-compression" in r.content]
        assert len(tool_turns) == 1
        # Should only contain non-compressed messages (Message 1, Response 1, Response 2)
        content = tool_turns[0].content
        assert "Message 1" in content
        assert "Response 1" in content
        assert "Response 2" in content
        assert "Message 2" not in content  # Skipped due to _compressed_summary

    def test_on_pre_compress_empty_returns_empty(self, provider):
        result = provider.on_pre_compress([])
        assert result == ""

        result = provider.on_pre_compress([{"role": "user", "content": "test", "_compressed_summary": True}])
        assert result == ""

    def test_on_session_end_flushes(self, provider):
        provider.sync_turn("Test", "Response", session_id="test-session")
        # Should not block
        provider.on_session_end([])

    def test_on_session_switch_rebinds(self, provider):
        # Add some turns to original session
        provider.sync_turn("Old session", "Response", session_id="test-session")
        provider._db.flush(timeout=2.0)

        # Switch to new session
        provider.on_session_switch("new-session")
        assert provider._session_id == "new-session"
        assert provider._turn_count == 0

        # New session should exist
        results = provider._db.recent("new-session", limit=5)
        assert len(results) == 0

    def test_on_session_switch_resumes_turn_count(self, provider):
        provider.sync_turn("Turn 1", "Response 1", session_id="test-session")
        provider.sync_turn("Turn 2", "Response 2", session_id="test-session")
        provider._db.flush(timeout=2.0)

        # Switch without reset
        provider.on_session_switch("test-session")
        assert provider._turn_count == 4  # Resumed from DB

    def test_identity_signature(self, provider):
        sig = provider.identity_signature()
        assert sig["provider"] == "conversation_memory"
        assert "db_path" in sig

    def test_get_tool_schemas(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 5
        names = {s["name"] for s in schemas}
        assert names == {
            "conversation_memory_search",
            "conversation_memory_recent",
            "conversation_memory_save",
            "conversation_memory_status",
            "conversation_memory_clear",
        }

    def test_handle_tool_call_search(self, provider):
        provider.sync_turn("I love Python", "Python is great", session_id="test-session")
        provider._db.flush(timeout=2.0)

        result = json.loads(provider.handle_tool_call("conversation_memory_search", {"query": "Python"}))
        assert "results" in result
        assert len(result["results"]) >= 1

    def test_handle_tool_call_recent(self, provider):
        provider.sync_turn("Recent test", "Response", session_id="test-session")
        provider._db.flush(timeout=2.0)

        result = json.loads(provider.handle_tool_call("conversation_memory_recent", {"session_id": "test-session"}))
        assert "turns" in result
        assert len(result["turns"]) >= 1

    def test_handle_tool_call_save(self, provider):
        result = json.loads(provider.handle_tool_call("conversation_memory_save", {
            "content": "Important fact: user likes cats",
            "tags": ["preference", "pets"],
            "importance": 0.9
        }))
        assert result["result"] == "Saved to conversation memory"

        # Verify it was saved
        provider._db.flush(timeout=2.0)
        results = provider._db.recent("test-session", limit=5)
        tool_turns = [r for r in results if r.role == "tool"]
        assert any("Important fact" in t.content for t in tool_turns)

    def test_handle_tool_call_status(self, provider):
        result = json.loads(provider.handle_tool_call("conversation_memory_status", {}))
        assert "sessions" in result
        assert "turns" in result
        assert "db_size_bytes" in result
        assert "fts_enabled" in result

    def test_handle_tool_call_clear_requires_confirm(self, provider):
        result = json.loads(provider.handle_tool_call("conversation_memory_clear", {"session_id": "test-session"}))
        assert "error" in result
        assert "confirm=true" in result["error"]

    def test_handle_tool_call_clear_with_confirm(self, provider):
        provider.sync_turn("To be cleared", "Response", session_id="test-session")
        provider._db.flush(timeout=2.0)

        result = json.loads(provider.handle_tool_call("conversation_memory_clear", {
            "session_id": "test-session",
            "confirm": True
        }))
        assert "Cleared session" in result["result"]

    def test_shutdown_cleans_up(self, provider):
        provider.sync_turn("Shutdown test", "Response", session_id="test-session")
        provider.shutdown()
        # Should not raise on double shutdown
        provider.shutdown()

    def test_prefetch_trivial_prompt_returns_empty(self, provider):
        result = provider.prefetch("hi")
        assert result == ""

        result = provider.prefetch("ok")
        assert result == ""

        result = provider.prefetch("/help")
        assert result == ""

    def test_queue_prefetch_populates_cache(self, provider):
        provider.sync_turn("Searchable content", "Response", session_id="test-session")
        provider._db.flush(timeout=2.0)

        provider.queue_prefetch("Searchable", session_id="test-session")
        # Wait for background thread
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=2.0)

        result = provider.prefetch("Searchable", session_id="test-session")
        assert "Searchable content" in result

    def test_recall_status_reflects_last_prefetch(self, provider):
        assert provider.recall_status() is None

        provider.sync_turn("Test content", "Response", session_id="test-session")
        provider._db.flush(timeout=2.0)

        provider.queue_prefetch("Test", session_id="test-session")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=2.0)

        provider.prefetch("Test", session_id="test-session")
        status = provider.recall_status()
        assert status is not None
        assert status.provider_label == "conversation_memory"
        assert status.count == 1

    def test_config_schema(self, provider):
        schema = provider.get_config_schema()
        assert len(schema) == 4
        keys = {s["key"] for s in schema}
        assert keys == {"max_context_chars", "max_turns_per_session", "fts_enabled", "auto_extract"}


class TestMultiProfileIsolation:
    """Test that multiple provider instances (different profiles) don't share state."""

    def test_separate_instances_have_separate_dbs(self, temp_home):
        """Two provider instances initialized with different hermes_home should have separate databases."""
        home1 = temp_home / "profile1"
        home2 = temp_home / "profile2"
        home1.mkdir()
        home2.mkdir()

        p1 = ConversationMemoryProvider({"auto_extract": True})
        p1.initialize(session_id="session1", hermes_home=str(home1))

        p2 = ConversationMemoryProvider({"auto_extract": True})
        p2.initialize(session_id="session2", hermes_home=str(home2))

        p1.sync_turn("Profile 1 data", "Response", session_id="session1")
        p2.sync_turn("Profile 2 data", "Response", session_id="session2")

        p1._db.flush(timeout=2.0)
        p2._db.flush(timeout=2.0)

        # Each should only see its own data
        results1 = p1._db.recent("session1", limit=5)
        results2 = p2._db.recent("session2", limit=5)

        assert any("Profile 1 data" in r.content for r in results1)
        assert not any("Profile 2 data" in r.content for r in results1)
        assert any("Profile 2 data" in r.content for r in results2)
        assert not any("Profile 1 data" in r.content for r in results2)

        p1.shutdown()
        p2.shutdown()

    def test_concurrent_access_thread_safety(self, temp_home):
        """Multiple threads using the same provider instance should not corrupt data."""
        p = ConversationMemoryProvider({"auto_extract": True, "max_turns_per_session": 1000})
        p.initialize(session_id="concurrent-test", hermes_home=str(temp_home))

        def write_turns(thread_id: int):
            for i in range(10):
                p.sync_turn(f"Thread {thread_id} turn {i}", f"Response {i}", session_id="concurrent-test")
                time.sleep(0.001)

        threads = [threading.Thread(target=write_turns, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        p._db.flush(timeout=2.0)
        results = p._db.recent("concurrent-test", limit=100)
        # Should have 5 threads * 10 turns * 2 (user+assistant) = 100 turns
        assert len(results) == 100

        p.shutdown()


class TestRetention:
    """Test turn retention / max_turns_per_session enforcement."""

    def test_retention_enforced(self, temp_home):
        config = {"max_turns_per_session": 10, "auto_extract": True}
        p = ConversationMemoryProvider(config)
        p.initialize(session_id="retention-test", hermes_home=str(temp_home))

        # Write 20 turns (10 user+assistant pairs = 20 turns)
        for i in range(20):
            p.sync_turn(f"User {i}", f"Assistant {i}", session_id="retention-test")

        p._db.flush(timeout=2.0)
        results = p._db.recent("retention-test", limit=20)
        # Should only keep the last 10 turns (max_turns_per_session)
        assert len(results) == 10
        # Should be the most recent ones
        assert results[0].content == "User 10"
        assert results[-1].content == "Assistant 19"

        p.shutdown()

    def test_retention_zero_unlimited(self, temp_home):
        config = {"max_turns_per_session": 0, "auto_extract": True}
        p = ConversationMemoryProvider(config)
        p.initialize(session_id="unlimited-test", hermes_home=str(temp_home))

        for i in range(50):
            p.sync_turn(f"User {i}", f"Assistant {i}", session_id="unlimited-test")

        p._db.flush(timeout=2.0)
        results = p._db.recent("unlimited-test", limit=100)
        assert len(results) == 100  # All 50 pairs = 100 turns

        p.shutdown()


class TestFTS5Availability:
    """Test behavior when FTS5 is not available."""

    def test_fts_disabled_in_config(self, temp_home):
        config = {"fts_enabled": False, "auto_extract": True}
        p = ConversationMemoryProvider(config)
        p.initialize(session_id="no-fts", hermes_home=str(temp_home))

        p.sync_turn("Searchable content", "Response", session_id="no-fts")
        p._db.flush(timeout=2.0)

        # Search should return empty when FTS disabled
        results = p._db.search("Searchable", session_id="no-fts", limit=5)
        assert results == []

        p.shutdown()


class TestAutoExtractToggle:
    """Test auto_extract config option."""

    def test_auto_extract_false_disables_sync(self, temp_home):
        config = {"auto_extract": False}
        p = ConversationMemoryProvider(config)
        p.initialize(session_id="no-auto", hermes_home=str(temp_home))

        p.sync_turn("Should not persist", "Response", session_id="no-auto")
        p._db.flush(timeout=2.0)

        results = p._db.recent("no-auto", limit=5)
        assert len(results) == 0

        p.shutdown()

    def test_auto_extract_false_disables_memory_mirror(self, temp_home):
        config = {"auto_extract": False}
        p = ConversationMemoryProvider(config)
        p.initialize(session_id="no-auto-mirror", hermes_home=str(temp_home))

        p.on_memory_write("add", "user", "Should not mirror")
        p._db.flush(timeout=2.0)

        results = p._db.recent("no-auto-mirror", limit=5)
        assert len(results) == 0

        p.shutdown()


class TestPluginRegistration:
    """Test the plugin registration entry point."""

    def test_register_creates_provider(self):
        from types import SimpleNamespace

        collector = SimpleNamespace(provider=None)

        def register_memory_provider(provider):
            collector.provider = provider

        ctx = SimpleNamespace(register_memory_provider=register_memory_provider)

        # Import and call register
        from plugins.memory.conversation_memory import register
        register(ctx)

        assert collector.provider is not None
        assert isinstance(collector.provider, ConversationMemoryProvider)
        assert collector.provider.name == "conversation_memory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])