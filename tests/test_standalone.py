#!/usr/bin/env python3
"""Standalone test runner for ConversationMemoryProvider (no pytest needed)."""

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

# Add the Hermes agent and user plugins to path
sys.path.insert(0, "/root/.hermes/hermes-agent")
sys.path.insert(0, "/root/.hermes/plugins")

from conversation_memory import ConversationMemoryProvider


def run_test(name, fn):
    try:
        fn()
        print(f"✓ {name}")
        return True
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False


def test_provider_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test", hermes_home=tmpdir)
        assert p.name == "conversation_memory"
        p.shutdown()


def test_is_available():
    p = ConversationMemoryProvider()
    assert p.is_available() is True


def test_initialize_creates_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test", hermes_home=tmpdir)
        db_path = Path(tmpdir) / "conversation_memory" / "conversation_memory.db"
        assert db_path.exists()
        p.shutdown()


def test_system_prompt_block():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test", hermes_home=tmpdir)
        block = p.system_prompt_block()
        assert "Conversation Memory" in block
        assert "conversation_memory_search" in block
        p.shutdown()


def test_sync_turn_persists_turns():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.sync_turn("Hello", "Hi there!", session_id="test-session")
        p._db.flush(timeout=2.0)

        results = p._db.recent("test-session", limit=10)
        assert len(results) == 2
        assert results[0].role == "user"
        assert results[0].content == "Hello"
        assert results[1].role == "assistant"
        assert results[1].content == "Hi there!"
        p.shutdown()


def test_sync_turn_increments_turn_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        assert p._turn_count == 0
        p.sync_turn("User 1", "Assistant 1", session_id="test-session")
        assert p._turn_count == 2
        p.sync_turn("User 2", "Assistant 2", session_id="test-session")
        assert p._turn_count == 4
        p.shutdown()


def test_search_returns_ranked_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.sync_turn("I like apples", "Apples are great", session_id="test-session")
        p.sync_turn("Bananas are yellow", "Yes, bananas are yellow fruits", session_id="test-session")
        p.sync_turn("Apple pie recipe", "Here is an apple pie recipe", session_id="test-session")
        p._db.flush(timeout=2.0)

        results = p._db.search("apple", session_id="test-session", limit=5)
        assert len(results) >= 2
        contents = [r.content.lower() for r in results]
        assert any("apple" in c for c in contents)
        p.shutdown()


def test_search_respects_session_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="session-a", hermes_home=tmpdir)
        p.sync_turn("Session A content", "Response A", session_id="session-a")
        p.sync_turn("Session B content", "Response B", session_id="session-b")
        p._db.flush(timeout=2.0)

        results_a = p._db.search("content", session_id="session-a", limit=5)
        results_b = p._db.search("content", session_id="session-b", limit=5)

        assert all(r.session_id == "session-a" for r in results_a)
        assert all(r.session_id == "session-b" for r in results_b)
        p.shutdown()


def test_recent_returns_recent_turns():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.sync_turn("First", "Response 1", session_id="test-session")
        p.sync_turn("Second", "Response 2", session_id="test-session")
        p.sync_turn("Third", "Response 3", session_id="test-session")
        p._db.flush(timeout=2.0)

        results = p._db.recent("test-session", limit=2)
        assert len(results) == 2
        assert results[0].content == "Second"
        assert results[1].content == "Third"
        p.shutdown()


def test_on_memory_write_mirrors_builtin():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.on_memory_write("add", "user", "User prefers dark mode")
        p.on_memory_write("add", "memory", "Agent should be concise")
        p._db.flush(timeout=2.0)

        results = p._db.recent("test-session", limit=10)
        tool_turns = [r for r in results if r.role == "tool"]
        assert len(tool_turns) >= 2
        contents = [t.content for t in tool_turns]
        assert any("User prefers dark mode" in c for c in contents)
        assert any("Agent should be concise" in c for c in contents)
        p.shutdown()


def test_on_pre_compress_skips_compressed_messages():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2", "_compressed_summary": True},
            {"role": "assistant", "content": "Response 2"},
        ]
        p.on_pre_compress(messages)
        p._db.flush(timeout=2.0)

        results = p._db.recent("test-session", limit=10)
        tool_turns = [r for r in results if r.role == "tool" and "Pre-compression" in r.content]
        assert len(tool_turns) == 1
        content = tool_turns[0].content
        assert "Message 1" in content
        assert "Response 1" in content
        assert "Response 2" in content
        assert "Message 2" not in content
        p.shutdown()


def test_on_pre_compress_empty_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        result = p.on_pre_compress([])
        assert result == ""

        result = p.on_pre_compress([{"role": "user", "content": "test", "_compressed_summary": True}])
        assert result == ""
        p.shutdown()


def test_on_session_end_flushes():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.sync_turn("Test", "Response", session_id="test-session")
        p.on_session_end([])  # Should not block
        p.shutdown()


def test_on_session_switch_rebinds():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.sync_turn("Old session", "Response", session_id="test-session")
        p._db.flush(timeout=2.0)

        p.on_session_switch("new-session")
        assert p._session_id == "new-session"
        assert p._turn_count == 0

        results = p._db.recent("new-session", limit=5)
        assert len(results) == 0
        p.shutdown()


def test_on_session_switch_resumes_turn_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.sync_turn("Turn 1", "Response 1", session_id="test-session")
        p.sync_turn("Turn 2", "Response 2", session_id="test-session")
        p._db.flush(timeout=2.0)

        p.on_session_switch("test-session")
        assert p._turn_count == 4
        p.shutdown()


def test_identity_signature():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test", hermes_home=tmpdir)
        sig = p.identity_signature()
        assert sig["provider"] == "conversation_memory"
        assert "db_path" in sig
        p.shutdown()


def test_get_tool_schemas():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test", hermes_home=tmpdir)
        schemas = p.get_tool_schemas()
        assert len(schemas) == 5
        names = {s["name"] for s in schemas}
        assert names == {
            "conversation_memory_search",
            "conversation_memory_recent",
            "conversation_memory_save",
            "conversation_memory_status",
            "conversation_memory_clear",
        }
        p.shutdown()


def test_handle_tool_call_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.sync_turn("I love Python", "Python is great", session_id="test-session")
        p._db.flush(timeout=2.0)

        result = json.loads(p.handle_tool_call("conversation_memory_search", {"query": "Python"}))
        assert "results" in result
        assert len(result["results"]) >= 1
        p.shutdown()


def test_handle_tool_call_recent():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.sync_turn("Recent test", "Response", session_id="test-session")
        p._db.flush(timeout=2.0)

        result = json.loads(p.handle_tool_call("conversation_memory_recent", {"session_id": "test-session"}))
        assert "turns" in result
        assert len(result["turns"]) >= 1
        p.shutdown()


def test_handle_tool_call_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        result = json.loads(p.handle_tool_call("conversation_memory_save", {
            "content": "Important fact: user likes cats",
            "tags": ["preference", "pets"],
            "importance": 0.9
        }))
        assert result["result"] == "Saved to conversation memory"

        p._db.flush(timeout=2.0)
        results = p._db.recent("test-session", limit=5)
        tool_turns = [r for r in results if r.role == "tool"]
        assert any("Important fact" in t.content for t in tool_turns)
        p.shutdown()


def test_handle_tool_call_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        result = json.loads(p.handle_tool_call("conversation_memory_status", {}))
        assert "sessions" in result
        assert "turns" in result
        assert "db_size_bytes" in result
        assert "fts_enabled" in result
        p.shutdown()


def test_handle_tool_call_clear_requires_confirm():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        result = json.loads(p.handle_tool_call("conversation_memory_clear", {"session_id": "test-session"}))
        assert "error" in result
        assert "confirm=true" in result["error"]
        p.shutdown()


def test_handle_tool_call_clear_with_confirm():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.sync_turn("To be cleared", "Response", session_id="test-session")
        p._db.flush(timeout=2.0)

        result = json.loads(p.handle_tool_call("conversation_memory_clear", {
            "session_id": "test-session",
            "confirm": True
        }))
        assert "Cleared session" in result["result"]
        p.shutdown()


def test_shutdown_cleans_up():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.sync_turn("Shutdown test", "Response", session_id="test-session")
        p.shutdown()
        p.shutdown()  # Double shutdown should not raise


def test_prefetch_trivial_prompt_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test", hermes_home=tmpdir)
        assert p.prefetch("hi") == ""
        assert p.prefetch("ok") == ""
        assert p.prefetch("/help") == ""
        p.shutdown()


def test_queue_prefetch_populates_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        p.sync_turn("Searchable content", "Response", session_id="test-session")
        p._db.flush(timeout=2.0)

        p.queue_prefetch("Searchable", session_id="test-session")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=2.0)

        result = p.prefetch("Searchable", session_id="test-session")
        assert "Searchable content" in result
        p.shutdown()


def test_recall_status_reflects_last_prefetch():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=tmpdir)
        assert p.recall_status() is None

        p.sync_turn("Test content", "Response", session_id="test-session")
        p._db.flush(timeout=2.0)

        p.queue_prefetch("Test", session_id="test-session")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=2.0)

        p.prefetch("Test", session_id="test-session")
        status = p.recall_status()
        assert status is not None
        assert status.provider_label == "conversation_memory"
        assert status.count == 1
        p.shutdown()


def test_config_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider()
        p.initialize(session_id="test", hermes_home=tmpdir)
        schema = p.get_config_schema()
        assert len(schema) == 4
        keys = {s["key"] for s in schema}
        assert keys == {"max_context_chars", "max_turns_per_session", "fts_enabled", "auto_extract"}
        p.shutdown()


def test_multi_profile_isolation():
    with tempfile.TemporaryDirectory() as tmpdir:
        home1 = Path(tmpdir) / "profile1"
        home2 = Path(tmpdir) / "profile2"
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

        results1 = p1._db.recent("session1", limit=5)
        results2 = p2._db.recent("session2", limit=5)

        assert any("Profile 1 data" in r.content for r in results1)
        assert not any("Profile 2 data" in r.content for r in results1)
        assert any("Profile 2 data" in r.content for r in results2)
        assert not any("Profile 1 data" in r.content for r in results2)

        p1.shutdown()
        p2.shutdown()


def test_concurrent_access_thread_safety():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = ConversationMemoryProvider({"auto_extract": True, "max_turns_per_session": 1000})
        p.initialize(session_id="concurrent-test", hermes_home=str(tmpdir))

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
        assert len(results) == 100

        p.shutdown()


def test_retention_enforced():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"max_turns_per_session": 10, "auto_extract": True}
        p = ConversationMemoryProvider(config)
        p.initialize(session_id="retention-test", hermes_home=str(tmpdir))

        for i in range(20):
            p.sync_turn(f"User {i}", f"Assistant {i}", session_id="retention-test")

        p._db.flush(timeout=2.0)
        results = p._db.recent("retention-test", limit=20)
        assert len(results) == 10
        assert results[0].content == "User 10"
        assert results[-1].content == "Assistant 19"

        p.shutdown()


def test_retention_zero_unlimited():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"max_turns_per_session": 0, "auto_extract": True}
        p = ConversationMemoryProvider(config)
        p.initialize(session_id="unlimited-test", hermes_home=str(tmpdir))

        for i in range(50):
            p.sync_turn(f"User {i}", f"Assistant {i}", session_id="unlimited-test")

        p._db.flush(timeout=2.0)
        results = p._db.recent("unlimited-test", limit=100)
        assert len(results) == 100

        p.shutdown()


def test_fts_disabled_in_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"fts_enabled": False, "auto_extract": True}
        p = ConversationMemoryProvider(config)
        p.initialize(session_id="no-fts", hermes_home=str(tmpdir))

        p.sync_turn("Searchable content", "Response", session_id="no-fts")
        p._db.flush(timeout=2.0)

        results = p._db.search("Searchable", session_id="no-fts", limit=5)
        assert results == []

        p.shutdown()


def test_auto_extract_false_disables_sync():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"auto_extract": False}
        p = ConversationMemoryProvider(config)
        p.initialize(session_id="no-auto", hermes_home=str(tmpdir))

        p.sync_turn("Should not persist", "Response", session_id="no-auto")
        p._db.flush(timeout=2.0)

        results = p._db.recent("no-auto", limit=5)
        assert len(results) == 0

        p.shutdown()


def test_auto_extract_false_disables_memory_mirror():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"auto_extract": False}
        p = ConversationMemoryProvider(config)
        p.initialize(session_id="no-auto-mirror", hermes_home=str(tmpdir))

        p.on_memory_write("add", "user", "Should not mirror")
        p._db.flush(timeout=2.0)

        results = p._db.recent("no-auto-mirror", limit=5)
        assert len(results) == 0

        p.shutdown()


def test_plugin_registration():
    from types import SimpleNamespace

    collector = SimpleNamespace(provider=None)

    def register_memory_provider(provider):
        collector.provider = provider

    ctx = SimpleNamespace(register_memory_provider=register_memory_provider)

    from conversation_memory import register
    register(ctx)

    assert collector.provider is not None
    assert isinstance(collector.provider, ConversationMemoryProvider)
    assert collector.provider.name == "conversation_memory"


if __name__ == "__main__":
    tests = [
        ("provider_name", test_provider_name),
        ("is_available", test_is_available),
        ("initialize_creates_db", test_initialize_creates_db),
        ("system_prompt_block", test_system_prompt_block),
        ("sync_turn_persists_turns", test_sync_turn_persists_turns),
        ("sync_turn_increments_turn_count", test_sync_turn_increments_turn_count),
        ("search_returns_ranked_results", test_search_returns_ranked_results),
        ("search_respects_session_filter", test_search_respects_session_filter),
        ("recent_returns_recent_turns", test_recent_returns_recent_turns),
        ("on_memory_write_mirrors_builtin", test_on_memory_write_mirrors_builtin),
        ("on_pre_compress_skips_compressed_messages", test_on_pre_compress_skips_compressed_messages),
        ("on_pre_compress_empty_returns_empty", test_on_pre_compress_empty_returns_empty),
        ("on_session_end_flushes", test_on_session_end_flushes),
        ("on_session_switch_rebinds", test_on_session_switch_rebinds),
        ("on_session_switch_resumes_turn_count", test_on_session_switch_resumes_turn_count),
        ("identity_signature", test_identity_signature),
        ("get_tool_schemas", test_get_tool_schemas),
        ("handle_tool_call_search", test_handle_tool_call_search),
        ("handle_tool_call_recent", test_handle_tool_call_recent),
        ("handle_tool_call_save", test_handle_tool_call_save),
        ("handle_tool_call_status", test_handle_tool_call_status),
        ("handle_tool_call_clear_requires_confirm", test_handle_tool_call_clear_requires_confirm),
        ("handle_tool_call_clear_with_confirm", test_handle_tool_call_clear_with_confirm),
        ("shutdown_cleans_up", test_shutdown_cleans_up),
        ("prefetch_trivial_prompt_returns_empty", test_prefetch_trivial_prompt_returns_empty),
        ("queue_prefetch_populates_cache", test_queue_prefetch_populates_cache),
        ("recall_status_reflects_last_prefetch", test_recall_status_reflects_last_prefetch),
        ("config_schema", test_config_schema),
        ("multi_profile_isolation", test_multi_profile_isolation),
        ("concurrent_access_thread_safety", test_concurrent_access_thread_safety),
        ("retention_enforced", test_retention_enforced),
        ("retention_zero_unlimited", test_retention_zero_unlimited),
        ("fts_disabled_in_config", test_fts_disabled_in_config),
        ("auto_extract_false_disables_sync", test_auto_extract_false_disables_sync),
        ("auto_extract_false_disables_memory_mirror", test_auto_extract_false_disables_memory_mirror),
        ("plugin_registration", test_plugin_registration),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        if run_test(name, fn):
            passed += 1
        else:
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)