"""Unit tests for LongTermMemory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes.memory import LongTermMemory


@pytest.fixture
def ltm() -> LongTermMemory:
    return LongTermMemory()


def test_add_creates_record(ltm: LongTermMemory) -> None:
    result = ltm.add("User likes jazz", tags=["music"], importance=0.8)
    assert result["ok"] is True
    assert result["op"] == "add"
    key = result["key"]
    assert key.startswith("ltm_")
    assert key in ltm._store


def test_add_multiple_increments_counter(ltm: LongTermMemory) -> None:
    ltm.add("fact A")
    ltm.add("fact B")
    ltm.add("fact C")
    assert len(ltm._store) == 3
    # Keys should be sequential
    assert "ltm_1" in ltm._store
    assert "ltm_3" in ltm._store


def test_retrieve_returns_relevant_hits(ltm: LongTermMemory) -> None:
    ltm.add("User prefers tea in the morning")
    ltm.add("Meeting scheduled for Tuesday")
    ltm.add("User lives in New York")

    result = ltm.retrieve("tea preferences", k=2)
    assert result["ok"] is True
    assert result["op"] == "retrieve"
    hits = result["hits"]
    # The tea entry should be the top hit
    assert len(hits) >= 1
    assert "tea" in hits[0]["content"].lower()


def test_retrieve_respects_k_limit(ltm: LongTermMemory) -> None:
    for i in range(10):
        ltm.add(f"fact about memory item {i}")
    result = ltm.retrieve("memory fact", k=3)
    assert len(result["hits"]) <= 3


def test_retrieve_updates_access_count(ltm: LongTermMemory) -> None:
    key = ltm.add("tasty coffee")["key"]
    assert ltm._store[key]["access_count"] == 0
    ltm.retrieve("coffee")
    assert ltm._store[key]["access_count"] == 1


def test_retrieve_tag_filter(ltm: LongTermMemory) -> None:
    ltm.add("I love Python", tags=["programming"])
    ltm.add("I love cats", tags=["animals"])
    result = ltm.retrieve("love", tags_any=["programming"])
    hits = result["hits"]
    assert all("programming" in h["tags"] for h in hits)


def test_retrieve_meta_filter(ltm: LongTermMemory) -> None:
    ltm.add("Work task", meta={"type": "work"})
    ltm.add("Personal task", meta={"type": "personal"})
    result = ltm.retrieve("task", meta_filter={"type": "work"})
    hits = result["hits"]
    assert all(h["meta"].get("type") == "work" for h in hits)


def test_update_modifies_content(ltm: LongTermMemory) -> None:
    key = ltm.add("Original content")["key"]
    result = ltm.update(key=key, content="Updated content")
    assert result["ok"] is True
    assert ltm._store[key]["content"] == "Updated content"


def test_update_missing_key(ltm: LongTermMemory) -> None:
    result = ltm.update(key="ltm_999", content="x")
    assert result["ok"] is False
    assert "key not found" in result["error"]


def test_update_partial_fields(ltm: LongTermMemory) -> None:
    key = ltm.add("fact", tags=["a"], importance=0.1)["key"]
    ltm.update(key=key, importance=0.9)
    assert ltm._store[key]["importance"] == 0.9
    assert ltm._store[key]["tags"] == ["a"]  # unchanged


def test_delete_removes_record(ltm: LongTermMemory) -> None:
    key = ltm.add("temporary fact")["key"]
    result = ltm.delete(key=key)
    assert result["ok"] is True
    assert key not in ltm._store


def test_delete_missing_key(ltm: LongTermMemory) -> None:
    result = ltm.delete(key="ltm_999")
    assert result["ok"] is False


def test_get_returns_record(ltm: LongTermMemory) -> None:
    key = ltm.add("full record", tags=["t1"], importance=0.5)["key"]
    result = ltm.get(key=key)
    assert result["ok"] is True
    rec = result["record"]
    assert rec["content"] == "full record"
    assert rec["tags"] == ["t1"]
    assert rec["importance"] == 0.5


def test_get_missing_key(ltm: LongTermMemory) -> None:
    result = ltm.get(key="ltm_nonexistent")
    assert result["ok"] is False


def test_list_returns_all_keys(ltm: LongTermMemory) -> None:
    ltm.add("item one")
    ltm.add("item two")
    result = ltm.list()
    assert result["ok"] is True
    keys = [it["key"] for it in result["items"]]
    assert "ltm_1" in keys
    assert "ltm_2" in keys


def test_list_preview_truncated(ltm: LongTermMemory) -> None:
    ltm.add("A" * 200)
    result = ltm.list()
    preview = result["items"][0]["preview"]
    assert len(preview) <= 80


def test_clear_empties_store(ltm: LongTermMemory) -> None:
    ltm.add("a")
    ltm.add("b")
    result = ltm.clear()
    assert result["ok"] is True
    assert len(ltm._store) == 0
    assert ltm._counter == 0


def test_snapshot_returns_copy(ltm: LongTermMemory) -> None:
    key = ltm.add("snapshot test")["key"]
    snap = ltm.snapshot()
    snap[key]["content"] = "MUTATED"
    # Original should be unchanged
    assert ltm._store[key]["content"] == "snapshot test"
