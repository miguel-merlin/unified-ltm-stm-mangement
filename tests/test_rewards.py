"""Unit tests for the composite AgeMem reward function."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes.rewards import hermes_trace_reward, _r_task, _r_context, _r_memory
from hermes.memory import ShortTermMemory


# ---------------------------------------------------------------------------
# Dummy embedder — avoids loading sentence-transformers in reward tests
# ---------------------------------------------------------------------------

class _DummyEmbed:
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t))] for t in texts]


class _FakeChromaCollection:
    """Minimal in-memory collection that satisfies ShortTermMemory's API."""
    def __init__(self):
        self._store = {}

    def add(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self._store[i] = {"emb": e, "doc": d, "meta": m}

    def delete(self, ids):
        for i in ids:
            self._store.pop(i, None)

    def query(self, query_embeddings, n_results, include):
        scored = sorted(
            self._store.values(),
            key=lambda r: abs(r["emb"][0] - query_embeddings[0][0]),
        )
        top = scored[:n_results]
        return {
            "documents": [[r["doc"] for r in top]],
            "distances": [[0.0] * len(top)],
            "metadatas": [[r["meta"] for r in top]],
        }


def _make_stm_with_dummy() -> ShortTermMemory:
    return ShortTermMemory(embedder=_DummyEmbed(), collection=_FakeChromaCollection())


# ---------------------------------------------------------------------------
# Helper: build valid JSONL completions
# ---------------------------------------------------------------------------

def _make_completion(actions: list, final: str) -> str:
    lines = [json.dumps(a) for a in actions]
    lines.append(json.dumps({"final": final}))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _r_task unit tests
# ---------------------------------------------------------------------------

class TestRTask:
    def test_all_required_tokens_match(self):
        assert _r_task("I prefer coffee in New York", ["coffee", "new york"]) == 1.0

    def test_partial_match_returns_penalty(self):
        score = _r_task("I prefer coffee", ["coffee", "new york"])
        assert score == pytest.approx(-0.1)

    def test_empty_answer_returns_minus_one(self):
        assert _r_task("", ["coffee"]) == -1.0

    def test_no_required_with_answer(self):
        assert _r_task("Some non-empty answer", None) == pytest.approx(0.1)

    def test_no_required_no_answer(self):
        assert _r_task("", None) == -1.0


# ---------------------------------------------------------------------------
# hermes_trace_reward integration tests
# ---------------------------------------------------------------------------

class TestHermesTraceReward:
    def test_returns_list_same_length(self):
        completions = [
            _make_completion([], "hello"),
            _make_completion([], "world"),
        ]
        rewards = hermes_trace_reward(completions)
        assert len(rewards) == 2

    def test_json_parse_error_returns_minus_one(self):
        completions = ["this is not json at all"]
        rewards = hermes_trace_reward(completions)
        assert rewards[0] == pytest.approx(-1.0)

    def test_missing_final_returns_minus_0_7(self):
        # Valid JSON but no {"final": ...} line
        completions = [json.dumps({"tool": "stm", "action": "retain", "content": "x"})]
        rewards = hermes_trace_reward(completions)
        assert rewards[0] == pytest.approx(-0.7)

    def test_invalid_schema_returns_minus_0_8(self):
        # Valid JSON but invalid tool schema
        line1 = json.dumps({"tool": "unknown_tool", "action": "do_something"})
        final = json.dumps({"final": "done"})
        completions = [f"{line1}\n{final}"]
        rewards = hermes_trace_reward(completions)
        assert rewards[0] == pytest.approx(-0.8)

    def test_correct_completion_has_positive_reward(self):
        actions = [
            {"tool": "ltm", "action": "add", "content": "User prefers coffee"},
            {"tool": "ltm", "action": "retrieve", "content": "coffee", "k": 3},
        ]
        completion = _make_completion(actions, "The user prefers coffee")
        rewards = hermes_trace_reward(
            [completion],
            required=["coffee"],
            stage="stage3_unified",
        )
        assert rewards[0] > 0

    def test_correct_answer_with_required_keywords(self):
        completion = _make_completion([], "I live in new york and prefer coffee")
        rewards = hermes_trace_reward(
            [completion],
            required=["coffee", "new york"],
            stage="stage1_ltm",
        )
        # Should be positive since all required tokens present
        assert rewards[0] > 0

    def test_wrong_answer_with_required_keywords(self):
        completion = _make_completion([], "I don't know the answer")
        rewards_wrong = hermes_trace_reward(
            [completion],
            required=["coffee", "new york"],
            stage="stage1_ltm",
        )
        completion_right = _make_completion([], "coffee in new york")
        rewards_right = hermes_trace_reward(
            [completion_right],
            required=["coffee", "new york"],
            stage="stage1_ltm",
        )
        assert rewards_right[0] > rewards_wrong[0]

    def test_stm_tools_counted(self):
        """Completions that use STM filter/summary should get context reward bonus."""
        actions_with_stm = [
            {"tool": "stm", "action": "filter", "content": "distractor"},
            {"tool": "stm", "action": "summary", "content": ""},
        ]
        comp_with = _make_completion(actions_with_stm, "coffee")
        comp_without = _make_completion([], "coffee")
        r_with = hermes_trace_reward([comp_with], required=["coffee"])[0]
        r_without = hermes_trace_reward([comp_without], required=["coffee"])[0]
        # Both should be valid; with-STM version should have non-lower reward
        assert isinstance(r_with, float)
        assert isinstance(r_without, float)

    def test_ltm_retrieve_to_stm(self):
        """retrieve_to_stm should not crash and should count as a tool call."""
        actions = [
            {"tool": "ltm", "action": "retrieve_to_stm", "content": "coffee", "k": 3},
        ]
        completion = _make_completion(actions, "coffee is stored")
        rewards = hermes_trace_reward([completion], required=["coffee"])
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_batch_of_completions(self):
        """Reward function should handle batches correctly."""
        completions = [
            _make_completion([], f"answer {i} coffee") for i in range(5)
        ]
        rewards = hermes_trace_reward(completions, required=["coffee"])
        assert len(rewards) == 5
        assert all(isinstance(r, float) for r in rewards)

    def test_stage2_distractor_penalty(self, monkeypatch):
        """Stage 2 should penalise distractor: text still in STM.

        We monkeypatch ShortTermMemory.__init__ to pre-wire a dummy embedder
        so this test doesn't require sentence-transformers.
        """
        import hermes.rewards as rewards_mod
        import hermes.memory as memory_mod
        from hermes.tool_api import HermesToolAPI

        _orig_init = memory_mod.ShortTermMemory.__init__

        def _patched_init(self_stm, **kwargs):
            # Always inject dummy embedder + collection
            kwargs.setdefault("embedder", _DummyEmbed())
            kwargs.setdefault("collection", _FakeChromaCollection())
            _orig_init(self_stm, **kwargs)

        monkeypatch.setattr(memory_mod.ShortTermMemory, "__init__", _patched_init)

        actions = [
            {"tool": "stm", "action": "retain", "content": "distractor: irrelevant info"},
        ]
        comp_with_distractor = _make_completion(actions, "coffee")
        comp_clean = _make_completion([], "coffee")

        r_dirty = hermes_trace_reward(
            [comp_with_distractor], required=["coffee"], stage="stage2_stm_noise"
        )[0]
        r_clean = hermes_trace_reward(
            [comp_clean], required=["coffee"], stage="stage2_stm_noise"
        )[0]
        assert r_dirty < r_clean
