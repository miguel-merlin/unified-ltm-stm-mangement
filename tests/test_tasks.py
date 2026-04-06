"""Unit tests for hermes/tasks.py — synthetic dataset generation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes.tasks import (
    NAMES,
    PREFS,
    CITIES,
    DISTRACTORS,
    build_examples,
    build_dataset,
)


# ---------------------------------------------------------------------------
# Vocabulary sanity checks
# ---------------------------------------------------------------------------

class TestVocabularies:
    def test_names_is_non_empty(self):
        assert len(NAMES) >= 10

    def test_prefs_is_non_empty(self):
        assert len(PREFS) >= 5

    def test_cities_is_non_empty(self):
        assert len(CITIES) >= 5

    def test_distractors_is_non_empty(self):
        assert len(DISTRACTORS) >= 5

    def test_distractors_contain_prefix(self):
        """All distractors should start with 'Distractor:' for reward parsing."""
        for d in DISTRACTORS:
            assert d.lower().startswith("distractor:"), f"Bad distractor: {d!r}"


# ---------------------------------------------------------------------------
# build_examples tests
# ---------------------------------------------------------------------------

class TestBuildExamples:
    def test_returns_correct_count(self):
        examples = build_examples("stage1_ltm", n=10, seed=0)
        assert len(examples) == 10

    def test_example_has_required_keys(self):
        examples = build_examples("stage1_ltm", n=5, seed=0)
        for ex in examples:
            assert "prompt" in ex
            assert "required" in ex
            assert "stage" in ex
            assert "task_id" in ex

    def test_stage_set_correctly(self):
        for stage in ("stage1_ltm", "stage2_stm_noise", "stage3_unified"):
            examples = build_examples(stage, n=3, seed=0)
            for ex in examples:
                assert ex["stage"] == stage

    def test_prompt_contains_trace_instructions(self):
        examples = build_examples("stage1_ltm", n=1, seed=0)
        assert "TOOL TRACE" in examples[0]["prompt"]

    def test_required_is_list_of_strings(self):
        examples = build_examples("stage1_ltm", n=20, seed=42)
        for ex in examples:
            assert isinstance(ex["required"], list)
            assert all(isinstance(r, str) for r in ex["required"])

    def test_required_tokens_appear_in_city_or_pref(self):
        """Required tokens should be from PREFS or CITIES vocabulary."""
        examples = build_examples("stage1_ltm", n=50, seed=7)
        prefs_lower = [p.lower() for p in PREFS]
        cities_lower = [c.lower() for c in CITIES]
        all_vocab = prefs_lower + cities_lower
        for ex in examples:
            for tok in ex["required"]:
                # Each required token should match a vocabulary item
                assert any(tok in v or v in tok for v in all_vocab), (
                    f"Required token '{tok}' not found in vocab"
                )

    def test_stage2_has_distractors_in_prompt(self):
        examples = build_examples("stage2_stm_noise", n=10, seed=0)
        for ex in examples:
            assert "Distractor:" in ex["prompt"]

    def test_stage3_has_distractors_in_prompt(self):
        examples = build_examples("stage3_unified", n=10, seed=0)
        for ex in examples:
            assert "Distractor:" in ex["prompt"]

    def test_stage1_no_distractors_in_prompt(self):
        examples = build_examples("stage1_ltm", n=10, seed=0)
        for ex in examples:
            assert "Distractor:" not in ex["prompt"]

    def test_deterministic_with_same_seed(self):
        examples_a = build_examples("stage3_unified", n=5, seed=99)
        examples_b = build_examples("stage3_unified", n=5, seed=99)
        assert examples_a[0]["prompt"] == examples_b[0]["prompt"]

    def test_different_seeds_produce_different_examples(self):
        examples_a = build_examples("stage1_ltm", n=5, seed=1)
        examples_b = build_examples("stage1_ltm", n=5, seed=2)
        # With different seeds, at least one example should differ
        assert any(a["prompt"] != b["prompt"] for a, b in zip(examples_a, examples_b))


# ---------------------------------------------------------------------------
# build_dataset tests
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def test_returns_dataset_object(self):
        from datasets import Dataset
        ds = build_dataset("stage1_ltm", n=5, seed=0)
        assert isinstance(ds, Dataset)

    def test_dataset_has_expected_columns(self):
        ds = build_dataset("stage1_ltm", n=5, seed=0)
        assert "prompt" in ds.column_names
        assert "required" in ds.column_names
        assert "stage" in ds.column_names

    def test_dataset_length(self):
        ds = build_dataset("stage2_stm_noise", n=12, seed=0)
        assert len(ds) == 12
