"""HotpotQA dataset loader for AgeMem training.

Converts HotpotQA QA pairs into 3-stage training examples:
  Stage 1: supporting_facts → context lines for LTM construction
  Stage 2: inject distractors into STM context
  Stage 3: question → query requiring LTM retrieval to answer

The paper (Section 4.1) uses HotpotQA as the sole RL training dataset,
then evaluates on all five benchmarks.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

from hermes.tasks import DISTRACTORS, TRACE_INSTRUCTIONS, _inject_distractors


# ---------------------------------------------------------------------------
# Stage-specific prompt builder for HotpotQA
# ---------------------------------------------------------------------------

_STAGE_HINTS: Dict[str, str] = {
    "stage1_ltm": (
        "Goal: Learn Long-Term Memory (LTM).\n"
        "- Read the supporting facts and store relevant ones in LTM.\n"
        "- Keep only high-quality, reusable information.\n"
        "- Do NOT answer the question yet."
    ),
    "stage2_stm_noise": (
        "Goal: Short-Term Memory (STM) management under distractors.\n"
        "- The context contains irrelevant material mixed with facts.\n"
        "- Filter/summarise distractors from STM.\n"
        "- Use LTM minimally."
    ),
    "stage3_unified": (
        "Goal: Unified STM + LTM.\n"
        "- Retrieve relevant facts from LTM.\n"
        "- Filter distractors in STM.\n"
        "- Answer the question using retrieved knowledge."
    ),
}


def _build_hotpotqa_prompt(
    stage: str,
    supporting_facts: List[str],
    question: str,
    distractors: Optional[List[str]] = None,
) -> str:
    """Build a HotpotQA-flavoured stage prompt."""
    stage_hint = _STAGE_HINTS.get(stage, "Goal: Manage memory and answer correctly.")

    ctx_lines = list(supporting_facts)
    if distractors:
        ctx_lines = distractors  # already interleaved

    ctx = "\n".join([f"- {x}" for x in ctx_lines]) if ctx_lines else "- (empty)"

    query_line = (
        f"USER_QUERY: {question}"
        if stage == "stage3_unified"
        else "USER_QUERY: (Store the above facts for later retrieval.)"
    )

    return (
        f"{TRACE_INSTRUCTIONS}\n\n"
        f"STAGE: {stage}\n"
        f"{stage_hint}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"{query_line}\n"
    )


# ---------------------------------------------------------------------------
# HuggingFace → GRPO example converter
# ---------------------------------------------------------------------------

def _hotpotqa_to_example(
    row: Dict[str, Any],
    stage: str,
    r: random.Random,
) -> Dict[str, Any]:
    """Convert a single HotpotQA row to an AgeMem training example."""
    question: str = row.get("question", "")
    answer: str = row.get("answer", "")
    # supporting_facts is a dict {"title": [...], "sent_id": [...]}
    # The actual sentences aren't directly in HotpotQA distractor splits,
    # so we use the context paragraphs that contain the supporting fact titles.
    sf_titles = set(row.get("supporting_facts", {}).get("title", []))
    context_titles = row.get("context", {}).get("title", [])
    context_sentences = row.get("context", {}).get("sentences", [])

    # Collect relevant supporting sentences
    supporting_lines: List[str] = []
    for title, sents in zip(context_titles, context_sentences):
        if title in sf_titles:
            for sent in sents:
                if sent.strip():
                    supporting_lines.append(f"[{title}] {sent.strip()}")
    if not supporting_lines:
        supporting_lines = [f"Question: {question}", f"Answer: {answer}"]

    # For distractor stages, interleave with random distractors
    if stage in ("stage2_stm_noise", "stage3_unified"):
        context = _inject_distractors(r, supporting_lines, min_k=2, max_k=4)
    else:
        context = supporting_lines

    # The 'required' list: answer words that should appear in final output
    required = [w.strip().lower() for w in answer.split() if len(w.strip()) > 2][:5]

    prompt = _build_hotpotqa_prompt(
        stage=stage,
        supporting_facts=context,
        question=question,
    )

    return {
        "prompt": prompt,
        "required": required,
        "stage": stage,
        "question": question,
        "answer": answer,
        "task_id": str(row.get("id", "")),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_hotpotqa_dataset(
    stage: str = "stage3_unified",
    split: str = "train",
    max_samples: int = 1000,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Load HotpotQA from HuggingFace and convert to AgeMem format.

    Args:
        stage:       One of stage1_ltm | stage2_stm_noise | stage3_unified
        split:       HuggingFace dataset split (train | validation)
        max_samples: Cap the number of examples (training can be slow)
        seed:        Random seed for distractor injection
        cache_dir:   Optional HuggingFace cache directory

    Returns:
        A datasets.Dataset with columns: prompt, required, stage, question, answer, task_id
    """
    print(f"[HotpotQA] Loading split='{split}', max_samples={max_samples}...")
    raw = load_dataset(
        "hotpot_qa",
        "distractor",
        split=split,
        cache_dir=cache_dir,
    )

    if max_samples and max_samples < len(raw):
        raw = raw.select(range(max_samples))

    r = random.Random(seed)
    examples = [_hotpotqa_to_example(row, stage=stage, r=r) for row in raw]
    print(f"[HotpotQA] Built {len(examples)} examples for stage '{stage}'.")
    return Dataset.from_list(examples)
