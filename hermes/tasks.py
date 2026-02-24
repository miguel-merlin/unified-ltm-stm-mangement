"""
Defines the RL training distribution for AgeMem

Controls:
- Which facts appear in context
- Whether distractors are injected
- What constitutes task success
- Stage-specific curriculum behavior

Future integration:
- Replace synthetic generators with real benchmarks (ALFWorld, ScienceWorld)
- Add multi-turn temporal separation (fact ingestion phase -> delayed query)
- Add long-horizon tasks for stronger LTM pressure

+ Need to reporcess data so that it suits training with GRPO policy
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Tuple
from datasets import Dataset

TRACE_INSTRUCTIONS = """You are a memory-management agent.

You MUST output a TOOL TRACE as JSONL (one JSON object per line). No extra text.

Allowed STM actions:
  {"tool":"stm","action":"retain|discard|retrieve|summary|filter","content":"...","k":optional_int}

Allowed LTM actions:
  {"tool":"ltm","action":"add|retrieve|update|delete|get|list|clear",
   "content":"...", "key":optional_str, "k":optional_int,
   "tags":optional_list, "importance":optional_float, "meta":optional_object,
   "tags_any":optional_list, "meta_filter":optional_object}

Special unified action (recommended when useful):
  {"tool":"ltm","action":"retrieve_to_stm","content":"...","k":optional_int}

Finish with exactly one line:
  {"final":"...your answer..."}

Rules:
- Output ONLY JSON per line, no markdown.
- Keep trace <= 10 lines total.
- Use tools deliberately; avoid unnecessary writes.
"""

# Import before training 
DISTRACTORS = []
NAMES = []
PREFS = []
CITIES = []

def _build_prompt(stage: str, context_lines: List[str], user_query: str) -> str:
    """
    Build a single prompt that encourages the right behavior per stage.
    We keep the same JSONL tool schema across stages, only the goal changes.
    """
    stage_hint = {
        "stage1_ltm": (
            "Goal: Learn Long-Term Memory (LTM).\n"
            "- Store important facts in LTM.\n"
            "- You may answer by retrieving from LTM.\n"
            "- Avoid storing irrelevant information."
        ),
        "stage2_stm_noise": (
            "Goal: Learn Short-Term Memory (STM) management under distractors.\n"
            "- Filter/summary/discard distractors.\n"
            "- Keep only task-relevant info in STM.\n"
            "- Use LTM minimally."
        ),
        "stage3_unified": (
            "Goal: Unified STM + LTM.\n"
            "- Store selectively in LTM.\n"
            "- Handle distractors in STM.\n"
            "- Retrieve from LTM into STM (retrieve_to_stm) when needed."
        ),
    }.get(stage, "Goal: Manage memory well.")

    ctx = "\n".join([f"- {x}" for x in context_lines]) if context_lines else "- (empty)"

    return (
        f"{TRACE_INSTRUCTIONS}\n\n"
        f"STAGE: {stage}\n"
        f"{stage_hint}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"USER_QUERY: {user_query}\n"
    )

def _sample_fact_episode(r: random.Random) -> Tuple[List[str], List[str], str]:
    name = r.choice(NAMES)
    pref = r.choice(PREFS)
    city = r.choice(CITIES)

    # Facts appear in the context:
    context = [f"My name is {name}.", f"I prefer {pref}.", f"I live in {city}."]

    # The final answer must include:
    required = [pref.lower(), city.lower()]
    user_query = "What drink do I prefer and what city do I live in?"
    return context, required, user_query

def _inject_distractors(r: random.Random, context: List[str], min_k: int = 3, max_k: int = 5) -> List[str]:
    ctx = list(context)
    k = r.randint(min_k, max_k)
    for i in range(k):
        ctx.insert(r.randint(0, len(ctx)), r.choice(DISTRACTORS))
    return ctx

def build_examples(stage: str, n: int, seed: int = 0) -> List[Dict[str, Any]]:
    r = random.Random(seed)
    examples: List[Dict[str, Any]] = []

    for i in range(n):
        context, required, user_query = _sample_fact_episode(r)

        # Stage-specific context shaping:
        if stage in ("stage2_stm_noise", "stage3_unified"):
            context = _inject_distractors(r, context)

        prompt = _build_prompt(stage=stage, context_lines=context, user_query=user_query)

        examples.append(
            {
                "prompt": prompt,
                "required": required,
                "stage": stage,
                "task_id": f"{stage}_t{i}",
            }
        )

    return examples

def build_dataset(stage: str, n: int, seed: int = 0) -> Dataset:
    return Dataset.from_list(build_examples(stage=stage, n=n, seed=seed))
