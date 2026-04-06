"""
Defines the RL training distribution for AgeMem.

Controls:
- Which facts appear in context
- Whether distractors are injected
- What constitutes task success
- Stage-specific curriculum behaviour

Three-stage episode structure (from Section 3.3):
  Stage 1 (LTM construction)  — agent sees casual facts and must store them
  Stage 2 (STM noise)         — distractors injected; agent must filter/compress
  Stage 3 (Unified)           — query issued; agent retrieves from LTM + manages STM

Integration notes:
  - For production training, replace synthetic generators with real HotpotQA
    (see hermes/loaders/hotpotqa_loader.py).
  - Add longer-horizon tasks (ALFWorld, SciWorld) for curriculum extension.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from datasets import Dataset

# ---------------------------------------------------------------------------
# System prompt / tool schema shown to the model
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Vocabulary for synthetic episode generation
# ---------------------------------------------------------------------------

NAMES: List[str] = [
    "Alice", "Bob", "Carlos", "Diana", "Ethan", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Marcus", "Nina", "Oscar", "Priya",
    "Quinn", "Rachel", "Samuel", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yasmine", "Zoe",
]

PREFS: List[str] = [
    "coffee", "tea", "sparkling water", "orange juice", "green tea",
    "black coffee", "hot chocolate", "lemonade", "chai", "espresso",
    "matcha latte", "herbal tea", "cold brew", "ginger ale", "cappuccino",
]

CITIES: List[str] = [
    "New York", "London", "Tokyo", "Paris", "Sydney", "Berlin", "Toronto",
    "Barcelona", "Singapore", "Amsterdam", "Seoul", "Dubai", "Mumbai",
    "São Paulo", "Chicago", "Vienna", "Zurich", "Stockholm", "Melbourne",
    "Copenhagen", "Prague", "Warsaw", "Helsinki", "Lisbon", "Athens",
]

DISTRACTORS: List[str] = [
    "Distractor: The weather today is partly cloudy with a chance of rain.",
    "Distractor: Stock markets rose 0.3% in early trading on Tuesday.",
    "Distractor: A new species of deep-sea fish was discovered near Greenland.",
    "Distractor: The annual tech conference has been postponed to next quarter.",
    "Distractor: Scientists have confirmed water ice deposits at the lunar south pole.",
    "Distractor: The recipe calls for two cups of flour and one teaspoon of vanilla.",
    "Distractor: Traffic congestion increased 12% in major cities last year.",
    "Distractor: The film festival received over 4,000 submissions this season.",
    "Distractor: A local startup raised $30 million in Series B funding.",
    "Distractor: Researchers published findings on migratory bird patterns.",
    "Distractor: The sports team won their third consecutive championship.",
    "Distractor: Municipal authorities plan to expand the public transit network.",
    "Distractor: A new art exhibit opened at the downtown gallery this weekend.",
    "Distractor: Temperature records were broken in several European capitals.",
    "Distractor: Quarterly earnings reports exceeded analyst expectations.",
]

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(stage: str, context_lines: List[str], user_query: str) -> str:
    """Build a stage-specific prompt with JSONL tool instructions."""
    stage_hint = {
        "stage1_ltm": (
            "Goal: Learn Long-Term Memory (LTM).\n"
            "- Store important facts in LTM.\n"
            "- You may answer by retrieving from LTM.\n"
            "- Avoid storing irrelevant information."
        ),
        "stage2_stm_noise": (
            "Goal: Learn Short-Term Memory (STM) management under distractors.\n"
            "- Filter/summary/discard distractors from STM.\n"
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


# ---------------------------------------------------------------------------
# Episode samplers
# ---------------------------------------------------------------------------

def _sample_fact_episode(r: random.Random) -> Tuple[List[str], List[str], str]:
    """Sample a single-person preference episode."""
    name = r.choice(NAMES)
    pref = r.choice(PREFS)
    city = r.choice(CITIES)

    context = [
        f"My name is {name}.",
        f"I prefer {pref}.",
        f"I live in {city}.",
    ]
    required = [pref.lower(), city.lower()]
    user_query = "What drink do I prefer and what city do I live in?"
    return context, required, user_query


def _sample_multi_person_episode(r: random.Random) -> Tuple[List[str], List[str], str]:
    """Sample an episode with two people to increase retrieval pressure."""
    names = r.sample(NAMES, 2)
    prefs = r.sample(PREFS, 2)
    cities = r.sample(CITIES, 2)

    context = [
        f"{names[0]} prefers {prefs[0]} and lives in {cities[0]}.",
        f"{names[1]} prefers {prefs[1]} and lives in {cities[1]}.",
    ]
    # Query about the first person only
    required = [prefs[0].lower(), cities[0].lower()]
    user_query = f"What drink does {names[0]} prefer and where do they live?"
    return context, required, user_query


def _inject_distractors(
    r: random.Random, context: List[str], min_k: int = 3, max_k: int = 5
) -> List[str]:
    """Interleave random distractors into the context list."""
    ctx = list(context)
    k = r.randint(min_k, max_k)
    for _ in range(k):
        ctx.insert(r.randint(0, len(ctx)), r.choice(DISTRACTORS))
    return ctx


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_examples(stage: str, n: int, seed: int = 0) -> List[Dict[str, Any]]:
    r = random.Random(seed)
    examples: List[Dict[str, Any]] = []

    for i in range(n):
        # Alternate between single-person and multi-person episodes
        if i % 3 == 2:
            context, required, user_query = _sample_multi_person_episode(r)
        else:
            context, required, user_query = _sample_fact_episode(r)

        # Stage-specific context shaping
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
    """Build a HuggingFace Dataset of GRPO training examples."""
    return Dataset.from_list(build_examples(stage=stage, n=n, seed=seed))
