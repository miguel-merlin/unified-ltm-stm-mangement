"""Reward functions for AgeMem GRPO training.

Implements the full composite reward from Section 3.5 of the paper:
    R_total = w_task * R_task + w_context * R_context + w_memory * R_memory - P_penalty

All weights default to 1/3 (uniform) as described in Appendix C.4.
"""

from __future__ import annotations

from typing import List, Optional, Set

from hermes.memory import LongTermMemory, ShortTermMemory
from hermes.tool_api import HermesToolAPI
from hermes.trace import parse_jsonl_trace


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _contains_all(answer: str, required: List[str]) -> bool:
    a = answer.lower()
    return all(x.lower() in a for x in required)


def _extract_key_tokens(text: str) -> Set[str]:
    """Extract crude 'key tokens' (non-stopword words ≥ 4 chars) from a query."""
    STOPWORDS = {
        "what", "when", "where", "which", "that", "this", "with",
        "from", "have", "does", "will", "should", "could", "would",
        "been", "more", "some", "into", "also", "just", "than",
    }
    words = set(text.lower().split())
    return {w for w in words if len(w) >= 4 and w not in STOPWORDS}


# ---------------------------------------------------------------------------
# Sub-reward components
# ---------------------------------------------------------------------------

def _r_task(final_answer: str, required: Optional[List[str]]) -> float:
    """R_task: primary signal — does the answer contain all required tokens?

    Returns a score in [-1, 1]:
      +1.0  all required tokens present
      -0.1  partial miss (required exists but not all matched)
      +0.1  no labels but answer exists
      -1.0  no answer at all
    """
    if not final_answer.strip():
        return -1.0
    if required:
        return 1.0 if _contains_all(final_answer, required) else -0.1
    return 0.1


def _r_context(
    stm: ShortTermMemory,
    tool_calls: int,
    stm_tool_counts: dict,
    status: str,
    token_budget: int = 8192,
) -> float:
    """R_context: evaluates STM management quality.

    Three equally-weighted sub-components (each ∈ [0,1]):
      - Compression efficiency: how compact is the final context?
      - Preventive action: did the agent proactively compress/filter?
      - Trace validity: no penalty for too-long traces
    """
    # Compression efficiency — proxy: fraction of budget *not* used
    # (we count STM items as a proxy for token usage)
    n_items = len(stm._items)
    capacity = max(stm.capacity, 1)
    r_compression = max(0.0, 1.0 - (n_items / capacity))

    # Preventive action — agent invoked summary or filter proactively
    preventive_actions = stm_tool_counts.get("summary", 0) + stm_tool_counts.get("filter", 0)
    r_preventive = min(1.0, preventive_actions * 0.5)

    # Information preservation — penalise too-long traces (context exploded)
    r_preservation = 0.0 if status == "too_long" else 1.0

    alpha = 1.0 / 3.0
    return alpha * r_compression + alpha * r_preventive + alpha * r_preservation


def _r_memory(
    ltm: LongTermMemory,
    ltm_writes: int,
    ltm_maintenance_ops: int,
    required: Optional[List[str]],
    final_answer: str,
) -> float:
    """R_memory: evaluates LTM operation quality.

    Three equally-weighted sub-components (each ∈ [0,1]):
      - Storage quality: fraction of stored entries with query-aligned content
      - Maintenance: reward meaningful update/delete ops (capped)
      - Semantic relevance: do retrieved/stored facts align with the answer?
    """
    snapshot = ltm.snapshot()
    n_total = max(len(snapshot), 1)

    # Storage quality — stored entries that contain at least one required token
    if required:
        key_tokens = set(t.lower() for t in required)
        n_high_quality = sum(
            1 for rec in snapshot.values()
            if any(tok in rec["content"].lower() for tok in key_tokens)
        )
        r_storage = n_high_quality / n_total
    else:
        # Heuristic: entries with importance > 0 count as "high quality"
        n_high_quality = sum(
            1 for rec in snapshot.values() if rec.get("importance", 0) > 0
        )
        r_storage = n_high_quality / n_total if snapshot else 0.5

    # Maintenance — reward update/delete operations (max 1.0 at 3 ops)
    r_maintenance = min(1.0, ltm_maintenance_ops / max(ltm_writes, 3))

    # Semantic relevance — do LTM contents overlap with final answer tokens?
    answer_tokens = _extract_key_tokens(final_answer)
    if answer_tokens and snapshot:
        n_relevant = sum(
            1 for rec in snapshot.values()
            if any(tok in rec["content"].lower() for tok in answer_tokens)
        )
        r_relevance = min(1.0, n_relevant / n_total)
    else:
        r_relevance = 0.0

    beta = 1.0 / 3.0
    return beta * r_storage + beta * r_maintenance + beta * r_relevance


def _p_penalty(
    tool_calls: int,
    ltm_writes: int,
    stm_text: str,
    n_ltm_entries: int,
    stage: str,
    max_tool_calls: int = 10,
    max_ltm_entries: int = 8,
) -> float:
    """P_penalty: penalises excessive or wasteful memory usage.

    Components (negative values):
      - Tool overuse penalty
      - LTM write cost
      - Distractor persistence (Stage 2 & 3)
      - LTM bloat (Stage 3)
    """
    p = 0.0

    # Efficiency: small per-tool-call cost
    p -= 0.01 * tool_calls

    # LTM write cost (writes are more expensive than reads)
    p -= 0.02 * ltm_writes

    # Distractor persistence penalty (Stages 2 & 3)
    if stage in ("stage2_stm_noise", "stage3_unified"):
        p -= 0.03 * stm_text.count("distractor:")

    # LTM bloat penalty (Stage 3 only)
    if stage == "stage3_unified":
        p -= 0.005 * max(0, n_ltm_entries - max_ltm_entries)

    return p


# ---------------------------------------------------------------------------
# Main TRL-compatible reward function
# ---------------------------------------------------------------------------

def hermes_trace_reward(completions: List[str], **kwargs) -> List[float]:
    """Full AgeMem composite reward for TRL GRPOTrainer.

    Signature matches TRL's reward_funcs protocol:
        reward_funcs: List[Callable[[List[str], **kwargs], List[float]]]

    kwargs passed by GRPOTrainer may include:
      - required: List[str]  — tokens the final answer must contain
      - stage:    str        — training stage name

    Returns a list of scalar rewards, one per completion.
    """
    required: Optional[List[str]] = kwargs.get("required")
    stage: str = kwargs.get("stage", "stage3_unified")

    # Reward weights (uniform as per paper Appendix C.4)
    w_task = w_context = w_memory = 1.0 / 3.0

    rewards: List[float] = []

    for completion in completions:
        stm = ShortTermMemory()
        ltm = LongTermMemory()
        tools = HermesToolAPI(stm, ltm)

        trace, final_answer, status = parse_jsonl_trace(completion, max_lines=12)

        # Hard failures — immediately penalise without decomposition
        if status == "json_parse_error":
            rewards.append(-1.0)
            continue
        if status == "missing_final":
            rewards.append(-0.7)
            continue
        if status == "invalid_schema":
            rewards.append(-0.8)
            continue

        # Execute the trace against live STM/LTM to measure actual behaviour
        tool_calls = 0
        ltm_writes = 0
        ltm_maintenance_ops = 0
        stm_tool_counts: dict = {}

        for obj in trace:
            tool = obj.get("tool")
            action = obj.get("action", "")
            content = str(obj.get("content", ""))
            k = int(obj.get("k", 5))

            if tool == "stm":
                tool_calls += 1
                stm_tool_counts[action] = stm_tool_counts.get(action, 0) + 1
                tools.stm_tool(action, content=content, k=k)

            elif tool == "ltm" and action == "retrieve_to_stm":
                tool_calls += 1
                tools.ltm_retrieve_to_stm(query=content, k=k)

            elif tool == "ltm":
                tool_calls += 1
                if action == "add":
                    ltm_writes += 1
                elif action in ("update", "delete"):
                    ltm_writes += 1
                    ltm_maintenance_ops += 1

                tools.ltm_tool(
                    action,
                    content=content,
                    key=str(obj.get("key", "")),
                    k=k,
                    tags=obj.get("tags"),
                    importance=float(obj.get("importance", 0.0)),
                    meta=obj.get("meta"),
                    tags_any=obj.get("tags_any"),
                    meta_filter=obj.get("meta_filter"),
                )

        # Compute the three sub-rewards
        r_task = _r_task(final_answer, required)
        r_context = _r_context(stm, tool_calls, stm_tool_counts, status)
        r_memory = _r_memory(
            ltm, ltm_writes, ltm_maintenance_ops, required, final_answer
        )
        stm_text = " ".join(stm._buffer).lower()
        n_ltm = len(ltm.snapshot())
        penalty = _p_penalty(
            tool_calls, ltm_writes, stm_text, n_ltm, stage
        )

        total = w_task * r_task + w_context * r_context + w_memory * r_memory + penalty
        rewards.append(float(total))

    return rewards
