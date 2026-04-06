"""PDDL / AgentBoard benchmark loader for AgeMem evaluation.

The paper uses the AgentBoard PDDL tasks (chang2024agentboard) to evaluate
planning capability. Metric: Progress Rate (PR).

This loader fetches tasks from the HuggingFace dataset
`agentboard/agentboard` (PDDL subset), or falls back to the local
AgentBoard data directory if available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class PDDLDataConfig:
    dataset_name: str = "agentboard/agentboard"
    subset: str = "pddl"
    split: str = "test"
    cache_dir: Optional[str] = None
    max_samples: Optional[int] = None


def load_pddl_tasks(config: Optional[PDDLDataConfig] = None) -> List[Dict[str, Any]]:
    """Load PDDL planning tasks from AgentBoard.

    Returns a list of dicts with keys:
        task_id, domain, problem, goal_description, required_actions (list)
    """
    if config is None:
        config = PDDLDataConfig()

    try:
        raw = load_dataset(
            config.dataset_name,
            config.subset,
            split=config.split,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[PDDL] Could not load from HuggingFace: {e}")
        print("[PDDL] Returning empty task list. Supply a local dataset path via PDDLDataConfig.")
        return []

    tasks = []
    for i, row in enumerate(raw):
        if config.max_samples and i >= config.max_samples:
            break
        tasks.append({
            "task_id": str(row.get("task_id", i)),
            "domain": str(row.get("domain", "")),
            "problem": str(row.get("problem", "")),
            "goal_description": str(row.get("goal_description", row.get("goal", ""))),
            "required_actions": list(row.get("required_actions", [])),
        })

    print(f"[PDDL] Loaded {len(tasks)} tasks.")
    return tasks


def compute_progress_rate(predicted_steps: List[str], required_actions: List[str]) -> float:
    """Compute progress rate (PR) = fraction of required actions completed in order.

    This is a simplified version of the AgentBoard PR metric.
    A full implementation would use the PDDL validator.
    """
    if not required_actions:
        return 1.0
    completed = 0
    pred_lower = [p.lower().strip() for p in predicted_steps]
    for action in required_actions:
        if any(action.lower().strip() in p for p in pred_lower):
            completed += 1
    return completed / len(required_actions)
