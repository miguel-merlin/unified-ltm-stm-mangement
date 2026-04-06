"""BabyAI benchmark loader for AgeMem evaluation.

The paper uses BabyAI (chevalier2018babyai) to evaluate instruction-following
in grid-world navigation tasks. Metric: Success Rate (SR).

This loader wraps the `minigrid` / `babyai-text` HuggingFace dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


@dataclass
class BabyAIDataConfig:
    dataset_name: str = "McGill-NLP/babyai-text"
    level: str = "BabyAI-GoToObj-v0"
    split: str = "test"
    cache_dir: Optional[str] = None
    max_samples: Optional[int] = None


def load_babyai_tasks(config: Optional[BabyAIDataConfig] = None) -> List[Dict[str, Any]]:
    """Load BabyAI instruction-following tasks.

    Returns a list of dicts with keys:
        task_id, instruction, mission, optimal_action_sequence (list)
    """
    if config is None:
        config = BabyAIDataConfig()

    try:
        raw = load_dataset(
            config.dataset_name,
            config.level,
            split=config.split,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[BabyAI] Could not load from HuggingFace: {e}")
        print("[BabyAI] Returning empty task list. Ensure 'minigrid' is installed.")
        return []

    tasks = []
    for i, row in enumerate(raw):
        if config.max_samples and i >= config.max_samples:
            break
        tasks.append({
            "task_id": str(row.get("id", i)),
            "instruction": str(row.get("mission", row.get("instruction", ""))),
            "mission": str(row.get("mission", "")),
            "optimal_action_sequence": list(row.get("optimal_action_sequence", [])),
            "descriptions": list(row.get("descriptions", [])),
        })

    print(f"[BabyAI] Loaded {len(tasks)} tasks from level '{config.level}'.")
    return tasks


def compute_success_rate(
    completed_tasks: List[bool],
) -> float:
    """Compute success rate from a boolean list of task outcomes."""
    if not completed_tasks:
        return 0.0
    return sum(completed_tasks) / len(completed_tasks)
