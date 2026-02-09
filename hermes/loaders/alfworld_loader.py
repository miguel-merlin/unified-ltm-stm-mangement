"""ALFWorld dataset retrieval + Torch DataLoader.

Default source: Hugging Face dataset `awawa-agi/alfworld-raw`.
This contains the raw TextWorld + PDDL game files aligned with ALFRED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


def _require(pkg: str) -> None:
    raise SystemExit(
        f"Missing dependency: {pkg}.\n" "Install with: pip install datasets torch"
    )


try:
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    _require("torch")

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    _require("datasets")


@dataclass
class ALFWorldDataConfig:
    dataset_name: str = "awawa-agi/alfworld-raw"
    split: str = "train"
    cache_dir: Optional[str] = None


class ALFWorldRawDataset(Dataset):
    """Torch Dataset wrapper for ALFWorld raw game files."""

    def __init__(self, config: ALFWorldDataConfig):
        self.config = config
        self.ds = load_dataset(
            self.config.dataset_name,
            split=self.config.split,
            cache_dir=self.config.cache_dir,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, str]:  # type: ignore[override]
        row = self.ds[idx]
        return {
            "id": row["id"],
            "task_type": row["task_type"],
            "game_file_path": row["game_file_path"],
            "game_content": row["game_content"],
        }


class SimpleTokenizerCollate:
    """Optional collator to tokenize `game_content` with a HF tokenizer."""

    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, str]]):
        texts = [b["game_content"] for b in batch]
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        meta = {
            "id": [b["id"] for b in batch],
            "task_type": [b["task_type"] for b in batch],
            "game_file_path": [b["game_file_path"] for b in batch],
        }
        return {"tokens": tokens, "meta": meta}


def make_dataloader(
    config: ALFWorldDataConfig,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    dataset = ALFWorldRawDataset(config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    cfg = ALFWorldDataConfig(split="train")
    loader = make_dataloader(cfg, batch_size=2)
    batch = next(iter(loader))
    print(batch["id"][0])
    print(batch["task_type"][0])
    print(batch["game_file_path"][0])
    print(batch["game_content"][0][:500])
