"""SciWorld dataset retrieval + Torch DataLoader.

Default source: Hugging Face dataset `ZHLiu627/sciworld_dataset`.
This dataset includes instruction-style triples (instruction, input, output).
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
class SciWorldDataConfig:
    dataset_name: str = "ZHLiu627/sciworld_dataset"
    split: str = "train"
    cache_dir: Optional[str] = None
    data_files: Optional[object] = None
    streaming: bool = False


def _build_prompt(instruction: str, input_text: str) -> str:
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


class SciWorldInstructionDataset(Dataset):
    """Torch Dataset wrapper for instruction-style SciWorld data."""

    def __init__(self, config: SciWorldDataConfig):
        self.config = config
        filtered_args = {k: v for k, v in vars(self.config).items() if v is not None}
        self.ds = load_dataset(**filtered_args)

    def __len__(self) -> int:  # type: ignore[override]
        if self.config.streaming:  # pragma: no cover
            raise TypeError("Length is not available for streaming datasets.")
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Optional[str]]:  # type: ignore[override]
        row = self.ds[idx]
        instruction = row.get("instruction", "")
        input_text = row.get("input", "")
        output_text = row.get("output", "")
        file_id = row.get("file_id") or row.get("id") or row.get("task_id")
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "file_id": file_id,
        }


class SimpleTokenizerCollate:
    """Optional collator to tokenize instruction+input with a HF tokenizer."""

    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Optional[str]]]):
        prompts = [
            _build_prompt(b["instruction"] or "", b["input"] or "") for b in batch
        ]
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        meta = {
            "instruction": [b["instruction"] for b in batch],
            "input": [b["input"] for b in batch],
            "output": [b["output"] for b in batch],
            "file_id": [b["file_id"] for b in batch],
        }
        return {"tokens": tokens, "meta": meta}


def make_dataloader(
    config: SciWorldDataConfig,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    dataset = SciWorldInstructionDataset(config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    cfg = SciWorldDataConfig(split="train")
    loader = make_dataloader(cfg, batch_size=2)
    batch = next(iter(loader))
    print(batch["instruction"][0])
    print(batch["input"][0])
    print(batch["output"][0])
