"""RL training pipeline for foundational LMs (PPO with TRL).

This is a minimal, extensible script that:
- loads an open-source causal LM
- defines a simple RL task with a reward function
- optimizes the LM with PPO (TRL)
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Iterable, List


def _require(pkg: str) -> None:
    raise SystemExit(
        f"Missing dependency: {pkg}.\n"
        "Install with: pip install transformers datasets trl accelerate torch"
    )


try:
    import torch
except Exception:  # pragma: no cover
    _require("torch")

try:
    from datasets import Dataset
except Exception:  # pragma: no cover
    _require("datasets")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    _require("transformers")

try:
    from trl.trainer.ppo_config import PPOConfig
    from trl.trainer.ppo_trainer import PPOTrainer
except Exception:  # pragma: no cover
    _require("trl")


@dataclass
class TrainConfig:
    model_name: str = "gpt2"
    output_dir: str = "outputs/rl"
    seed: int = 42
    max_steps: int = 200
    batch_size: int = 4
    mini_batch_size: int = 2
    learning_rate: float = 1e-5
    max_new_tokens: int = 64
    prompt_file: str | None = None
    device: str | None = None


DEFAULT_PROMPTS = [
    "Summarize the user request in one sentence.",
    "Extract the key facts from the note.",
    "Decide whether this memory should be stored: 'User likes jazz'.",
    "Given the task, retrieve only relevant memories.",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_prompts(prompt_file: str | None) -> List[str]:
    if not prompt_file:
        return list(DEFAULT_PROMPTS)
    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [l for l in lines if l]


def build_dataset(prompts: Iterable[str]) -> Dataset:
    return Dataset.from_dict({"prompt": list(prompts)})


# --- Reward function ---
# Replace this with a task-specific reward for memory management.
# Example here: reward outputs that include the keyword "MEMORY"
# and keep responses concise.


def compute_reward(text: str) -> float:
    reward = 0.0
    if "memory" in text.lower():
        reward += 1.0
    if len(text.split()) <= 20:
        reward += 0.5
    reward -= max(0, len(text.split()) - 40) * 0.01
    return reward


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL training pipeline for an open-source LM"
    )
    parser.add_argument("--model_name", type=str, default=TrainConfig.model_name)
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument(
        "--mini_batch_size", type=int, default=TrainConfig.mini_batch_size
    )
    parser.add_argument(
        "--learning_rate", type=float, default=TrainConfig.learning_rate
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=TrainConfig.max_new_tokens
    )
    parser.add_argument("--prompt_file", type=str, default=TrainConfig.prompt_file)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    prompts = load_prompts(cfg.prompt_file)
    dataset = build_dataset(prompts)

    ppo_config = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    step = 0
    while step < cfg.max_steps:
        for batch in trainer.dataloader:
            if step >= cfg.max_steps:
                break

            prompts_batch = batch["prompt"]
            tokenized = tokenizer(
                prompts_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            response_tensors = trainer.generate_completions(tokenized["input_ids"])

            responses = tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )
            rewards = [compute_reward(r) for r in responses]
            trainer.step(tokenized["input_ids"], response_tensors, rewards)
            step += 1
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()
