from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from hermes.rewards import hermes_trace_reward
from hermes.tasks import build_dataset


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "outputs/grpo_qwen"
    seed: int = 42

    stage: str = "stage3_unified"
    n: int = 200

    max_steps: int = 200
    per_device_train_batch_size: int = 1
    num_generations: int = 4
    learning_rate: float = 5e-7
    max_new_tokens: int = 256


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=TrainConfig.model_name)
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--stage", type=str, default=TrainConfig.stage)
    parser.add_argument("--n", type=int, default=TrainConfig.n)
    parser.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--num_generations", type=int, default=TrainConfig.num_generations)
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )

    dataset = build_dataset(stage=cfg.stage, n=cfg.n, seed=cfg.seed)

    def preprocess(example):
        messages = [
            {"role": "system", "content": "You are a memory-management agent."},
            {"role": "user", "content": example["prompt"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": text, "required": example["required"], "stage": example["stage"]}

    dataset = dataset.map(preprocess)

    grpo_config = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        num_generations=cfg.num_generations,
        max_new_tokens=cfg.max_new_tokens,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_funcs=[hermes_trace_reward],
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()
