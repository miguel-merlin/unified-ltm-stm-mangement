"""GRPO training entrypoint for AgeMem.

Trains Qwen2.5-7B-Instruct or Qwen3-4B-Instruct with the three-stage
progressive RL strategy from Section 3.3 of the paper.

Usage:
    # Stage 1 — LTM construction
    python hermes/grpo_train.py --stage stage1_ltm --max_steps 200

    # Stage 2 — STM noise management
    python hermes/grpo_train.py --stage stage2_stm_noise --max_steps 200

    # Stage 3 — Unified (loads a stage-2 checkpoint)
    python hermes/grpo_train.py --stage stage3_unified --max_steps 200

    # HotpotQA-based training (use real QA data)
    python hermes/grpo_train.py --use_hotpotqa --max_steps 500
"""

from __future__ import annotations

import argparse
import dataclasses
import random
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from hermes.rewards import hermes_trace_reward
from hermes.tasks import build_dataset
from hermes.logging_utils import TrainingLogger


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "outputs/grpo_qwen"
    seed: int = 42

    # Curriculum
    stage: str = "stage3_unified"
    n: int = 200
    use_hotpotqa: bool = False
    hotpotqa_split: str = "train"
    hotpotqa_max_samples: int = 1000

    # GRPO hyperparameters (from Appendix C.4)
    max_steps: int = 200
    per_device_train_batch_size: int = 1
    num_generations: int = 4           # K rollouts (paper uses 8; must be >=2. Use 8 with large GPU)
    learning_rate: float = 5e-7
    max_completion_length: int = 256   # max_new_tokens in TRL 1.0.0
    kl_coeff: float = 0.1              # β KL divergence coefficient (beta in TRL 1.0.0)

    # Memory optimization
    use_peft: bool = True              # Use LoRA to save memory
    gradient_checkpointing: bool = True # Save memory at the cost of speed

    # Logging
    log_wandb: bool = False
    log_tensorboard: bool = False
    log_csv: bool = True
    wandb_project: str = "agemem"
    wandb_run_name: str = ""
    log_every_n_steps: int = 10


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dataclass_from_namespace(cls, ns: argparse.Namespace):
    """Build a dataclass instance from argparse namespace, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in vars(ns).items() if k in known}
    return cls(**kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_grpo_dataset(cfg: TrainConfig):
    """Select dataset based on config: HotpotQA or synthetic."""
    if cfg.use_hotpotqa:
        from hermes.loaders.hotpotqa_loader import build_hotpotqa_dataset
        return build_hotpotqa_dataset(
            stage=cfg.stage,
            split=cfg.hotpotqa_split,
            max_samples=cfg.hotpotqa_max_samples,
            seed=cfg.seed,
        )
    return build_dataset(stage=cfg.stage, n=cfg.n, seed=cfg.seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AgeMem GRPO training")

    # Add all TrainConfig fields as CLI args
    for f in fields(TrainConfig):
        if f.type in (bool,) or str(f.type) in ("bool", "<class 'bool'>"):
            parser.add_argument(
                f"--{f.name}",
                action="store_true" if not f.default else "store_false",
                default=f.default,
            )
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)

    args = parser.parse_args()
    cfg = _dataclass_from_namespace(TrainConfig, args)

    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AgeMem] Training on device: {device}")
    print(f"[AgeMem] Stage: {cfg.stage} | Model: {cfg.model_name}")

    # ---- Load model & tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        # Load in 8bit if PEFT is enabled to save more VRAM, though 16bit is standard.
        # But for OOM safety, we stick to bfloat16 for PEFT so as not to complicate dependency on bitsandbytes unless necessary.
    )

    peft_config = None
    if cfg.use_peft:
        try:
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
                bias="none",
            )
            print("[AgeMem] Enabled PEFT (LoRA) for memory optimization.")
        except ImportError:
            print("[AgeMem] WARNING: peft not installed. Proceeding without LoRA.")

    # ---- Build dataset ----
    dataset = build_grpo_dataset(cfg)

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
        return {
            "prompt": text,
            "required": example.get("required", []),
            "stage": example.get("stage", cfg.stage),
        }

    dataset = dataset.map(preprocess)

    # ---- Logger ----
    run_name = cfg.wandb_run_name or f"{cfg.stage}_{cfg.model_name.split('/')[-1]}"
    logger = TrainingLogger(
        log_dir=str(Path(cfg.output_dir) / "logs"),
        use_wandb=cfg.log_wandb,
        use_tensorboard=cfg.log_tensorboard,
        use_csv=cfg.log_csv,
        run_name=run_name,
        project=cfg.wandb_project,
        config=dataclasses.asdict(cfg),
    )

    # ---- GRPO config ----
    # TRL 1.0.0 constraint: generation_batch_size % num_generations == 0
    # and num_generations >= 2. Set generation_batch_size = num_generations
    # so the constraint is always satisfied regardless of batch size.
    n_gen = max(cfg.num_generations, 2)  # TRL requires >= 2

    grpo_config = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        num_generations=n_gen,
        generation_batch_size=n_gen,     # must be divisible by num_generations
        max_completion_length=cfg.max_completion_length,
        beta=cfg.kl_coeff,
        logging_steps=cfg.log_every_n_steps,
        save_steps=max(cfg.max_steps // 5, 1),
        report_to=[],
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    # ---- Trainer ----
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,      # TRL 1.0.0: renamed from 'tokenizer'
        reward_funcs=[hermes_trace_reward],
        peft_config=peft_config,
    )

    # Inject logger callbacks
    class LogCallback:
        """Custom callback that pipes TRL metrics to our TrainingLogger."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.global_step % cfg.log_every_n_steps == 0:
                step = state.global_step
                reward = logs.get("reward", logs.get("train/reward", None))
                logger.log_step(
                    step=step,
                    metrics={
                        "reward": reward,
                        "loss": logs.get("loss", logs.get("train/loss")),
                        "grad_norm": logs.get("grad_norm"),
                        **{k: v for k, v in logs.items()},
                    },
                )

    try:
        from transformers import TrainerCallback
        class _CB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and state.global_step % max(cfg.log_every_n_steps, 1) == 0:
                    logger.log_step(step=state.global_step, metrics=logs)
        trainer.add_callback(_CB())
    except Exception:
        pass  # non-critical

    print(f"[AgeMem] Starting training — {cfg.max_steps} steps")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    logger.close()
    print(f"[AgeMem] Training complete. Model saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
