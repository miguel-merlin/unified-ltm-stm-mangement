"""Training logger with W&B / TensorBoard / CSV backends.

Provides a unified interface so training code doesn't need to know
which logging backend is active. All backends are optional with graceful
fallback to CSV-only mode.

Usage:
    logger = TrainingLogger(
        log_dir="outputs/logs",
        use_wandb=True,
        run_name="stage1_qwen",
        project="agemem",
        config={"lr": 5e-7, "stage": "stage1_ltm"},
    )
    for step in training_loop:
        logger.log_step(step, {"reward": 0.42, "loss": 1.23})
    logger.close()
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    """Unified logger supporting W&B, TensorBoard, and CSV."""

    def __init__(
        self,
        log_dir: str = "outputs/logs",
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        use_csv: bool = True,
        run_name: str = "agemem_run",
        project: str = "agemem",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self._start_time = time.time()

        # ---- W&B ----
        self._wandb = None
        if use_wandb:
            try:
                import wandb  # type: ignore[import]
                self._wandb = wandb
                wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                )
                print(f"[Logger] W&B run: {wandb.run.url}")
            except ImportError:
                print("[Logger] WARNING: wandb not installed; skipping W&B logging.")
            except Exception as e:
                print(f"[Logger] WARNING: W&B init failed ({e}); skipping.")

        # ---- TensorBoard ----
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
                tb_dir = self.log_dir / "tensorboard" / run_name
                self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
                print(f"[Logger] TensorBoard dir: {tb_dir}")
            except ImportError:
                print("[Logger] WARNING: tensorboard not installed; skipping.")
            except Exception as e:
                print(f"[Logger] WARNING: TensorBoard init failed ({e}); skipping.")

        # ---- CSV ----
        self._csv_file = None
        self._csv_writer = None
        if use_csv:
            csv_path = self.log_dir / f"{run_name}_metrics.csv"
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            print(f"[Logger] CSV log: {csv_path}")
            # Header will be written on first log_step call
            self._csv_header_written = False

        # ---- JSON summary ----
        self._config_path = self.log_dir / f"{run_name}_config.json"
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(config or {}, f, indent=2)

        self._step_count = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def log_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log a dict of scalar metrics at the given global step."""
        self._step_count += 1
        metrics = {k: v for k, v in metrics.items() if v is not None}
        # Add wall-clock time
        metrics["elapsed_sec"] = round(time.time() - self._start_time, 1)
        metrics["step"] = step

        # W&B
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass

        # TensorBoard
        if self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    try:
                        self._tb_writer.add_scalar(k, v, global_step=step)
                    except Exception:
                        pass

        # CSV
        if self._csv_file is not None:
            if not self._csv_header_written:
                fieldnames = list(metrics.keys())
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=fieldnames, extrasaction="ignore"
                )
                self._csv_writer.writeheader()
                self._csv_header_written = True
            try:
                self._csv_writer.writerow(metrics)
                self._csv_file.flush()
            except Exception:
                pass

        # Console
        reward = metrics.get("reward", metrics.get("train/reward"))
        loss = metrics.get("loss", metrics.get("train/loss"))
        print(
            f"[Step {step:>5}] "
            + (f"reward={reward:.4f}  " if reward is not None else "")
            + (f"loss={loss:.4f}" if loss is not None else "")
        )

    def log_eval(self, step: int, eval_metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics (prefixed with 'eval/')."""
        prefixed = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.log_step(step, prefixed)

    def close(self) -> None:
        """Flush and close all logging backends."""
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
        print(
            f"[Logger] Closed after {self._step_count} steps "
            f"({round(time.time() - self._start_time, 1)}s elapsed)."
        )
