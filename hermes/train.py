"""Legacy training entrypoint — superseded by grpo_train.py.

This file is kept for reference only. All training should use:
    python hermes/grpo_train.py [args]

The original file had several issues that are documented here:
  - Syntax error: `from trl import GRPOConfig, GRPO Trainer` (space breaks import)
  - Mixed PPO/GRPO imports that conflict with current TRL API
  - Incorrect GRPOTrainer instantiation pattern
"""

import sys

if __name__ == "__main__":
    print(
        "WARNING: hermes/train.py is deprecated.\n"
        "Use `python hermes/grpo_train.py` instead.",
        file=sys.stderr,
    )
    sys.exit(1)
