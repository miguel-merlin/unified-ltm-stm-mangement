# Unified Long Term and Short Term Memory Management for LLMs

## Goals

- Enable LLMs to manage both long-term memory and short-term memory as a single, learnable policy
- Current systems separate LTM and STM with heuristic triggers or external controllers to decide when to store, retrieve, or compress information.
- Memory as a tool ADD, UPDATE, DELETE for LMT and RETRIEVE, SUMMARY, FILTER for STM directly controlled by the agent
- RL strategy to fine-tune the LLM to learn to build LTM then to manage the STM in the presence of distractors, then integrate both LTM and STM

Improvement ideas

- Reasoning memory bank. Store CoT chains in another memory bank
- Multi-agent orchestration use case. How can we extend Agentic Memory to multi-agent architectures
- Agents decide if memory globally relevant or relevant to their specific task
- Improve retrieval model based on whether the query of the user was satisfied - LLM-as-Judge to build our dataset
- Is language the best for memory retrievals? Can we do better than cosine similarity?
- Improve reward function for training RL algorithms

[Paper](https://arxiv.org/pdf/2601.01885)

## Quickstart

### 1. Installation

The repository is structured as an installable package. You must install it in editable mode alongside its dependencies.
We also require `peft` and `bitsandbytes` to perform memory-optimized LoRA training on standard GPUs.

```bash
pip install -e "."
pip install peft bitsandbytes sentence-transformers
```

### 2. The 3-Stage GRPO Training Pipeline

We implement the composite reward function ($R_{total} = w_{task}R_{task} + w_{context}R_{context} + w_{memory}R_{memory} - P_{penalty}$) directly natively integrated with TRL's Group Relative Policy Optimization (GRPO).

Train the model progressively using the provided entrypoint:

```bash
# Stage 1 — LTM construction (synthetic data)
python3 hermes/grpo_train.py --stage stage1_ltm --max_steps 500

# Stage 2 — STM noise injection (teaching distractor ignorance)
python3 hermes/grpo_train.py --stage stage2_stm_noise --max_steps 500

# Stage 3 — Realistic Unified Evaluation (HotpotQA)
python3 hermes/grpo_train.py --stage stage3_unified --use_hotpotqa --max_steps 3000
```

> **Note**: `hermes/grpo_train.py` defaults to `num_generations=4` and leverages **LoRA** and **gradient checkpointing** to naturally fit training a 7B model within 48GB/24GB GPUs. Pass `--num_generations 8` if you have high-end multi-GPU clusters to exactly match the paper's rollout bounds!

### 3. Evaluation & Benchmarking

After training, calculate the Memory Quality (MQ) and classical LLM-as-a-judge scores over the HotpotQA validation set:

```bash
python3 hermes/evaluate.py \
    --model_name outputs/grpo_qwen \
    --eval_hotpotqa \
    --eval_split validation \
    --max_samples 200 \
    --output_json outputs/eval_results.json
```

Generate the metric charts (Figures 2–5 from the paper) showcasing token efficiency and ablations:

```bash
python3 scripts/plot_results.py --eval_json outputs/eval_results.json --outdir outputs/plots
```

---

## Inference Testing

You can interactively test the agent's memory capability offline without training using the native standard scripts mapping to different backends:

### OpenAI Backend

```bash
python3 hermes/agent.py \
  --llm_backend openai \
  --api_key "$OPENAI_API_KEY" \
  --model gpt-4o-mini \
  --query "Remember that I prefer concise answers, then summarize."
```

### Auto Backend

`auto` uses `vllm` when a CUDA GPU is available, otherwise falls back to `openai`.

```bash
python3 hermes/agent.py \
  --llm_backend auto \
  --api_key "$OPENAI_API_KEY" \
  --query "Store my preference for concise answers."
```

### Local vLLM Inference

Start vLLM server:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
python3 hermes/agent.py \
  --llm_backend vllm \
  --vllm_base_url http://127.0.0.1:8000/v1 \
  --api_key EMPTY \
  --model Qwen/Qwen2.5-7B-Instruct \
  --query "Remember I like concise answers and summarize."
```
