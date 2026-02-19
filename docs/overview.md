# Hermes: Unified LTM and STM management

## Core Purpose
The goal of this project is to implement **AgeMem**, a framework that enables Large Language Model (LLM) agents to learn unified management of both **Short-Term Memory (STM)** and **Long-Term Memory (LTM)**. Unlike traditional agents that rely on fixed heuristics (like sliding windows or simple RAG), AgeMem allows an agent to autonomously decide what information to retain, what to move to long-term storage, and what to discard based on the task context.

## Key Objectives
* **Unified Memory Management**: Replace manual memory strategies with a learned policy that optimizes for long-horizon tasks.
* **Adaptive Context Handling**: Enable agents to manage limited context windows efficiently by prioritizing "task-relevant" over "recency-based" information.
* **Experience-Driven Growth**: Utilize a three-stage progressive reinforcement learning (RL) strategy to refine memory operations based on performance feedback.

## Technical Architecture
The codebase is structured around the following components:
1.  **Memory Operations**: 
    * `Retain`: Keeps critical information in the immediate context (STM).
    * `Archive`: Offloads relevant but not immediately necessary info to LTM.
    * `Discard`: Removes redundant or irrelevant data to save context space.
2.  **Controller (Policy)**: A learned module (often a smaller LLM) that generates memory management actions alongside standard task responses.
3.  **Reinforcement Learning Pipeline**:
    * **Stage 1: Learning from Heuristics**: Initializing the policy via imitation learning from basic strategies.
    * **Stage 2: Outcome-Based Tuning**: Refining the policy using task success as a reward signal.
    * **Stage 3: Fine-Grained Optimization**: Using trajectory-level feedback to sharpen memory-specific decisions.

## Implementation Context for Agents
When working within this repository, agents should focus on:
* **`src/memory/`**: Implementation of the LTM vector store and STM buffer management.
* **`src/policy/`**: The logic governing memory action selection (Retain/Archive/Discard).
* **`src/training/`**: Scripts for the three-stage progressive RL training loop.

## Evaluation Benchmarks
The project measures success using:
* **ScienceWorld / ALFWorld**: Testing long-term planning and state tracking.
* **AgentBoard**: Comprehensive evaluation across multi-turn interactions.
* **HotpotQA**: Testing retrieval-heavy reasoning.

---
**Summary for Agents:**
You are maintaining a system that treats memory management as a **trainable skill** rather than a hard-coded utility. Every time the agent interacts with the environment, it must also manage its internal state to ensure it doesn't lose track of vital information during long, complex tasks.

## Running Hermes Locally

Run with OpenAI:
```bash
python3 hermes/agent.py \
  --llm_backend openai \
  --api_key "$OPENAI_API_KEY" \
  --model gpt-4o-mini \
  --query "Remember that I prefer concise replies."
```

Run with local vLLM (GPU required):
```bash
# terminal 1
vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000

# terminal 2
python3 hermes/agent.py \
  --llm_backend vllm \
  --vllm_base_url http://127.0.0.1:8000/v1 \
  --api_key EMPTY \
  --model Qwen/Qwen2.5-7B-Instruct \
  --query "Store this preference and summarize."
```

Use `--llm_backend auto` to prefer vLLM when a CUDA GPU is available and otherwise fall back to OpenAI.
