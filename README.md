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

## Run The Application

### 1. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Hermes agent (OpenAI backend)
```bash
python3 hermes/agent.py \
  --llm_backend openai \
  --api_key "$OPENAI_API_KEY" \
  --model gpt-4o-mini \
  --query "Remember that I prefer concise answers, then summarize."
```

### 3. Run Hermes agent (auto backend)
`auto` uses `vllm` when CUDA GPU is available, otherwise falls back to `openai`.

```bash
python3 hermes/agent.py \
  --llm_backend auto \
  --api_key "$OPENAI_API_KEY" \
  --query "Store my preference for concise answers."
```

### 4. Run with local vLLM (GPU required)

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
