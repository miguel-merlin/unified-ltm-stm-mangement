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