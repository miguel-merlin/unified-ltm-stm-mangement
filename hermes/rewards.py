from __future__ import annotations

from typing import List, Optional

from hermes.memory import LongTermMemory, ShortTermMemory
from hermes.tool_api import HermesToolAPI
from hermes.trace import parse_jsonl_trace

def _contains_all(answer: str, required: List[str]) -> bool:
    a = answer.lower()
    return all(x.lower() in a for x in required)

def hermes_trace_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Transformer Reinforcement Learning(TRL) GRPO.
    """
    required: Optional[List[str]] = kwargs.get("required")
    stage: str = kwargs.get("stage", "stage3_unified")

    rewards: List[float] = []

    for completion in completions:
        stm = ShortTermMemory()
        ltm = LongTermMemory()
        tools = HermesToolAPI(stm, ltm)

        trace, final_answer, status = parse_jsonl_trace(completion, max_lines=12)

        if status == "json_parse_error":
            rewards.append(-1.0)
            continue
        if status == "missing_final":
            rewards.append(-0.7)
            continue
        if status == "invalid_schema":
            rewards.append(-0.8)
            continue

        tool_calls = 0
        ltm_writes = 0
        retrieved_to_stm = 0

        # Execute trace actions against STM/LTM
        for obj in trace:
            tool = obj.get("tool")
            action = obj.get("action")
            content = str(obj.get("content", ""))
            k = int(obj.get("k", 5))

            if tool == "stm":
                tool_calls += 1
                tools.stm_tool(action, content=content, k=k)

            elif tool == "ltm" and action == "retrieve_to_stm":
                tool_calls += 1
                _, injected = tools.ltm_retrieve_to_stm(query=content, k=k)
                if injected > 0:
                    retrieved_to_stm += 1

            elif tool == "ltm":
                tool_calls += 1
                if action in ("add", "update", "delete"):
                    ltm_writes += 1

                tools.ltm_tool(
                    action,
                    content=content,
                    key=str(obj.get("key", "")),
                    k=k,
                    tags=obj.get("tags"),
                    importance=float(obj.get("importance", 0.0)),
                    meta=obj.get("meta"),
                    tags_any=obj.get("tags_any"),
                    meta_filter=obj.get("meta_filter"),
                )

        r = 0.0

        # Primary task success signal
        if required:
            r += 1.0 if _contains_all(final_answer, required) else -0.1
        else:
            # if labels missing, final answer must still exist
            r += 0.1 if final_answer.strip() else -0.2

        # Slight penalty for too-long traces 
        if status == "too_long":
            r -= 0.2

        # Efficiency penalties
        r -= 0.01 * tool_calls
        r -= 0.02 * ltm_writes

        # Encourage unified behavior in Stage 3 (small bonus for retrieving into STM)
        if stage == "stage3_unified" and retrieved_to_stm > 0:
            r += 0.05

        # Noise handling penalty (Stage 2&3)
        if stage in ("stage2_stm_noise", "stage3_unified"):
            stm_text = " ".join(list(stm._buffer)).lower()
            r -= 0.03 * stm_text.count("distractor:")

        # LTM bloat penalty (Stage 3)
        if stage == "stage3_unified":
            r -= 0.005 * max(0, len(ltm.snapshot()) - 8)

        rewards.append(r)

    return rewards
