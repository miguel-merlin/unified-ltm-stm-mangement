from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple

VALID_TOOLS = {"stm", "ltm"}
VALID_STM_ACTIONS = {"retain", "discard", "retrieve", "summary", "filter"}
VALID_LTM_ACTIONS = {"add", "retrieve", "update", "delete", "get", "list", "clear", "retrieve_to_stm"}

Status = str  # "ok" | "json_parse_error" | "missing_final" | "too_long" | "invalid_schema"

def parse_jsonl_trace(text: str, max_lines: int = 12) -> Tuple[List[Dict[str, Any]], str, Status]:
    """
    Returns: (trace_objs, final_answer, status)
    - trace_objs excludes the final line (if present)
    - final_answer is extracted from {"final": "..."} line
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        too_long_flag = True
    else:
        too_long_flag = False

    objs: List[Dict[str, Any]] = []
    for l in lines:
        try:
            objs.append(json.loads(l))
        except Exception:
            return [], "", "json_parse_error"

    final_answer = ""
    trace: List[Dict[str, Any]] = []

    for obj in objs:
        if isinstance(obj, dict) and "final" in obj:
            final_answer = str(obj["final"])
            break
        trace.append(obj)

    if not final_answer:
        return [], "", "missing_final"

    # validate trace
    for obj in trace:
        if not isinstance(obj, dict):
            return [], "", "invalid_schema"

        tool = obj.get("tool")
        action = obj.get("action")

        if tool not in VALID_TOOLS:
            return [], "", "invalid_schema"

        if tool == "stm" and action not in VALID_STM_ACTIONS:
            return [], "", "invalid_schema"

        if tool == "ltm" and action not in VALID_LTM_ACTIONS:
            return [], "", "invalid_schema"

    if too_long_flag:
        return trace, final_answer, "too_long"

    return trace, final_answer, "ok"
