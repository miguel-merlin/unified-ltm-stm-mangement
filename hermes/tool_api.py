from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple
from hermes.memory import LongTermMemory, ShortTermMemory

class HermesToolAPI:
    """
    Deterministic wrapper around hermes/memory.py with a unified effect:
    LTM retrieve -> inject hits into STM.
    """

    def __init__(self, stm: ShortTermMemory, ltm: LongTermMemory):
        self.stm = stm
        self.ltm = ltm

    def stm_tool(
        self,
        action: Literal["retain", "discard", "retrieve", "summary", "filter"],
        content: str = "",
        k: int = 5,
    ) -> Any:
        if action == "retain":
            return self.stm.retain(content)
        if action == "discard":
            return self.stm.discard(content)
        if action == "retrieve":
            return self.stm.retrieve_memory(content, k=k)
        if action == "summary":
            return self.stm.summary_context()
        if action == "filter":
            return self.stm.filter_context(content)
        raise ValueError(f"Unsupported STM action: {action}")

    def ltm_tool(
        self,
        action: Literal["add", "retrieve", "update", "delete", "get", "list", "clear"],
        content: str = "",
        key: str = "",
        k: int = 5,
        tags: Optional[List[str]] = None,
        importance: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
        tags_any: Optional[List[str]] = None,
        meta_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if action == "add":
            return self.ltm.add(content=content, tags=tags, meta=meta, importance=importance)
        if action == "retrieve":
            return self.ltm.retrieve(query=content, k=k, tags_any=tags_any, meta_filter=meta_filter)
        if action == "update":
            return self.ltm.update(key=key, content=content, tags=tags, meta=meta, importance=importance)
        if action == "delete":
            return self.ltm.delete(key=key)
        if action == "get":
            return self.ltm.get(key=key)
        if action == "list":
            return self.ltm.list()
        if action == "clear":
            return self.ltm.clear()
        raise ValueError(f"Unsupported LTM action: {action}")

    def ltm_retrieve_to_stm(
        self,
        query: str,
        k: int = 5,
        prefix: str = "[LTM]",
        tags_any: Optional[List[str]] = None,
        meta_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Retrieve from LTM and inject hits into STM so they influence later steps.
        """
        out = self.ltm_tool("retrieve", content=query, k=k, tags_any=tags_any, meta_filter=meta_filter)
        hits = out.get("hits", []) if out.get("ok") else []
        for h in hits:
            self.stm.retain(f"{prefix} {h['key']}: {h['content']}")
        return out, len(hits)
