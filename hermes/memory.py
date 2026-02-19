from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Any
from datetime import datetime


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _token_overlap_score(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q or not t:
        return 0.0
    return len(q & t) / max(len(q), 1)


@dataclass
class ShortTermMemory:
    """Bounded in-memory context buffer (retain/discard/retrieve/summary)."""

    capacity: int = 20
    _buffer: Deque[str] = field(default_factory=deque)

    def retain(self, content: str) -> str:
        if len(self._buffer) >= self.capacity:
            self._buffer.popleft()
        self._buffer.append(content.strip())
        return f"Retained in STM. Size={len(self._buffer)}/{self.capacity}"

    def discard(self, content: str) -> str:
        target = content.strip()
        for item in list(self._buffer):
            if item == target:
                self._buffer.remove(item)
                return f"Discarded from STM: {target}"
        return f"STM discard skipped; not found: {target}"

    def retrieve_memory(self, query: str, k: int = 3) -> List[str]:
        """ Retrieves relevant memories and adds them to current context. """
        scored = []
        buff_list = list(self._buffer)

        # Weighted formula for accounting recency and relevancy 
        for i, item in enumerate(buff_list):
            relevance = _token_overlap_score(query, item)
            recent = (i + 1) / len(buff_list)
            rank = (relevance * 0.8) + (recent * 0.2)
            scored.append((rank, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored if score > 0][:k]

    def filter_context(self, keyword: str) -> List[str]:
        """Filters out irrelevant or outdated content from the conversation context to improve task-solving efficiency."""
        key = keyword.lower().strip()
        return [item for item in self._buffer if key in item.lower()]

    def summary_context(self) -> str:
        if not self._buffer:
            return "STM is empty."
        head = list(self._buffer)[-5:]
        return " | ".join(head)



@dataclass
class LongTermMemory:
    """
    LTM store.

    Each memory item is a record:
      {
        "key": str,
        "content": str,
        "tags": List[str],
        "meta": Dict[str, Any],
        "created_at": str,
        "updated_at": str,
        "last_accessed": str,
        "access_count": int,
        "importance": float
      }
    """

    _store: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _counter: int = 0

    def add(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        importance: float = 0.0,
    ) -> Dict[str, Any]:
        """add: create a new memory record."""
        self._counter += 1
        key = f"ltm_{self._counter}"
        ts = _now_iso()
        rec = {
            "key": key,
            "content": content.strip(),
            "tags": tags or [],
            "meta": meta or {},
            "created_at": ts,
            "updated_at": ts,
            "last_accessed": ts,
            "access_count": 0,
            "importance": float(importance),
        }
        self._store[key] = rec
        return {"ok": True, "op": "add", "key": key}

    def retrieve(
        self,
        query: str,
        k: int = 5,
        tags_any: Optional[List[str]] = None,
        meta_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """RETRIEVE: return top-k matches (simple token overlap + filters)."""
        q = query.strip()
        hits: List[Dict[str, Any]] = []

        for rec in self._store.values():
            if tags_any and not any(t in rec["tags"] for t in tags_any):
                continue
            if meta_filter:
                ok = True
                for mk, mv in meta_filter.items():
                    if rec["meta"].get(mk) != mv:
                        ok = False
                        break
                if not ok:
                    continue

            score = _token_overlap_score(q, rec["content"])
            if score > 0:
                hits.append({"score": score, "key": rec["key"], "content": rec["content"], "tags": rec["tags"], "meta": rec["meta"]})

        hits.sort(key=lambda x: x["score"], reverse=True)
        total = len(hits)
        hits = hits[:k]
        

        # update access stats
        ts = _now_iso()
        for h in hits:
            rec = self._store[h["key"]]
            rec["last_accessed"] = ts
            rec["access_count"] += 1

        return {"ok": True, "op": "retrieve", "query": q, "hits": hits, "total": total}

    def update(
        self,
        key: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """UPDATE: modify an existing record."""
        if key not in self._store:
            return {"ok": False, "op": "update", "error": f"key not found: {key}"}

        rec = self._store[key]
        if content is not None:
            rec["content"] = content.strip()
        if tags is not None:
            rec["tags"] = tags
        if meta is not None:
            rec["meta"] = meta
        if importance is not None:
            rec["importance"] = float(importance)

        rec["updated_at"] = _now_iso()
        return {"ok": True, "op": "update", "key": key}

    def delete(self, key: str) -> Dict[str, Any]:
        """DELETE: remove a record."""
        if key not in self._store:
            return {"ok": False, "op": "delete", "error": f"key not found: {key}"}
        del self._store[key]
        return {"ok": True, "op": "delete", "key": key}
    
    def get(self, key: str) -> Dict[str, Any]:
        """Fetch a full record by key."""
        rec = self._store.get(key)
        if not rec:
            return {"ok": False, "op": "get", "error": f"key not found: {key}"}
        return {"ok": True, "op": "get", "record": dict(rec)}
    
    def list(self) -> Dict[str, Any]:
        """List all keys + brief content preview."""
        out = [{"key": k, "preview": v["content"][:80], "tags": v["tags"]} for k, v in self._store.items()]
        return {"ok": True, "op": "list", "items": out}
    
    def clear(self) -> Dict[str, Any]:
        self._store.clear()
        self._counter = 0
        return {"ok": True, "op": "clear"}

    
    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Raw snapshot (useful for saving)."""
        return {k: dict(v) for k, v in self._store.items()}