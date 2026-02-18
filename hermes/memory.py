from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


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

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        scored = [
            (_token_overlap_score(query, item), item) for item in list(self._buffer)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored if score > 0][:k]

    def filter(self, keyword: str) -> List[str]:
        key = keyword.lower().strip()
        return [item for item in self._buffer if key in item.lower()]

    def summary(self) -> str:
        if not self._buffer:
            return "STM is empty."
        head = list(self._buffer)[-5:]
        return " | ".join(head)

    def as_list(self) -> List[str]:
        return list(self._buffer)


@dataclass
class LongTermMemory:
    """Simple LTM store with archive/retrieve/update/delete operations."""

    _store: Dict[str, str] = field(default_factory=dict)
    _counter: int = 0

    def archive(self, content: str) -> str:
        self._counter += 1
        key = f"ltm_{self._counter}"
        self._store[key] = content.strip()
        return f"Archived to LTM with key={key}"

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        scored = []
        for key, value in self._store.items():
            scored.append((_token_overlap_score(query, value), key, value))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"key": key, "content": value} for score, key, value in scored if score > 0
        ][:k]

    def update(self, key: str, content: str) -> str:
        if key not in self._store:
            return f"LTM update skipped; key not found: {key}"
        self._store[key] = content.strip()
        return f"Updated LTM key={key}"

    def delete(self, key: str) -> str:
        if key not in self._store:
            return f"LTM delete skipped; key not found: {key}"
        del self._store[key]
        return f"Deleted LTM key={key}"

    def snapshot(self) -> Dict[str, str]:
        return dict(self._store)
