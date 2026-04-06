from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import chromadb

from hermes.embeddings import EmbeddingProvider, SentenceTransformerEmbedding


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _token_overlap_score(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q or not t:
        return 0.0
    return len(q & t) / max(len(q), 1)


def _default_stm_collection():
    """
    Create an in-memory Chroma collection for ShortTermMemory.

    Each STM instance gets its own ephemeral collection, scoped to the current
    process and not persisted to disk.
    """
    if chromadb is None:
        raise RuntimeError(
            "chromadb is required for ShortTermMemory vector storage. "
            "Install it or provide a custom collection."
        )
    client = chromadb.Client()
    name = f"stm_{uuid4().hex}"
    return client.create_collection(name=name)


@dataclass
class ShortTermMemory:
    """STM tools for managing short term memory."""

    capacity: int = 20
    embedder: Optional[EmbeddingProvider] = None
    collection: Optional[Any] = field(default=None, repr=False)
    # Local list of items to preserve ordering semantics and helper behaviors.
    # Each entry is a dict with keys: "id", "content", "index".
    _items: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _next_index: int = field(default=0, repr=False)

    def _ensure_embedder(self) -> EmbeddingProvider:
        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedding()
        return self.embedder

    def _ensure_collection(self) -> Any:
        if self.collection is None:
            self.collection = _default_stm_collection()
        return self.collection

    def retain(self, content: str) -> str:
        text = content.strip()

        # Lazily initialize vector components when we first retain anything.
        collection = self._ensure_collection()
        embedder = self._ensure_embedder()

        self._next_index += 1
        doc_id = f"stm_{self._next_index}"

        embedding = embedder.embed([text])[0]
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"index": self._next_index}],
        )

        self._items.append({"id": doc_id, "content": text, "index": self._next_index})

        # Enforce bounded capacity by evicting the oldest items first.
        while len(self._items) > self.capacity:
            oldest = self._items.pop(0)
            try:
                collection.delete(ids=[oldest["id"]])
            except Exception:
                # Best-effort cleanup; STM semantics should not depend on delete success.
                pass

        return f"Retained in STM. Size={len(self._items)}/{self.capacity}"

    def discard(self, content: str) -> str:
        target = content.strip()
        collection = self.collection

        for idx, item in enumerate(list(self._items)):
            if item["content"] == target:
                removed = self._items.pop(idx)
                if collection is not None:
                    try:
                        collection.delete(ids=[removed["id"]])
                    except Exception:
                        pass
                return f"Discarded from STM: {target}"
        return f"STM discard skipped; not found: {target}"

    def retrieve_memory(self, query: str, k: int = 3) -> List[str]:
        """ Retrieves relevant memories and adds them to current context. """
        if not self._items:
            return []

        collection = self._ensure_collection()
        embedder = self._ensure_embedder()

        query_emb = embedder.embed([query])[0]
        # Use Chroma similarity search; we request up to k results but not more
        # than we actually have stored.
        n_results = min(k, len(self._items))
        raw = collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        docs = raw.get("documents") or [[]]
        dists = raw.get("distances") or [[]]
        metas = raw.get("metadatas") or [[]]

        # Chroma returns a list per query; we only passed a single query.
        docs_q = docs[0] if docs else []
        dists_q = dists[0] if dists else []
        metas_q = metas[0] if metas else []

        scored: List[tuple[float, str]] = []
        for doc, dist, meta in zip(docs_q, dists_q, metas_q):
            # Convert distance to a similarity-like value in (0, 1].
            similarity = 1.0 / (1.0 + float(dist))

            # Small recency bias using the insertion index stored in metadata.
            index = float(meta.get("index", 0.0)) if isinstance(meta, dict) else 0.0
            recency = index / float(self._next_index or 1)

            rank = (similarity * 0.8) + (recency * 0.2)
            scored.append((rank, str(doc)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored][:k]

    def filter_context(self, keyword: str) -> List[str]:
        """Filters out irrelevant or outdated content from the conversation context to improve task-solving efficiency."""
        key = keyword.lower().strip()
        return [item["content"] for item in self._items if key in item["content"].lower()]

    def summary_context(self) -> str:
        if not self._items:
            return "STM is empty."
        head = [item["content"] for item in self._items[-5:]]
        return " | ".join(head)

    @property
    def _buffer(self) -> List[str]:
        """Alias returning item text list for reward computation."""
        return [item["content"] for item in self._items]

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Public alias for retrieve_memory (used by agent.py stm_tool)."""
        return self.retrieve_memory(query, k=k)

    def filter(self, keyword: str) -> List[str]:
        """Public alias for filter_context (used by agent.py stm_tool)."""
        return self.filter_context(keyword)

    def summary(self) -> str:
        """Public alias for summary_context (used by agent.py stm_tool)."""
        return self.summary_context()



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