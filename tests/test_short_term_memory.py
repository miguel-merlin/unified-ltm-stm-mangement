import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes.embeddings import EmbeddingProvider
from hermes.memory import ShortTermMemory


class DummyEmbeddingProvider(EmbeddingProvider):
    """
    Simple, deterministic embedder for tests.

    It maps each text to a 1D vector whose single component is the length of the
    text, so that similarity ordering is easy to assert.
    """

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t))] for t in texts]


class FakeCollection:
    """
    Lightweight stand-in for a Chroma collection used in tests.

    Stores embeddings in-memory and performs a basic L2-distance similarity
    search to exercise ShortTermMemory's vector-based code paths.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        for i, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self._store[i] = {"embedding": emb, "document": doc, "metadata": meta}

    def delete(self, ids: List[str]) -> None:
        for i in ids:
            self._store.pop(i, None)

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int,
        include: List[str],
    ) -> Dict[str, Any]:
        q_emb = query_embeddings[0]

        def l2(a: List[float], b: List[float]) -> float:
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

        scored = []
        for rec in self._store.values():
            d = l2(q_emb, rec["embedding"])
            scored.append((d, rec))

        scored.sort(key=lambda x: x[0])
        top = scored[:n_results]

        documents: List[str] = []
        distances: List[float] = []
        metadatas: List[Dict[str, Any]] = []

        for dist, rec in top:
            documents.append(rec["document"])
            distances.append(dist)
            metadatas.append(rec["metadata"])

        out: Dict[str, Any] = {}
        if "documents" in include:
            out["documents"] = [documents]
        if "distances" in include:
            out["distances"] = [distances]
        if "metadatas" in include:
            out["metadatas"] = [metadatas]
        if "ids" in include:
            # IDs are not currently required by ShortTermMemory, so we can omit.
            out["ids"] = [[]]
        return out


def make_stm(capacity: int = 3) -> ShortTermMemory:
    return ShortTermMemory(
        capacity=capacity,
        embedder=DummyEmbeddingProvider(),
        collection=FakeCollection(),
    )


def test_retain_respects_capacity_and_status_message() -> None:
    stm = make_stm(capacity=2)

    msg1 = stm.retain("first")
    msg2 = stm.retain("second")
    msg3 = stm.retain("third")  # should evict "first"

    assert "Size=2/2" in msg2
    assert "Size=2/2" in msg3

    # Only the most recent two items should remain, ordered by recency.
    assert stm.summary_context() == "second | third"


def test_discard_removes_exact_match_only() -> None:
    stm = make_stm()
    stm.retain("keep this")
    stm.retain("delete me")

    msg = stm.discard("delete me")
    assert "Discarded from STM" in msg
    assert "delete me" not in stm.summary_context()

    # Non-existent content should be reported as skipped.
    msg2 = stm.discard("not present")
    assert "STM discard skipped; not found" in msg2


def test_retrieve_memory_uses_vector_similarity() -> None:
    stm = make_stm()
    stm.retain("short")
    stm.retain("a bit longer text")
    stm.retain("this is the longest piece of content here")

    # With the DummyEmbeddingProvider, longer texts are "closer" to queries
    # whose length is also large, so we expect the longest content to be ranked first.
    results = stm.retrieve_memory("query that is also quite long", k=2)
    assert len(results) == 2
    assert results[0] == "this is the longest piece of content here"


def test_filter_context_and_summary_context_compatible_behavior() -> None:
    stm = make_stm(capacity=10)
    stm.retain("alpha event")
    stm.retain("beta event")
    stm.retain("another alpha thing")
    stm.retain("gamma detail")

    filtered = stm.filter_context("alpha")
    assert filtered == ["alpha event", "another alpha thing"]

    summary = stm.summary_context()
    # Last 5 items joined by " | ", preserving insertion order
    assert summary == "alpha event | beta event | another alpha thing | gamma detail"


def test_summary_context_empty_message() -> None:
    stm = make_stm()
    assert stm.summary_context() == "STM is empty."


@pytest.mark.integration
def test_stm_with_real_chroma() -> None:
    """Lightweight integration test that hits a real Chroma client."""
    try:
        import chromadb  # type: ignore[import]
    except ImportError:
        pytest.skip("chromadb not installed; skipping Chroma integration test")

    client = chromadb.Client()
    collection = client.create_collection(name="stm_test_integration")

    stm = ShortTermMemory(
        capacity=3,
        collection=collection,
        embedder=DummyEmbeddingProvider(),
    )

    stm.retain("the quick brown fox")
    stm.retain("jumps over the lazy dog")

    results = stm.retrieve_memory("fox", k=1)
    assert results
    assert "fox" in results[0]

