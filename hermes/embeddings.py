from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Small abstraction over embedding models used by memory components."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        ...


class SentenceTransformerEmbedding:
    """
    Default EmbeddingProvider backed by a local sentence-transformers model.
    """

    def __init__(self,model_name: str = "sentence-transformers/all-MiniLM-L6-v2",device: Optional[str] = None,) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - exercised only when missing
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerEmbeddingProvider. "
                "Install it or supply a custom EmbeddingProvider."
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # We return a plain list-of-lists of floats to keep the interface simple
        embeddings = self._model.encode(texts, convert_to_numpy=False)
        return [list(map(float, emb)) for emb in embeddings]

