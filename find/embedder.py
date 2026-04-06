"""Embedder — wraps sentence-transformers with GPU support and progress."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .config import EMBED_MODEL, BATCH_SIZE, MODEL_CACHE

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder:
    """Lazy-loaded embedding model. Uses GPU if available."""

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        self._model_name = model_name
        self._model: "SentenceTransformer | None" = None

    def _load(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(
                self._model_name,
                cache_folder=str(MODEL_CACHE),
                device=device,
            )
        return self._model

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Return float32 array of shape (len(texts), dim)."""
        model = self._load()
        return model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=show_progress,
            normalize_embeddings=False,
            convert_to_numpy=True,
        ).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string, return shape (dim,)."""
        return self.embed([text])[0]
