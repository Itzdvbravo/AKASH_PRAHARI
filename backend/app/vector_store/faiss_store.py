"""FAISS vector store implementation with robust pure-NumPy fallback."""
from pathlib import Path
import threading
from typing import List, Tuple
import numpy as np
from .store_interface import VectorStore

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FaissVectorStore(VectorStore):
    """
    Offline vector index wrapper supporting cosine inner-product search.
    Automatically uses FAISS if available; provides exact numpy cosine fallback.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._lock = threading.RLock()
        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._faiss_index = None

        if HAS_FAISS:
            try:
                self._faiss_index = faiss.IndexFlatIP(dim)
            except Exception:
                self._faiss_index = None

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._ids)

    def add(self, embeddings: np.ndarray, ids: List[str]) -> None:
        with self._lock:
            # Ensure float32 2D array
            vecs = np.atleast_2d(embeddings).astype(np.float32)
            # Ensure normalized
            norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms

            if self._faiss_index is not None:
                self._faiss_index.add(vecs)

            if self._vectors is None:
                self._vectors = vecs
            else:
                self._vectors = np.vstack([self._vectors, vecs])

            self._ids.extend(ids)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        with self._lock:
            if not self._ids or self._vectors is None:
                return []

            q = query_embedding.astype(np.float32).reshape(1, -1)
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm

            k_actual = min(k, len(self._ids))

            if self._faiss_index is not None and self._faiss_index.ntotal == len(self._ids):
                distances, indices = self._faiss_index.search(q, k_actual)
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if 0 <= idx < len(self._ids):
                        results.append((self._ids[idx], float(dist)))
                return results

            # Exact NumPy cosine similarity
            sims = np.dot(self._vectors, q.T).flatten()
            top_indices = np.argsort(-sims)[:k_actual]
            return [(self._ids[i], float(sims[i])) for i in top_indices]

    def save(self, filepath: str) -> None:
        with self._lock:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(path),
                vectors=self._vectors if self._vectors is not None else np.empty((0, self.dim)),
                ids=np.array(self._ids)
            )

    def load(self, filepath: str) -> None:
        with self._lock:
            path = Path(filepath)
            if not path.exists():
                # Check for .npz variant
                if Path(str(filepath) + ".npz").exists():
                    path = Path(str(filepath) + ".npz")
                else:
                    return

            try:
                data = np.load(str(path), allow_pickle=True)
                vecs = data["vectors"]
                ids = list(data["ids"])
                self._ids = []
                self._vectors = None
                if HAS_FAISS and self._faiss_index is not None:
                    self._faiss_index.reset()
                if len(ids) > 0 and len(vecs) > 0:
                    self.add(vecs, ids)
            except Exception as e:
                pass
