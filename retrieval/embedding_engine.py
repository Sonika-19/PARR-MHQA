import os
import pickle
from typing import Dict, List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    def __init__(self, model_name: str, cache_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._query_cache: Dict[str, np.ndarray] = {}

    def embed_documents(
        self,
        chunks: List[dict],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        embeddings_path = os.path.join(self.cache_dir, "doc_embeddings.npy")
        if os.path.exists(embeddings_path):
            cached_embeddings = self.load_embeddings(embeddings_path)
            if cached_embeddings.ndim == 2 and cached_embeddings.shape[0] == len(chunks):
                if cached_embeddings.dtype != np.float32:
                    cached_embeddings = cached_embeddings.astype(np.float32)
                faiss.normalize_L2(cached_embeddings)
                return cached_embeddings

        texts = [str(chunk.get("text", "")) for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.save_embeddings(embeddings, embeddings_path)
        return embeddings

    def embed_query(self, query_text: str) -> np.ndarray:
        key = query_text.strip().lower()
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached

        embedding = self.model.encode(
            [query_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embedding = np.asarray(embedding, dtype=np.float32)
        faiss.normalize_L2(embedding)
        self._query_cache[key] = embedding
        return embedding

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def save_embeddings(self, embeddings: np.ndarray, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, embeddings)

    def load_embeddings(self, path: str) -> np.ndarray:
        return np.load(path)

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.dtype != np.float32:
            raise ValueError("embeddings must be float32")

        vectors = embeddings.copy()

        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def save_index(self, index: faiss.Index, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(index, path)

    def load_index(self, path: str) -> faiss.Index:
        return faiss.read_index(path)


if __name__ == "__main__":
    chunks_path = "embeddings/processed_chunks.pkl"
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            "Missing chunks file at embeddings/processed_chunks.pkl. Run data/dataset_loader.py first."
        )

    with open(chunks_path, "rb") as file:
        chunks = pickle.load(file)

    engine = EmbeddingEngine(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="embeddings/cache",
    )

    doc_embeddings = engine.embed_documents(chunks, batch_size=64, show_progress=True)
    index = engine.build_faiss_index(doc_embeddings)

    print(f"Embeddings shape: {doc_embeddings.shape}")
    print(f"Index total vectors (ntotal): {index.ntotal}")
