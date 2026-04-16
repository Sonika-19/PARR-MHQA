from typing import List

from retrieval.document_store import RetrievalResult
from retrieval.embedding_engine import EmbeddingEngine


class CrossEncoderReranker:
    """Compatibility reranker that preserves retrieval order without model loading."""

    def __init__(self, model_name: str = "", diversity_weight: float = 0.0):
        _ = model_name
        _ = diversity_weight

    def score_single(self, query: str, text: str) -> float:
        _ = query
        _ = text
        return 0.0

    def compute_diversity_scores(self, results: List[RetrievalResult], embedding_engine: EmbeddingEngine):
        _ = embedding_engine
        return [0.0 for _ in results]

    def rerank_with_diversity(
        self,
        query: str,
        results: List[RetrievalResult],
        embedding_engine: EmbeddingEngine,
        alpha: float = 0.7,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        _ = query
        _ = embedding_engine
        _ = alpha
        if top_k <= 0:
            return []
        selected = list(results)[: min(top_k, len(results))]
        for idx, item in enumerate(selected):
            item.rank = idx
        return selected

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
        embedding_engine: EmbeddingEngine,
    ) -> List[RetrievalResult]:
        return self.rerank_with_diversity(
            query=query,
            results=results,
            embedding_engine=embedding_engine,
            alpha=1.0,
            top_k=top_k,
        )
