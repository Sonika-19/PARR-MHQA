import logging
from collections import Counter
from typing import List

import faiss
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from config import Config
from policy.policy_network import PolicyDecision
from retrieval.document_store import DocumentStore, RetrievalResult
from retrieval.embedding_engine import EmbeddingEngine


class AdaptiveRetriever:
    ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT"}

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        faiss_index: faiss.Index,
        document_store: DocumentStore,
        config: Config,
    ):
        self.embedding_engine = embedding_engine
        self.faiss_index = faiss_index
        self.document_store = document_store
        self.config = config

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = spacy.blank("en")
        self._tfidf_vectorizer = None
        self._tfidf_fitted = False

        self.logger = logging.getLogger(__name__)

    def retrieve(self, query: str, k: int) -> List[RetrievalResult]:
        query_emb = self.embedding_engine.embed_query(query).astype(np.float32).copy()
        faiss.normalize_L2(query_emb)

        scores, indices = self.faiss_index.search(query_emb, k)
        results: List[RetrievalResult] = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue

            chunk = self.document_store.get(int(idx))
            if chunk is None:
                continue

            results.append(
                RetrievalResult(
                    chunk_id=str(chunk.get("chunk_id", "")),
                    doc_id=str(chunk.get("doc_id", "")),
                    title=str(chunk.get("title", "")),
                    text=str(chunk.get("text", "")),
                    score=float(score),
                    rank=rank,
                    retrieval_stage="initial",
                )
            )

        top_log = [(result.title, round(result.score, 4)) for result in results[: min(5, len(results))]]
        self.logger.info("retrieve | query=%s | k=%s | top=%s", query, k, top_log)
        return results

    def extract_entities(self, results: List[RetrievalResult]) -> List[str]:
        if not results:
            self.logger.info("extract_entities | no results to process")
            return []

        texts = [result.text for result in results if result.text]
        combined_text = "\n".join(texts)

        doc = self.nlp(combined_text)

        ner_candidates = [
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in self.ENTITY_LABELS and ent.text.strip()
        ]

        noun_chunk_candidates = []
        if "parser" in self.nlp.pipe_names:
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip()
                if len(phrase.split()) >= 2:
                    noun_chunk_candidates.append(phrase)

        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)

        tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
        self._tfidf_fitted = True

        feature_names = self._tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        top_idx = np.argsort(tfidf_scores)[::-1]

        ner_set_lower = {candidate.lower() for candidate in ner_candidates}
        tfidf_keywords: List[str] = []
        for idx in top_idx:
            token = feature_names[idx].strip()
            if not token:
                continue
            if token.lower() in ner_set_lower:
                continue
            tfidf_keywords.append(token)
            if len(tfidf_keywords) >= 5:
                break

        all_candidates = ner_candidates + noun_chunk_candidates + tfidf_keywords

        deduped: List[str] = []
        seen = set()
        for candidate in all_candidates:
            key = candidate.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(candidate.strip())

        freq_counter = Counter()
        lower_texts = [text.lower() for text in texts]
        for candidate in deduped:
            candidate_lower = candidate.lower()
            freq_counter[candidate] = sum(1 for text in lower_texts if candidate_lower in text)

        ranked = sorted(deduped, key=lambda item: (-freq_counter[item], item.lower()))
        top_entities = ranked[:8]

        self.logger.info("extract_entities | entities=%s", top_entities)
        return top_entities

    def expand_retrieval(
        self,
        query: str,
        initial_results: List[RetrievalResult],
        k_expand: int = 3,
    ) -> List[RetrievalResult]:
        entities = self.extract_entities(initial_results)

        expanded_results: List[RetrievalResult] = []
        for entity in entities:
            entity_results = self.retrieve(entity, k_expand)
            for result in entity_results:
                expanded_results.append(
                    RetrievalResult(
                        chunk_id=result.chunk_id,
                        doc_id=result.doc_id,
                        title=result.title,
                        text=result.text,
                        score=result.score,
                        rank=result.rank,
                        retrieval_stage="expansion",
                    )
                )

        combined = list(initial_results)
        seen_chunk_ids = {result.chunk_id for result in initial_results}

        added = 0
        for result in expanded_results:
            if result.chunk_id in seen_chunk_ids:
                continue
            combined.append(result)
            seen_chunk_ids.add(result.chunk_id)
            added += 1

        self.logger.info(
            "expand_retrieval | query=%s | entities=%s | added_docs=%s",
            query,
            len(entities),
            added,
        )
        return combined

    def adaptive_retrieve(self, query: str, policy_decision: PolicyDecision) -> List[RetrievalResult]:
        results = self.retrieve(query, policy_decision.k)
        if policy_decision.expand_retrieval:
            results = self.expand_retrieval(query, results, policy_decision.k_expand)
        return results
