import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import spacy

from config import Config
from generation.learned_hallucination_model import LearnedHallucinationDetector
from retrieval.embedding_engine import EmbeddingEngine


@dataclass
class HallucinationResult:
    score: float
    semantic_sim: float
    unsupported_ratio: float
    entity_mismatch: float
    lexical_overlap: float
    claims: List[str]
    is_hallucinated: bool
    breakdown: dict
    grounded_prob: float = 0.0
    token_support: Optional[List[float]] = None


class HallucinationDetector:
    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
    CLAIM_SPLIT_PATTERN = re.compile(r"[.!?]\s+")
    ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART"}
    FUSION_W_LEARNED = 0.6
    FUSION_W_SIM = 0.2
    FUSION_W_OVERLAP = 0.2

    def __init__(self, embedding_engine: EmbeddingEngine, config: Config):
        self.embedding_engine = embedding_engine
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.hallucination_threshold = float(config.HALLUCINATION_THRESHOLD)
        self.use_multi_signal = True

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = spacy.blank("en")

        self.learned_detector = LearnedHallucinationDetector(config, train_from_scratch=False)
        if Path(config.LEARNED_HALLUCINATION_BEST_PATH).exists():
            self.learned_detector.load(config.LEARNED_HALLUCINATION_BEST_PATH)
        elif Path(config.LEARNED_HALLUCINATION_MODEL_PATH).exists():
            self.learned_detector.load(config.LEARNED_HALLUCINATION_MODEL_PATH)

        if Path(config.LEARNED_HALLUCINATION_CALIBRATION_PATH).exists():
            self.learned_detector.load_calibration(config.LEARNED_HALLUCINATION_CALIBRATION_PATH)

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        return vectors / norms

    def compute_similarity(self, answer: str, context_texts: List[str]) -> float:
        if not context_texts:
            return 0.0

        answer_emb = self.embedding_engine.embed_texts([answer]).astype(np.float32)
        ctx_emb = self.embedding_engine.embed_texts(context_texts).astype(np.float32)

        answer_emb = self._normalize_rows(answer_emb)
        ctx_emb = self._normalize_rows(ctx_emb)
        sims = np.dot(ctx_emb, answer_emb[0])
        return float(np.max(sims))

    def compute_overlap(self, answer: str, context_texts: List[str]) -> float:
        stopwords = self.nlp.Defaults.stop_words
        answer_tokens = {
            tok.lower()
            for tok in self.TOKEN_PATTERN.findall(answer or "")
            if tok.lower() not in stopwords
        }
        context_tokens = {
            tok.lower()
            for tok in self.TOKEN_PATTERN.findall(" ".join(context_texts))
            if tok.lower() not in stopwords
        }

        if not answer_tokens:
            return 0.0

        union = answer_tokens | context_tokens
        if not union:
            return 0.0

        return float(len(answer_tokens & context_tokens) / len(union))

    def split_into_claims(self, answer: str) -> List[str]:
        parts = [part.strip() for part in self.CLAIM_SPLIT_PATTERN.split(answer or "")]
        return [claim for claim in parts if len(claim.split()) >= 5]

    def extract_entities_from_text(self, text: str) -> Set[str]:
        doc = self.nlp(text or "")
        return {
            ent.text.strip().lower()
            for ent in doc.ents
            if ent.label_ in self.ENTITY_LABELS and ent.text.strip()
        }

    def _unpack_args(self, *args):
        if len(args) == 2:
            query = ""
            answer, context_texts = args
            return query, str(answer), list(context_texts)

        if len(args) == 3:
            first, second, third = args
            if isinstance(second, (list, tuple)):
                return str(third), str(first), list(second)
            return str(first), str(second), list(third)

        raise ValueError("compute_hallucination_score expects (answer, context_texts) or (query, answer, context_texts)")

    def compute_hallucination_score(self, *args) -> HallucinationResult:
        query, answer, context_texts = self._unpack_args(*args)

        pred = self.learned_detector.predict(query, answer, context_texts)
        p_learned = float(pred.get("hallucinated_prob", 0.0))
        semantic_sim = float(self.compute_similarity(answer, context_texts))
        overlap_score = float(self.compute_overlap(answer, context_texts))
        unsupported_ratio = float(pred.get("unsupported_ratio", 0.0))
        entity_mismatch = float(pred.get("entity_mismatch", 0.0))
        token_support = pred.get("token_support", [])
        claims = self.split_into_claims(answer)

        if self.use_multi_signal:
            final_score = float(
                self.FUSION_W_LEARNED * p_learned
                + self.FUSION_W_SIM * (1.0 - semantic_sim)
                + self.FUSION_W_OVERLAP * (1.0 - overlap_score)
            )
        else:
            final_score = float(p_learned)

        threshold = float(self.hallucination_threshold)
        result = HallucinationResult(
            score=final_score,
            semantic_sim=semantic_sim,
            unsupported_ratio=unsupported_ratio,
            entity_mismatch=entity_mismatch,
            lexical_overlap=float(1.0 - overlap_score),
            claims=claims,
            is_hallucinated=bool(final_score > threshold),
            breakdown={
                "backend": "learned",
                "threshold": threshold,
                "final_score": final_score,
                "p_learned": p_learned,
                "similarity": semantic_sim,
                "overlap": overlap_score,
                "multi_signal": bool(self.use_multi_signal),
            },
            grounded_prob=float(pred.get("grounded_prob", 1.0 - final_score)),
            token_support=[float(v) for v in token_support],
        )
        self.logger.info("hallucination_breakdown | %s", result.breakdown)
        return result


if __name__ == "__main__":
    cfg = Config()
    embed = EmbeddingEngine(model_name=cfg.EMBEDDING_MODEL, cache_dir=cfg.CACHE_DIR)
    detector = HallucinationDetector(embedding_engine=embed, config=cfg)
    print("HallucinationDetector initialized.")
