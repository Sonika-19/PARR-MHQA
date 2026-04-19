import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config import Config
from generation.hallucination_detector import HallucinationDetector
from generation.prompt_engine import PromptEngine
from generation.self_critique import CritiqueResult, SelfCritiqueModule
from policy.policy_network import PolicyDecision
from retrieval.document_store import RetrievalResult
from retrieval.retrieval_system import AdaptiveRetriever


@dataclass
class PipelineResult:
    query: str
    final_answer: str
    all_answers: List[str]
    hallucination_scores: List[float]
    final_hallucination_score: float
    iterations_used: int
    retrieved_docs: List[RetrievalResult]
    policy_decision: PolicyDecision
    critique_result: CritiqueResult
    execution_time_total: float
    time_retrieval: float
    time_reranking: float
    time_generation: float
    time_hallucination: float
    num_llm_calls: int
    csr_score: float = 0.0
    reasoning_consistency: float = 0.0
    reasoning_trace: Optional[List[str]] = None


class RefinementController:
    REFINE_ACCEPT_MARGIN = 0.15
    EMPTY_ANSWER_FALLBACK = "No answer generated from model output."

    def __init__(
        self,
        retriever: AdaptiveRetriever,
        generator,
        hallucination_detector: HallucinationDetector,
        critique_module: SelfCritiqueModule,
        config: Config,
        prompt_engine: PromptEngine,
    ):
        self.retriever = retriever
        self.generator = generator
        self.hallucination_detector = hallucination_detector
        self.critique_module = critique_module
        self.config = config
        self.prompt_engine = prompt_engine
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in re.findall(r"[A-Za-z0-9]+", text.lower()) if tok]

    def _entity_match_score(self, answer: str, docs: List[RetrievalResult]) -> float:
        answer_entities = self.hallucination_detector.extract_entities_from_text(answer)
        context_entities = self.hallucination_detector.extract_entities_from_text(" ".join(doc.text for doc in docs))

        if not answer_entities:
            return 0.5
        return float(len(answer_entities & context_entities) / len(answer_entities))

    def _length_score(self, answer: str) -> float:
        token_count = len(self._tokenize(answer))
        if token_count >= 3:
            return 1.0
        return float(np.clip(token_count / 3.0, 0.0, 1.0))

    def _answer_quality(self, answer: str, docs: List[RetrievalResult]) -> float:
        answer_tokens = set(self._tokenize(answer))
        if not answer_tokens:
            return 0.0

        context_tokens = set(self._tokenize(" ".join(doc.text for doc in docs)))
        if not context_tokens:
            return 0.0

        overlap = len(answer_tokens & context_tokens)
        overlap_score = float(overlap / max(len(answer_tokens), 1))
        entity_match_score = self._entity_match_score(answer, docs)
        length_score = self._length_score(answer)
        quality = 0.5 * overlap_score + 0.3 * entity_match_score + 0.2 * length_score
        return float(np.clip(quality, 0.0, 1.0))

    def _is_weaker_refinement(self, original_answer: str, refined_answer: str) -> bool:
        if len((refined_answer or "").strip()) < len((original_answer or "").strip()):
            return True

        original_entities = self.hallucination_detector.extract_entities_from_text(original_answer)
        refined_entities = self.hallucination_detector.extract_entities_from_text(refined_answer)
        return len(refined_entities) < len(original_entities)

    def _build_prompt(self, query: str, docs: List[RetrievalResult], prompt_style: str) -> str:
        context_texts = self.prompt_engine.format_context(docs)
        if prompt_style == "chain_of_thought":
            return self.prompt_engine.build_chain_of_thought_prompt(query, context_texts)
        return self.prompt_engine.build_direct_prompt(query, context_texts)

    @classmethod
    def _safe_answer_text(cls, extracted_answer: str, raw_text: str) -> str:
        answer = (extracted_answer or "").strip()
        if answer:
            return answer

        raw = (raw_text or "").strip()
        if not raw:
            return cls.EMPTY_ANSWER_FALLBACK

        # If the model only echoes the prompt marker, salvage any residual content.
        raw = re.sub(r"(?i)^\s*answer\s*:\s*", "", raw).strip()
        raw = re.sub(r"[\s\.,;:]+$", "", raw)
        return raw or cls.EMPTY_ANSWER_FALLBACK

    def _score_answer(self, query: str, answer: str, docs: List[RetrievalResult]):
        context_texts = self.prompt_engine.format_context(docs)
        h_result = self.hallucination_detector.compute_hallucination_score(answer, context_texts, query)
        c_result = self.critique_module.critique(query, answer, context_texts)
        return h_result, c_result

    def run_pipeline(self, query, policy_decision) -> PipelineResult:
        t0_total = time.perf_counter()
        time_retrieval = 0.0
        time_generation = 0.0
        time_hallucination = 0.0

        prompt_style = policy_decision.prompt_style

        t0 = time.perf_counter()
        docs = self.retriever.adaptive_retrieve(query, policy_decision)
        time_retrieval += time.perf_counter() - t0

        # Step 1: initial answer (always generated).
        t0 = time.perf_counter()
        prompt = self._build_prompt(query, docs, prompt_style)
        raw_answer = self.generator.generate_answer(prompt, max_tokens=500)
        original_answer = self._safe_answer_text(
            self.generator.extract_final_answer(raw_answer),
            raw_answer,
        )
        time_generation += time.perf_counter() - t0

        t0 = time.perf_counter()
        original_h_result, original_c_result = self._score_answer(query, original_answer, docs)
        time_hallucination += time.perf_counter() - t0

        # Step 2: one refinement attempt (always generated, no conditions).
        t0 = time.perf_counter()
        context_texts = self.prompt_engine.format_context(docs)
        refinement_prompt = self.prompt_engine.build_refinement_prompt(
            query,
            original_answer,
            original_c_result.critique_text,
            context_texts,
        )
        refined_raw_answer = self.generator.generate_refinement(refinement_prompt, max_tokens=500)
        refined_answer = self._safe_answer_text(
            self.generator.extract_final_answer(refined_raw_answer),
            refined_raw_answer,
        )
        time_generation += time.perf_counter() - t0

        t0 = time.perf_counter()
        refined_h_result, refined_c_result = self._score_answer(query, refined_answer, docs)
        time_hallucination += time.perf_counter() - t0

        # Step 3: strict acceptance rule to protect already-good originals.
        q_original = self._answer_quality(original_answer, docs)
        q_refined = self._answer_quality(refined_answer, docs)
        accept_refined = (
            q_refined >= (q_original + self.REFINE_ACCEPT_MARGIN)
            and not self._is_weaker_refinement(original_answer, refined_answer)
        )

        all_answers = [original_answer, refined_answer]
        h_scores = [original_h_result.score, refined_h_result.score]
        critique_results = [original_c_result, refined_c_result]
        final_answer = refined_answer if accept_refined else original_answer
        final_hallucination_score = refined_h_result.score if accept_refined else original_h_result.score
        final_critique = refined_c_result if accept_refined else original_c_result
        total_time = time.perf_counter() - t0_total

        return PipelineResult(
            query=query,
            final_answer=final_answer,
            all_answers=all_answers,
            hallucination_scores=h_scores,
            final_hallucination_score=float(final_hallucination_score),
            iterations_used=1,
            retrieved_docs=docs,
            policy_decision=policy_decision,
            critique_result=final_critique,
            execution_time_total=total_time,
            time_retrieval=time_retrieval,
            time_reranking=0.0,
            time_generation=time_generation,
            time_hallucination=time_hallucination,
            num_llm_calls=2,
            csr_score=0.0,
            reasoning_consistency=0.0,
            reasoning_trace=[],
        )
