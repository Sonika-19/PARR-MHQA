import json
import os
import time
from dataclasses import asdict
from typing import Dict, List

from config import Config
from evaluation.metrics import EvaluationReport, batch_evaluate, print_report
from generation.refinement_controller import PipelineResult
from generation.self_critique import CritiqueResult
from policy.policy_network import PolicyDecision
from retrieval.document_store import RetrievalResult


class LLMOnlyBaseline:
    def __init__(self, retriever, reranker, prompt_engine, llm_generator, hallucination_detector=None, k: int = 5):
        self.retriever = retriever
        self.prompt_engine = prompt_engine
        self.llm_generator = llm_generator
        self.hallucination_detector = hallucination_detector
        self.k = k

    def _retrieve_and_generate(self, query):
        docs = self.retriever.retrieve(query, self.k)

        context_texts = self.prompt_engine.format_context(docs)
        prompt = self.prompt_engine.build_direct_prompt(query, context_texts)
        raw = self.llm_generator.generate_answer(prompt, max_tokens=300)
        predicted = self.llm_generator.extract_final_answer(raw)
        return predicted, docs, context_texts

    def answer(self, query) -> str:
        predicted, _, _ = self._retrieve_and_generate(query)
        return predicted

    def _compute_hallucination_eval_score(self, query: str, predicted: str, context_texts: List[str]) -> float:
        if self.hallucination_detector is None:
            return 0.0

        hall = self.hallucination_detector.compute_hallucination_score(predicted, context_texts, query)
        return float(hall.score)

    def run_on_dataset(self, qa_pairs, n: int = 200):
        outputs = []
        for item in qa_pairs[:n]:
            question = str(item.get("question", ""))
            truth = str(item.get("answer", ""))

            t0 = time.perf_counter()
            predicted, docs, context_texts = self._retrieve_and_generate(question)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            hallucination_score = self._compute_hallucination_eval_score(question, predicted, context_texts)

            outputs.append(
                {
                    "question": question,
                    "predicted": predicted,
                    "truth": truth,
                    "latency_ms": latency_ms,
                    "num_llm_calls": 1,
                    "retrieved_docs": docs,
                    "hallucination_score": hallucination_score,
                }
            )

        return outputs


class StandardRAGBaseline:
    def __init__(self, retriever, reranker, prompt_engine, llm_generator, k: int = 5):
        self.retriever = retriever
        self.prompt_engine = prompt_engine
        self.llm_generator = llm_generator
        self.k = k

    def answer(self, query):
        t0 = time.perf_counter()

        docs = self.retriever.retrieve(query, self.k)

        context_texts = self.prompt_engine.format_context(docs)
        prompt = self.prompt_engine.build_direct_prompt(query, context_texts)
        raw = self.llm_generator.generate_answer(prompt, max_tokens=300)
        predicted = self.llm_generator.extract_final_answer(raw)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return predicted, docs, latency_ms

    def run_on_dataset(self, qa_pairs, n: int = 200):
        outputs = []
        for item in qa_pairs[:n]:
            question = str(item.get("question", ""))
            truth = str(item.get("answer", ""))

            predicted, docs, latency_ms = self.answer(question)

            outputs.append(
                {
                    "question": question,
                    "predicted": predicted,
                    "truth": truth,
                    "latency_ms": latency_ms,
                    "num_llm_calls": 1,
                    "retrieved_docs": docs,
                }
            )

        return outputs


class HeuristicBaseline:
    def __init__(self, retriever, reranker, prompt_engine, llm_generator, hallucination_detector, k: int = 5, threshold: float = 0.5):
        self._llm_only = LLMOnlyBaseline(
            retriever=retriever,
            reranker=reranker,
            prompt_engine=prompt_engine,
            llm_generator=llm_generator,
            k=k,
        )
        self.hallucination_detector = hallucination_detector
        self.threshold = float(threshold)

    def answer(self, query):
        t0 = time.perf_counter()

        predicted, docs, context_texts = self._llm_only._retrieve_and_generate(query)

        original_flag = bool(getattr(self.hallucination_detector, "use_multi_signal", True))
        self.hallucination_detector.use_multi_signal = False
        try:
            hall = self.hallucination_detector.compute_hallucination_score(predicted, context_texts, query)
        finally:
            self.hallucination_detector.use_multi_signal = original_flag
        hallucination_score = float(hall.score)
        is_hallucinated = hallucination_score > self.threshold

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return predicted, docs, latency_ms, hallucination_score, is_hallucinated

    def run_on_dataset(self, qa_pairs, n: int = 200):
        outputs = []
        for item in qa_pairs[:n]:
            question = str(item.get("question", ""))
            truth = str(item.get("answer", ""))

            predicted, docs, latency_ms, hallucination_score, is_hallucinated = self.answer(question)

            outputs.append(
                {
                    "question": question,
                    "predicted": predicted,
                    "truth": truth,
                    "latency_ms": latency_ms,
                    "num_llm_calls": 1,
                    "retrieved_docs": docs,
                    "hallucination_score": hallucination_score,
                    "is_hallucinated": bool(is_hallucinated),
                }
            )

        return outputs


class LearnedBaseline:
    def __init__(self, retriever, reranker, prompt_engine, llm_generator, hallucination_detector, k: int = 5, threshold: float = 0.5):
        self._llm_only = LLMOnlyBaseline(
            retriever=retriever,
            reranker=reranker,
            prompt_engine=prompt_engine,
            llm_generator=llm_generator,
            k=k,
        )
        self.hallucination_detector = hallucination_detector
        self.threshold = float(threshold)

    def answer(self, query):
        t0 = time.perf_counter()
        predicted, docs, context_texts = self._llm_only._retrieve_and_generate(query)

        original_flag = bool(getattr(self.hallucination_detector, "use_multi_signal", True))
        self.hallucination_detector.use_multi_signal = True
        try:
            hall = self.hallucination_detector.compute_hallucination_score(predicted, context_texts, query)
        finally:
            self.hallucination_detector.use_multi_signal = original_flag

        hallucination_score = float(hall.score)
        is_hallucinated = hallucination_score > self.threshold
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return predicted, docs, latency_ms, hallucination_score, is_hallucinated

    def run_on_dataset(self, qa_pairs, n: int = 200):
        outputs = []
        for item in qa_pairs[:n]:
            question = str(item.get("question", ""))
            truth = str(item.get("answer", ""))
            if not question.strip():
                continue

            predicted, docs, latency_ms, hallucination_score, is_hallucinated = self.answer(question)
            outputs.append(
                {
                    "question": question,
                    "predicted": predicted,
                    "truth": truth,
                    "latency_ms": latency_ms,
                    "num_llm_calls": 1,
                    "retrieved_docs": docs,
                    "hallucination_score": hallucination_score,
                    "is_hallucinated": bool(is_hallucinated),
                }
            )

        return outputs


class AblationStudy:
    def __init__(self, full_system):
        self.full_system = full_system
        self.config: Config = full_system.config

    def _policy_for_query(self, query: str) -> PolicyDecision:
        # Use the system's deterministic non-RL policy heuristic.
        return self.full_system._default_policy_decision(query)

    def _to_eval_pipeline_result(
        self,
        query: str,
        answer: str,
        docs: List[RetrievalResult],
        latency_ms: float,
        num_llm_calls: int,
        policy_decision: PolicyDecision,
        hall_score: float = 0.0,
        iterations_used: int = 0,
    ) -> PipelineResult:
        critique = CritiqueResult(
            support_level="PARTIALLY_SUPPORTED",
            unsupported_parts=[],
            needs_refinement=False,
            critique_text="",
        )

        return PipelineResult(
            query=query,
            final_answer=answer,
            all_answers=[answer],
            hallucination_scores=[hall_score],
            final_hallucination_score=hall_score,
            iterations_used=iterations_used,
            retrieved_docs=docs,
            policy_decision=policy_decision,
            critique_result=critique,
            execution_time_total=latency_ms / 1000.0,
            time_retrieval=0.0,
            time_reranking=0.0,
            time_generation=0.0,
            time_hallucination=0.0,
            num_llm_calls=num_llm_calls,
        )

    def run_without_policy(self, queries, qa_pairs) -> EvaluationReport:
        results: List[PipelineResult] = []
        forced_policy = PolicyDecision(
            query_type="simple",
            k=5,
            prompt_style="direct",
            confidence=1.0,
            expand_retrieval=False,
            k_expand=3,
        )

        for query in queries:
            t0 = time.perf_counter()
            result = self.full_system.refinement_controller.run_pipeline(query, forced_policy)
            result.execution_time_total = time.perf_counter() - t0
            results.append(result)

        return batch_evaluate(results, qa_pairs)

    def run_without_refinement(self, queries, qa_pairs) -> EvaluationReport:
        original_iters = self.config.MAX_REFINE_ITERS
        self.config.MAX_REFINE_ITERS = 0

        try:
            results = [self.full_system.answer(query) for query in queries]
            for result in results:
                if result.iterations_used != 0:
                    raise AssertionError("Expected iterations_used == 0 when MAX_REFINE_ITERS=0")
            return batch_evaluate(results, qa_pairs)
        finally:
            self.config.MAX_REFINE_ITERS = original_iters

    def run_without_hallucination_detector(self, queries, qa_pairs) -> EvaluationReport:
        original_threshold = float(self.config.HALLUCINATION_THRESHOLD)
        try:
            object.__setattr__(self.config, "HALLUCINATION_THRESHOLD", 1.1)
            results = [self.full_system.answer(query) for query in queries]
        finally:
            object.__setattr__(self.config, "HALLUCINATION_THRESHOLD", original_threshold)

        return batch_evaluate(results, qa_pairs)

    def run_without_expansion(self, queries, qa_pairs) -> EvaluationReport:
        results: List[PipelineResult] = []
        for query in queries:
            policy = self._policy_for_query(query)
            forced_policy = PolicyDecision(
                query_type=policy.query_type,
                k=policy.k,
                prompt_style=policy.prompt_style,
                confidence=policy.confidence,
                expand_retrieval=False,
                k_expand=policy.k_expand,
            )
            results.append(self.full_system.refinement_controller.run_pipeline(query, forced_policy))

        return batch_evaluate(results, qa_pairs)

    def run_full_ablation(self, queries, qa_pairs, n: int = 100) -> dict:
        selected_queries = [str(q).strip() for q in queries[:n] if str(q).strip()]

        def _ablation_view(report: EvaluationReport) -> dict:
            data = asdict(report)
            return {
                "em": float(data.get("em", 0.0)),
                "f1": float(data.get("f1", 0.0)),
                "avg_hallucination": float(data.get("avg_hallucination", 0.0)),
                "ece": float(data.get("ece", 0.0)),
            }

        full_results = [self.full_system.answer(query) for query in selected_queries]
        full_report = batch_evaluate(full_results, qa_pairs)

        report_no_policy = self.run_without_policy(selected_queries, qa_pairs)
        report_no_refine = self.run_without_refinement(selected_queries, qa_pairs)
        report_no_hall = self.run_without_hallucination_detector(selected_queries, qa_pairs)
        report_no_expand = self.run_without_expansion(selected_queries, qa_pairs)

        payload = {
            "full_system": _ablation_view(full_report),
            "without_policy": _ablation_view(report_no_policy),
            "without_refinement": _ablation_view(report_no_refine),
            "without_hallucination_detector": _ablation_view(report_no_hall),
            "without_expansion": _ablation_view(report_no_expand),
        }

        os.makedirs(self.config.EVALUATION_DIR, exist_ok=True)
        with open(self.config.ABLATION_RESULTS_PATH, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=True)

        print("\nAblation Table (F1 drop vs full)")
        print(f"{'Setting':<34} | {'F1':>8} | {'Drop':>8}")
        print("-" * 58)
        full_f1 = full_report.f1

        def _row(name: str, rpt: EvaluationReport):
            drop = full_f1 - rpt.f1
            print(f"{name:<34} | {rpt.f1:>8.4f} | {drop:>8.4f}")

        _row("Full PARR-MHQA", full_report)
        _row("Without policy", report_no_policy)
        _row("Without refinement", report_no_refine)
        _row("Without hallucination detector", report_no_hall)
        _row("Without expansion", report_no_expand)

        return payload


def _baseline_rows_to_pipeline_results(rows: List[dict], default_policy: PolicyDecision) -> List[PipelineResult]:
    converted: List[PipelineResult] = []
    for row in rows:
        critique = CritiqueResult(
            support_level="PARTIALLY_SUPPORTED",
            unsupported_parts=[],
            needs_refinement=False,
            critique_text="",
        )
        docs = row.get("retrieved_docs", [])

        hall_score = float(row.get("hallucination_score", 0.0))

        converted.append(
            PipelineResult(
                query=str(row.get("question", "")),
                final_answer=str(row.get("predicted", "")),
                all_answers=[str(row.get("predicted", ""))],
                hallucination_scores=[hall_score],
                final_hallucination_score=hall_score,
                iterations_used=0,
                retrieved_docs=docs,
                policy_decision=default_policy,
                critique_result=critique,
                execution_time_total=float(row.get("latency_ms", 0.0)) / 1000.0,
                time_retrieval=0.0,
                time_reranking=0.0,
                time_generation=0.0,
                time_hallucination=0.0,
                num_llm_calls=int(row.get("num_llm_calls", 1)),
            )
        )

    return converted


def run_full_comparison(system, qa_pairs, n: int = 200):
    selected: List[dict] = []
    for item in qa_pairs:
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        selected.append(item)
        if len(selected) >= n:
            break

    queries = [str(item.get("question", "")).strip() for item in selected]

    llm_baseline = LLMOnlyBaseline(
        retriever=system.refinement_controller.retriever,
        reranker=None,
        prompt_engine=system.prompt_engine,
        llm_generator=system.generator,
        hallucination_detector=system.hallucination_detector,
        k=5,
    )
    rag_baseline = StandardRAGBaseline(
        retriever=system.refinement_controller.retriever,
        reranker=None,
        prompt_engine=system.prompt_engine,
        llm_generator=system.generator,
        k=5,
    )

    llm_rows = llm_baseline.run_on_dataset(selected, n=len(selected))
    rag_rows = rag_baseline.run_on_dataset(selected, n=len(selected))
    parr_results = [system.answer(query) for query in queries]

    dummy_policy = PolicyDecision(
        query_type="simple",
        k=5,
        prompt_style="direct",
        confidence=1.0,
        expand_retrieval=False,
        k_expand=3,
    )

    llm_eval_results = _baseline_rows_to_pipeline_results(llm_rows, dummy_policy)
    rag_eval_results = _baseline_rows_to_pipeline_results(rag_rows, dummy_policy)

    llm_report = batch_evaluate(llm_eval_results, selected)
    rag_report = batch_evaluate(rag_eval_results, selected)
    parr_report = batch_evaluate(parr_results, selected)

    ablation = AblationStudy(system)
    ablation_payload = ablation.run_full_ablation(queries, selected, n=min(100, len(queries)))

    print("\nCost Analysis")
    print(f"{'System':<16} | {'Avg LLM calls/query':>20} | {'F1':>8}")
    print("-" * 52)
    print(f"{'LLM-only':<16} | {llm_report.avg_llm_calls:>20.2f} | {llm_report.f1:>8.4f}")
    print(f"{'Standard RAG':<16} | {rag_report.avg_llm_calls:>20.2f} | {rag_report.f1:>8.4f}")
    print(f"{'PARR-MHQA':<16} | {parr_report.avg_llm_calls:>20.2f} | {parr_report.f1:>8.4f}")

    print("\nFull Comparison Reports")
    print("\n[LLM-only]")
    print_report(llm_report)
    print("\n[Standard RAG]")
    print_report(rag_report)
    print("\n[PARR-MHQA]")
    print_report(parr_report)

    payload = {
        "llm_only": asdict(llm_report),
        "standard_rag": asdict(rag_report),
        "parr_mhqa": asdict(parr_report),
        "ablation": ablation_payload,
    }

    os.makedirs(system.config.EVALUATION_DIR, exist_ok=True)
    with open(system.config.COMPARISON_RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)

    return payload
