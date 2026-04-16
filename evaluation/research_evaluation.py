import json
import os
import logging
from dataclasses import asdict
from typing import Dict, List

from tqdm import tqdm

from config import Config, setup_reproducibility
from evaluation.baselines import HeuristicBaseline, LLMOnlyBaseline, LearnedBaseline
from evaluation.metrics import EvaluationReport, batch_evaluate
from generation.refinement_controller import PipelineResult
from generation.self_critique import CritiqueResult
from policy.policy_network import PolicyDecision


def _save_paper_tables(cfg: Config, payload: Dict[str, dict]) -> None:
    lines: List[str] = []

    lines.append("## Main Table")
    lines.append("| Model | F1 | Hallucination ↓ | ECE ↓ |")
    lines.append("| --- | --- | --- | --- |")

    reports = payload.get("reports", {})
    llm = reports.get("llm_only", {})
    heuristic = reports.get("heuristic", {})
    learned = reports.get("learned", {})

    lines.append(
        f"| LLM_ONLY | {float(llm.get('f1', 0.0)):.4f} | {float(llm.get('avg_hallucination', 0.0)):.4f} | {float(llm.get('ece', 0.0)):.4f} |"
    )
    lines.append(
        f"| HEURISTIC | {float(heuristic.get('f1', 0.0)):.4f} | {float(heuristic.get('avg_hallucination', 0.0)):.4f} | {float(heuristic.get('ece', 0.0)):.4f} |"
    )
    lines.append(
        f"| LEARNED | {float(learned.get('f1', 0.0)):.4f} | {float(learned.get('avg_hallucination', 0.0)):.4f} | {float(learned.get('ece', 0.0)):.4f} |"
    )

    lines.append("")
    lines.append("## Ablation")
    lines.append("| Model | F1 | Hallucination ↓ | ECE ↓ |")
    lines.append("| --- | --- | --- | --- |")

    ablation = payload.get("ablation", {})
    lines.append(
        f"| FULL | {float(ablation.get('FULL', {}).get('f1', 0.0)):.4f} | {float(ablation.get('FULL', {}).get('avg_hallucination', 0.0)):.4f} | {float(ablation.get('FULL', {}).get('ece', 0.0)):.4f} |"
    )
    lines.append(
        f"| NO_DETECTOR | {float(ablation.get('NO_DETECTOR', {}).get('f1', 0.0)):.4f} | {float(ablation.get('NO_DETECTOR', {}).get('avg_hallucination', 0.0)):.4f} | {float(ablation.get('NO_DETECTOR', {}).get('ece', 0.0)):.4f} |"
    )
    lines.append(
        f"| NO_MULTI_SIGNAL | {float(ablation.get('NO_MULTI_SIGNAL', {}).get('f1', 0.0)):.4f} | {float(ablation.get('NO_MULTI_SIGNAL', {}).get('avg_hallucination', 0.0)):.4f} | {float(ablation.get('NO_MULTI_SIGNAL', {}).get('ece', 0.0)):.4f} |"
    )

    with open(cfg.PAPER_TABLES_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def _report_to_dict(report: EvaluationReport, include_llm_calls: bool = True) -> Dict[str, float]:
    data = asdict(report)
    if include_llm_calls:
        return data

    return {
        "em": float(data.get("em", 0.0)),
        "f1": float(data.get("f1", 0.0)),
        "avg_hallucination": float(data.get("avg_hallucination", 0.0)),
        "ece": float(data.get("ece", 0.0)),
    }


def _run_system_queries(system, queries: List[str], desc: str) -> List[PipelineResult]:
    results: List[PipelineResult] = []
    for query in tqdm(queries, desc=desc, leave=False, dynamic_ncols=True, mininterval=0.2):
        results.append(system.answer(query))
    return results


def _run_no_detector(system, selected: List[dict]) -> List[PipelineResult]:
    cfg = system.config
    original_threshold = float(cfg.HALLUCINATION_THRESHOLD)
    queries = [str(item.get("question", "")).strip() for item in selected if str(item.get("question", "")).strip()]
    try:
        object.__setattr__(cfg, "HALLUCINATION_THRESHOLD", 1.1)
        return _run_system_queries(system, queries, desc="Ablation (NO_DETECTOR)")
    finally:
        object.__setattr__(cfg, "HALLUCINATION_THRESHOLD", original_threshold)


def _run_no_multi_signal(system, queries: List[str]) -> List[PipelineResult]:
    detector = system.hallucination_detector
    original_flag = bool(getattr(detector, "use_multi_signal", True))
    detector.use_multi_signal = False
    try:
        return _run_system_queries(system, queries, desc="Ablation (NO_MULTI_SIGNAL)")
    finally:
        detector.use_multi_signal = original_flag


def _pipeline_stub(
    query: str,
    answer: str,
    docs,
    latency_ms: float,
    llm_calls: int,
    hallucination_score: float = 0.0,
) -> PipelineResult:
    return PipelineResult(
        query=query,
        final_answer=answer,
        all_answers=[answer],
        hallucination_scores=[hallucination_score],
        final_hallucination_score=hallucination_score,
        iterations_used=0,
        retrieved_docs=docs,
        policy_decision=PolicyDecision(
            query_type="simple",
            k=5,
            prompt_style="direct",
            confidence=1.0,
            expand_retrieval=False,
            k_expand=3,
        ),
        critique_result=CritiqueResult(
            support_level="PARTIALLY_SUPPORTED",
            unsupported_parts=[],
            needs_refinement=False,
            critique_text="",
        ),
        execution_time_total=latency_ms / 1000.0,
        time_retrieval=0.0,
        time_reranking=0.0,
        time_generation=0.0,
        time_hallucination=0.0,
        num_llm_calls=llm_calls,
        csr_score=0.0,
        reasoning_consistency=0.0,
        reasoning_trace=[],
    )


def _rows_to_pipeline_results(rows: List[dict]) -> List[PipelineResult]:
    results: List[PipelineResult] = []
    for row in tqdm(rows, desc="Converting baseline rows", leave=False, dynamic_ncols=True, mininterval=0.2):
        results.append(
            _pipeline_stub(
                query=str(row.get("question", "")),
                answer=str(row.get("predicted", "")),
                docs=row.get("retrieved_docs", []),
                latency_ms=float(row.get("latency_ms", 0.0)),
                llm_calls=int(row.get("num_llm_calls", 1)),
                hallucination_score=float(row.get("hallucination_score", 0.0)),
            )
        )
    return results


def run_research_evaluation(system, qa_pairs: List[dict], n: int = 200) -> Dict[str, dict]:
    logger = logging.getLogger(__name__)
    cfg: Config = system.config
    setup_reproducibility(cfg.RANDOM_SEED)
    selected = [item for item in qa_pairs if str(item.get("question", "")).strip()][:n]
    queries = [str(item.get("question", "")).strip() for item in selected]
    logger.info("research_eval_start | n=%s", len(selected))

    llm_base = LLMOnlyBaseline(
        retriever=system.refinement_controller.retriever,
        reranker=None,
        prompt_engine=system.prompt_engine,
        llm_generator=system.generator,
        hallucination_detector=system.hallucination_detector,
        k=5,
    )
    heuristic_base = HeuristicBaseline(
        retriever=system.refinement_controller.retriever,
        reranker=None,
        prompt_engine=system.prompt_engine,
        llm_generator=system.generator,
        hallucination_detector=system.hallucination_detector,
        k=5,
        threshold=float(cfg.HALLUCINATION_THRESHOLD),
    )
    learned_base = LearnedBaseline(
        retriever=system.refinement_controller.retriever,
        reranker=None,
        prompt_engine=system.prompt_engine,
        llm_generator=system.generator,
        hallucination_detector=system.hallucination_detector,
        k=5,
        threshold=float(cfg.HALLUCINATION_THRESHOLD),
    )

    llm_rows = llm_base.run_on_dataset(selected, n=len(selected))
    heuristic_rows = heuristic_base.run_on_dataset(selected, n=len(selected))
    learned_rows = learned_base.run_on_dataset(selected, n=len(selected))

    llm_results = _rows_to_pipeline_results(llm_rows)
    heuristic_results = _rows_to_pipeline_results(heuristic_rows)
    learned_results = _rows_to_pipeline_results(learned_rows)
    full_system_results = _run_system_queries(system, queries, desc="FULL SYSTEM")

    reports: Dict[str, EvaluationReport] = {
        "llm_only": batch_evaluate(llm_results, selected),
        "heuristic": batch_evaluate(heuristic_results, selected),
        "learned": batch_evaluate(learned_results, selected),
        # Keep compatibility with existing consumers expecting parr_mhqa key.
        "parr_mhqa": batch_evaluate(full_system_results, selected),
    }
    logger.info("research_eval_reports | keys=%s", list(reports.keys()))

    full_results = full_system_results
    no_detector_results = _run_no_detector(system, selected)
    no_multi_signal_results = _run_no_multi_signal(system, queries)

    ablation_reports: Dict[str, EvaluationReport] = {
        "FULL": batch_evaluate(full_results, selected),
        "NO_DETECTOR": batch_evaluate(no_detector_results, selected),
        "NO_MULTI_SIGNAL": batch_evaluate(no_multi_signal_results, selected),
    }
    ablation = {key: _report_to_dict(val, include_llm_calls=False) for key, val in ablation_reports.items()}

    payload = {
        "reports": {key: _report_to_dict(val, include_llm_calls=True) for key, val in reports.items()},
        "ablation": ablation,
    }

    os.makedirs(cfg.EVALUATION_DIR, exist_ok=True)
    with open(cfg.RESEARCH_RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)

    _save_paper_tables(cfg, payload)

    logger.info("research_eval_complete | output=%s", cfg.RESEARCH_RESULTS_PATH)

    return payload
