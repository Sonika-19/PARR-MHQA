import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from generation.refinement_controller import PipelineResult


ALIASES = {
    "uk": ["united kingdom", "great britain"],
    "usa": ["united states", "united states of america"],
    "us": ["united states", "united states of america"],
    "ussr": ["soviet union"],
}


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    chars = []
    for i, ch in enumerate(text):
        if ch in string.punctuation:
            if ch == "-":
                prev_is_alnum = i > 0 and text[i - 1].isalnum()
                next_is_alnum = i + 1 < len(text) and text[i + 1].isalnum()
                if prev_is_alnum and next_is_alnum:
                    chars.append(ch)
                    continue
            chars.append(" ")
        else:
            chars.append(ch)

    text = "".join(chars)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_with_alias(pred: str, gold: str) -> int:
    pred_n = normalize_answer(pred)
    gold_n = normalize_answer(gold)

    if pred_n == gold_n:
        return 1

    for key, vals in ALIASES.items():
        vals_n = [normalize_answer(v) for v in vals]
        key_n = normalize_answer(key)

        if (pred_n in vals_n and gold_n == key_n) or (gold_n in vals_n and pred_n == key_n):
            return 1

        if pred_n == key_n and gold_n in vals_n:
            return 1
        if gold_n == key_n and pred_n in vals_n:
            return 1

    return 0


def exact_match(predicted, ground_truth) -> float:
    pred_n = normalize_answer(predicted)
    gold_n = normalize_answer(ground_truth)
    return float(match_with_alias(pred_n, gold_n))


def f1_score_qa(predicted, ground_truth) -> float:
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return float(2 * precision * recall / (precision + recall))


@dataclass
class EvaluationReport:
    em: float
    f1: float
    avg_hallucination: float
    ece: float
    avg_llm_calls: float


def _safe_avg(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _context_confidence(predicted: str, context_texts: List[str]) -> float:
    pred_tokens = set(normalize_answer(predicted).split())
    context_tokens = set(normalize_answer(" ".join(context_texts)).split())

    if not pred_tokens or not context_tokens:
        return 0.1

    overlap_ratio = float(len(pred_tokens & context_tokens) / len(pred_tokens))
    return float(np.clip(overlap_ratio, 0.1, 0.9))


def expected_calibration_error(confidences: List[float], correctness: List[float], n_bins: int = 10) -> float:
    if not confidences:
        return 0.0

    conf = np.asarray(confidences, dtype=np.float32)
    corr = np.asarray(correctness, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for idx in range(n_bins):
        low, high = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf >= low) & (conf < high)

        if not np.any(mask):
            continue

        bin_conf = float(np.mean(conf[mask]))
        bin_acc = float(np.mean(corr[mask]))
        weight = float(np.mean(mask.astype(np.float32)))
        ece += abs(bin_acc - bin_conf) * weight

    return float(ece)


def batch_evaluate(results: List[PipelineResult], qa_pairs: List[dict]) -> EvaluationReport:
    qa_by_question = {str(item.get("question", "")).strip(): item for item in qa_pairs}

    em_scores: List[float] = []
    f1_scores: List[float] = []
    hallucination_scores: List[float] = []
    grounding_conf: List[float] = []
    correctness_labels: List[float] = []
    llm_calls: List[float] = []

    for result in results:
        qa = qa_by_question.get(str(result.query).strip())
        if qa is None:
            continue

        gold = str(qa.get("answer", ""))
        pred = str(result.final_answer)
        context_texts = [doc.text for doc in result.retrieved_docs]

        em = exact_match(pred, gold)
        f1 = f1_score_qa(pred, gold)

        em_scores.append(em)
        f1_scores.append(f1)
        hallucination_scores.append(float(result.final_hallucination_score))
        grounding_conf.append(_context_confidence(pred, context_texts))
        correctness_labels.append(float(em))
        llm_calls.append(float(result.num_llm_calls))

    return EvaluationReport(
        em=_safe_avg(em_scores),
        f1=_safe_avg(f1_scores),
        avg_hallucination=_safe_avg(hallucination_scores),
        ece=expected_calibration_error(grounding_conf, correctness_labels),
        avg_llm_calls=_safe_avg(llm_calls),
    )


def compare_systems(baseline_llm, baseline_rag, parr_results, qa_pairs) -> dict:
    llm_report = batch_evaluate(baseline_llm, qa_pairs)
    rag_report = batch_evaluate(baseline_rag, qa_pairs)
    parr_report = batch_evaluate(parr_results, qa_pairs)

    em_improve_pct = 0.0
    if rag_report.em != 0:
        em_improve_pct = float(((parr_report.em - rag_report.em) / rag_report.em) * 100.0)

    f1_improve_pct = 0.0
    if rag_report.f1 != 0:
        f1_improve_pct = float(((parr_report.f1 - rag_report.f1) / rag_report.f1) * 100.0)

    return {
        "baseline_llm": llm_report,
        "baseline_rag": rag_report,
        "parr_mhqa": parr_report,
        "improvement_over_rag_pct": {
            "em": em_improve_pct,
            "f1": f1_improve_pct,
        },
    }


def print_report(report):
    print("+----------------------+----------------+")
    print("| Metric               | Value          |")
    print("+----------------------+----------------+")
    print(f"| EM                   | {report.em:>14.4f} |")
    print(f"| F1                   | {report.f1:>14.4f} |")
    print(f"| Avg hallucination    | {report.avg_hallucination:>14.4f} |")
    print(f"| ECE                  | {report.ece:>14.4f} |")
    print(f"| Avg LLM calls        | {report.avg_llm_calls:>14.2f} |")
    print("+----------------------+----------------+")
