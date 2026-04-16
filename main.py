import argparse
import json
import logging
import os
import pickle
import runpy
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from config import Config, setup_logging, setup_reproducibility
from generation.hallucination_detector import HallucinationDetector
from generation.llm_generator import get_generator
from generation.prompt_engine import PromptEngine
from generation.refinement_controller import PipelineResult, RefinementController
from generation.self_critique import SelfCritiqueModule
from policy.policy_network import PolicyDecision
from retrieval.document_store import DocumentStore
from retrieval.embedding_engine import EmbeddingEngine
from retrieval.retrieval_system import AdaptiveRetriever


class PARRMHQASystem:
    def __init__(self, config: Config, llm_mode: str = "eval"):
        self.config = config
        self.llm_mode = llm_mode
        self.logger = logging.getLogger(__name__)

        self.embedding_engine: Optional[EmbeddingEngine] = None
        self.document_store: Optional[DocumentStore] = None
        self.faiss_index = None
        self.generator = None
        self.hallucination_detector: Optional[HallucinationDetector] = None
        self.critique_module: Optional[SelfCritiqueModule] = None
        self.refinement_controller: Optional[RefinementController] = None
        self.prompt_engine: Optional[PromptEngine] = None

    def _load_chunks(self) -> List[dict]:
        candidates = [
            Path("embeddings/processed_chunks.pkl"),
            Path("embeddings/chunks.pkl"),
            Path(self.config.DOCS_PATH),
        ]

        for path in candidates:
            if not path.exists():
                continue
            with open(path, "rb") as file:
                data = pickle.load(file)

            if isinstance(data, list) and data:
                sample = data[0]
                if isinstance(sample, dict) and "chunk_id" in sample and "text" in sample:
                    return data
                if isinstance(sample, dict) and "doc_id" in sample and "text" in sample:
                    chunks = []
                    for idx, doc in enumerate(data):
                        chunks.append(
                            {
                                "chunk_id": f"{doc.get('doc_id', f'doc_{idx}')}_chunk_0",
                                "doc_id": str(doc.get("doc_id", f"doc_{idx}")),
                                "title": str(doc.get("title", "")),
                                "text": str(doc.get("text", "")),
                                "chunk_index": 0,
                            }
                        )
                    return chunks

        raise FileNotFoundError(
            "No chunk/doc pickle found. Expected one of: embeddings/processed_chunks.pkl, "
            "embeddings/chunks.pkl, or config.DOCS_PATH"
        )

    def _init_component(
        self,
        name: str,
        loader: Callable[[], Any],
        rows: List[Dict[str, str]],
    ) -> Any:
        self.logger.info("[INIT] Loading %s...", name)
        start = time.perf_counter()
        try:
            value = loader()
            elapsed = time.perf_counter() - start
            rows.append({"component": name, "status": "OK", "time": f"{elapsed:.3f}"})
            self.logger.info("[INIT] Loaded %s in %.3fs", name, elapsed)
            return value
        except Exception as exc:
            elapsed = time.perf_counter() - start
            rows.append({"component": name, "status": f"FAILED ({exc})", "time": f"{elapsed:.3f}"})
            self.logger.exception("[INIT] Failed loading %s", name)
            raise

    def _print_init_table(self, rows: List[Dict[str, str]]) -> None:
        print("\nInitialization Summary")
        print(f"{'component':<24} | {'status':<32} | {'time(s)':>8}")
        print("-" * 72)
        for row in rows:
            print(f"{row['component']:<24} | {row['status']:<32} | {row['time']:>8}")

    def initialize(self):
        rows: List[Dict[str, str]] = []

        self.embedding_engine = self._init_component(
            "EmbeddingEngine",
            lambda: EmbeddingEngine(self.config.EMBEDDING_MODEL, self.config.CACHE_DIR),
            rows,
        )

        chunks = self._load_chunks()

        self.document_store = self._init_component(
            "DocumentStore",
            lambda: self._build_document_store(chunks),
            rows,
        )

        self.faiss_index = self._init_component(
            "FAISSIndex",
            lambda: self._load_or_build_faiss(chunks),
            rows,
        )

        self.generator = self._init_component(
            "LLMGenerator",
            lambda: get_generator(self.config, mode=self.llm_mode),
            rows,
        )

        self.hallucination_detector = self._init_component(
            "HallucinationDetector",
            lambda: HallucinationDetector(self.embedding_engine, self.config),
            rows,
        )

        self.prompt_engine = self._init_component(
            "PromptEngine",
            PromptEngine,
            rows,
        )

        self.critique_module = self._init_component(
            "SelfCritiqueModule",
            lambda: SelfCritiqueModule(self.generator, self.prompt_engine),
            rows,
        )

        self.refinement_controller = self._init_component(
            "RefinementController",
            lambda: RefinementController(
                retriever=AdaptiveRetriever(
                    embedding_engine=self.embedding_engine,
                    faiss_index=self.faiss_index,
                    document_store=self.document_store,
                    config=self.config,
                ),
                generator=self.generator,
                hallucination_detector=self.hallucination_detector,
                critique_module=self.critique_module,
                config=self.config,
                prompt_engine=self.prompt_engine,
            ),
            rows,
        )

        self._print_init_table(rows)

    def _build_document_store(self, chunks: List[dict]) -> DocumentStore:
        store = DocumentStore()
        store.build(chunks)
        return store

    def _load_or_build_faiss(self, chunks: List[dict]):
        if os.path.exists(self.config.FAISS_INDEX_PATH):
            return self.embedding_engine.load_index(self.config.FAISS_INDEX_PATH)

        embeddings = self.embedding_engine.embed_documents(chunks, batch_size=64, show_progress=True)
        index = self.embedding_engine.build_faiss_index(embeddings)
        self.embedding_engine.save_index(index, self.config.FAISS_INDEX_PATH)
        return index

    def _default_policy_decision(self, query: str) -> PolicyDecision:
        query_text = str(query).strip().lower()
        is_multi_hop = any(token in query_text for token in [" and ", " or ", "compare", "versus", "both", "which"])
        return PolicyDecision(
            query_type="multi_hop" if is_multi_hop else "simple",
            k=self.config.TOP_K_MULTIHOP if is_multi_hop else self.config.TOP_K_DEFAULT,
            prompt_style="chain_of_thought" if is_multi_hop else "direct",
            confidence=1.0,
            expand_retrieval=is_multi_hop,
            k_expand=3,
        )

    def answer(self, query: str) -> PipelineResult:
        policy = self._default_policy_decision(query)
        self.logger.info("policy_decision | %s", asdict(policy))

        result = self.refinement_controller.run_pipeline(query, policy)
        self.logger.info(
            "answer_complete | hall_score=%.4f | iters=%s | llm_calls=%s | answer=%s",
            result.final_hallucination_score,
            result.iterations_used,
            result.num_llm_calls,
            result.final_answer,
        )
        return result

    def _result_to_dict(self, result: PipelineResult) -> dict:
        data = asdict(result)

        def _convert(obj):
            if is_dataclass(obj):
                return asdict(obj)
            if isinstance(obj, list):
                return [_convert(item) for item in obj]
            if isinstance(obj, dict):
                return {key: _convert(value) for key, value in obj.items()}
            return obj

        return _convert(data)

    def batch_answer(self, queries, save_results: bool = True) -> List[PipelineResult]:
        results: List[PipelineResult] = []
        for query in tqdm(queries, desc="PARR-MHQA Batch", unit="query"):
            if not str(query).strip():
                continue
            results.append(self.answer(str(query).strip()))

        if save_results:
            os.makedirs("evaluation", exist_ok=True)
            payload = [self._result_to_dict(result) for result in results]
            with open("evaluation/batch_results.json", "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=True)

        return results

    def run_evaluate_mode(self, queries_file: Optional[str]) -> List[PipelineResult]:
        evaluator_candidates = [
            Path("evaluation/phase13.py"),
            Path("evaluation/run_evaluation.py"),
            Path("evaluation/evaluator.py"),
        ]

        for candidate in evaluator_candidates:
            if candidate.exists():
                self.logger.info("[EVALUATE] Running Phase 13 evaluator: %s", candidate)
                runpy.run_path(str(candidate), run_name="__main__")
                return []

        self.logger.info("[EVALUATE] Phase 13 evaluator file not found, falling back to batch evaluation.")
        if not queries_file:
            raise ValueError("Evaluate mode requires --queries-file when no Phase 13 script is present.")

        with open(queries_file, "r", encoding="utf-8") as file:
            queries = [line.strip() for line in file if line.strip()]
        return self.batch_answer(queries, save_results=True)


def _score_color_tag(score: float) -> str:
    if score < 0.35:
        return "\033[92mPASS\033[0m"
    if score <= 0.6:
        return "\033[93mWARN\033[0m"
    return "\033[91mFAIL\033[0m"


def _print_interactive_result(result: PipelineResult) -> None:
    top_titles = [doc.title for doc in result.retrieved_docs[:3]]
    print("\nAnswer:")
    print(result.final_answer)
    print(f"Hallucination score: {_score_color_tag(result.final_hallucination_score)} {result.final_hallucination_score:.4f}")
    print(f"Iterations used: {result.iterations_used}")
    print(f"Query type: {result.policy_decision.query_type}")
    print(f"CSR score: {getattr(result, 'csr_score', 0.0):.4f}")
    print(f"Reasoning consistency: {getattr(result, 'reasoning_consistency', 0.0):.4f}")
    print(f"Num LLM calls: {result.num_llm_calls}")
    print(f"Top 3 retrieved docs: {top_titles}")
    print(f"Total time (ms): {result.execution_time_total * 1000:.2f}")


def _print_batch_summary(results: List[PipelineResult]) -> None:
    if not results:
        print("No results generated.")
        return

    hall_scores = [result.final_hallucination_score for result in results]
    times_ms = [result.execution_time_total * 1000 for result in results]
    total_llm_calls = sum(result.num_llm_calls for result in results)

    print("\nBatch Summary")
    print(f"Queries processed: {len(results)}")
    print(f"Avg hallucination score: {float(np.mean(hall_scores)):.4f}")
    print(f"Avg total time (ms): {float(np.mean(times_ms)):.2f}")
    print(f"Total LLM calls: {total_llm_calls}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PARR-MHQA System")
    parser.add_argument("--mode", choices=["interactive", "batch", "evaluate"], default="interactive")
    parser.add_argument("--queries-file", type=str, default=None, help="Path to newline-delimited query file")
    parser.add_argument("--no-save", action="store_true", help="Do not save batch/eval outputs")
    args = parser.parse_args()

    cfg = Config()
    setup_reproducibility()
    setup_logging(cfg.LOG_DIR)

    system = PARRMHQASystem(cfg)
    system.initialize()

    if args.mode == "interactive":
        print("PARR-MHQA Interactive Mode. Type 'exit' to quit.")
        while True:
            query = input("\nQuery> ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue
            result = system.answer(query)
            _print_interactive_result(result)

    elif args.mode == "batch":
        if not args.queries_file:
            raise ValueError("Batch mode requires --queries-file")
        with open(args.queries_file, "r", encoding="utf-8") as file:
            queries = [line.strip() for line in file if line.strip()]
        batch_results = system.batch_answer(queries, save_results=not args.no_save)
        _print_batch_summary(batch_results)

    else:  # evaluate
        eval_results = system.run_evaluate_mode(args.queries_file)
        if eval_results:
            _print_batch_summary(eval_results)
