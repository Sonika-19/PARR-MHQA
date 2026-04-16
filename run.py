import argparse
import json
import logging
import os
import pickle
from collections import defaultdict
from typing import List
from tqdm import tqdm

from config import Config, setup_logging, setup_reproducibility
from data.dataset_loader import HotpotQALoader
from evaluation.research_evaluation import run_research_evaluation
from generation.learned_hallucination_model import (
    LearnedHallucinationDetector,
    build_synthetic_hallucination_dataset,
    load_synthetic_dataset,
    save_synthetic_dataset,
    stratified_split_samples,
)
from main import PARRMHQASystem
from retrieval.embedding_engine import EmbeddingEngine


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass


def _configure_tqdm_logging() -> None:
    root = logging.getLogger()
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")

    keep_handlers = []
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            keep_handlers.append(handler)
    root.handlers = keep_handlers

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(formatter)
    root.addHandler(tqdm_handler)


def _ensure_dirs(cfg: Config) -> None:
    for path in [cfg.CACHE_DIR, cfg.LOG_DIR, cfg.EVALUATION_DIR, "embeddings", "models", "data"]:
        os.makedirs(path, exist_ok=True)


def _build_processed_artifacts(cfg: Config) -> None:
    has_chunks = os.path.exists(cfg.PROCESSED_CHUNKS_PATH)
    has_qa = os.path.exists(cfg.PROCESSED_QA_PATH)
    has_docs = os.path.exists(cfg.PROCESSED_DOCS_PATH)
    if has_chunks and has_qa and has_docs:
        return

    loader = HotpotQALoader()
    dataset = loader.load(split="train")
    documents = loader.extract_documents(dataset, max_samples=50000)
    qa_pairs = loader.extract_qa_pairs(dataset)
    chunks = loader.chunk_documents(documents, chunk_size=cfg.CHUNK_SIZE, overlap=cfg.CHUNK_OVERLAP)

    loader.save_processed(documents, cfg.PROCESSED_DOCS_PATH)
    loader.save_processed(qa_pairs, cfg.PROCESSED_QA_PATH)
    loader.save_processed(chunks, cfg.PROCESSED_CHUNKS_PATH)


def _build_embeddings_and_index(cfg: Config) -> None:
    if os.path.exists(cfg.FAISS_INDEX_PATH) and os.path.exists(os.path.join(cfg.CACHE_DIR, "doc_embeddings.npy")):
        return

    if not os.path.exists(cfg.PROCESSED_CHUNKS_PATH):
        _build_processed_artifacts(cfg)

    with open(cfg.PROCESSED_CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    engine = EmbeddingEngine(cfg.EMBEDDING_MODEL, cfg.CACHE_DIR)
    embeddings = engine.embed_documents(chunks, batch_size=64, show_progress=True)
    index = engine.build_faiss_index(embeddings)
    engine.save_index(index, cfg.FAISS_INDEX_PATH)


def _build_title_to_contexts(chunks: List[dict]) -> dict:
    mapping = defaultdict(list)
    for chunk in chunks:
        title = str(chunk.get("title", "")).strip()
        text = str(chunk.get("text", "")).strip()
        if not title or not text:
            continue
        if len(mapping[title]) < 5:
            mapping[title].append(text)
    return dict(mapping)


def _train_learned_hallucination_model(cfg: Config, max_samples: int = 20000) -> None:
    if os.path.exists(cfg.LEARNED_HALLUCINATION_MODEL_PATH):
        return

    qa_pairs = _load_qa_pairs(cfg)
    with open(cfg.PROCESSED_CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    if os.path.exists(cfg.HALLUCINATION_TRAIN_20K_PATH):
        samples = load_synthetic_dataset(cfg.HALLUCINATION_TRAIN_20K_PATH)
        print(f"Loaded fixed dataset from: {cfg.HALLUCINATION_TRAIN_20K_PATH}")
    else:
        print("Generating dataset (first run only)...")
        title_map = _build_title_to_contexts(chunks)
        samples = build_synthetic_hallucination_dataset(
            qa_pairs=qa_pairs,
            title_to_contexts=title_map,
            max_samples=max_samples,
            seed=cfg.RANDOM_SEED,
        )
        save_synthetic_dataset(samples, cfg.HALLUCINATION_TRAIN_20K_PATH, seed=cfg.RANDOM_SEED)

    if len(samples) != max_samples:
        raise RuntimeError(f"Expected {max_samples} samples, found {len(samples)} in {cfg.HALLUCINATION_TRAIN_20K_PATH}")

    baseline_samples = samples[: cfg.HALLUCINATION_BASELINE_SIZE]

    save_synthetic_dataset(baseline_samples, cfg.HALLUCINATION_TRAIN_4K_PATH, seed=cfg.RANDOM_SEED)
    save_synthetic_dataset(samples, cfg.HALLUCINATION_SYNTHETIC_DATA_PATH, seed=cfg.RANDOM_SEED)

    train_samples, val_samples = stratified_split_samples(samples, train_ratio=0.8, seed=cfg.RANDOM_SEED)

    detector = LearnedHallucinationDetector(cfg, train_from_scratch=cfg.TRAIN_FROM_SCRATCH)
    detector.fit(train_samples=train_samples, val_samples=val_samples, epochs=2, batch_size=8)


def setup_command(cfg: Config) -> None:
    _ensure_dirs(cfg)
    _build_processed_artifacts(cfg)
    _build_embeddings_and_index(cfg)
    print("Setup complete.")


def train_command(cfg: Config) -> None:
    _ensure_dirs(cfg)
    _build_processed_artifacts(cfg)
    _build_embeddings_and_index(cfg)
    _train_learned_hallucination_model(cfg, max_samples=cfg.HALLUCINATION_TRAIN_SIZE)
    print("Training complete.")


def _load_qa_pairs(cfg: Config) -> List[dict]:
    if not os.path.exists(cfg.PROCESSED_QA_PATH):
        _build_processed_artifacts(cfg)
    with open(cfg.PROCESSED_QA_PATH, "rb") as file:
        return pickle.load(file)


def _print_final_summary(comparison_payload: dict) -> None:
    if "reports" in comparison_payload:
        llm = comparison_payload.get("reports", {}).get("llm_only", {})
        parr = comparison_payload.get("reports", {}).get("parr_mhqa", {})
    else:
        llm = comparison_payload.get("llm_only", {})
        parr = comparison_payload.get("parr_mhqa", {})

    print("\n=== FINAL RESULTS ===")
    print("Model                F1     Hallucination Rate")
    print("-----------------------------------------------")
    print(f"Baseline (LLM)       {llm.get('f1', 0.0):.3f}  {llm.get('avg_hallucination', 0.0):.3f}")
    print(f"Proposed (+Detector) {parr.get('f1', 0.0):.3f}  {parr.get('avg_hallucination', 0.0):.3f}")

    llm_f1 = float(llm.get("f1", 0.0))
    parr_f1 = float(parr.get("f1", 0.0))
    f1_improve = ((parr_f1 - llm_f1) / llm_f1 * 100.0) if llm_f1 > 0 else 0.0
    print("\nImprovement over baseline:")
    print(f"F1: +{f1_improve:.2f}%")


def generate_all_results(n: int = 200) -> dict:
    cfg = Config()
    setup_reproducibility(cfg.RANDOM_SEED)
    setup_logging(cfg.LOG_DIR)
    _configure_tqdm_logging()

    _ensure_dirs(cfg)
    _build_processed_artifacts(cfg)
    _build_embeddings_and_index(cfg)
    _train_learned_hallucination_model(cfg, max_samples=cfg.HALLUCINATION_TRAIN_SIZE)

    qa_pairs = _load_qa_pairs(cfg)
    system = PARRMHQASystem(cfg)
    system.initialize()

    research_payload = run_research_evaluation(system, qa_pairs, n=n)

    _print_final_summary(research_payload)

    return {"research": research_payload}


def evaluate_command(cfg: Config, n: int = 200) -> None:
    payload = generate_all_results(n=n)
    with open(cfg.COMPARISON_RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)


def research_command(cfg: Config, n: int = 200) -> None:
    _configure_tqdm_logging()
    _ensure_dirs(cfg)
    _build_processed_artifacts(cfg)
    _build_embeddings_and_index(cfg)
    _train_learned_hallucination_model(cfg, max_samples=cfg.HALLUCINATION_TRAIN_SIZE)

    qa_pairs = _load_qa_pairs(cfg)
    system = PARRMHQASystem(cfg)
    system.initialize()

    payload = run_research_evaluation(system, qa_pairs, n=n)
    _print_final_summary(payload)
    print(f"Research report saved to {cfg.RESEARCH_RESULTS_PATH}")


def demo_command(cfg: Config) -> None:
    _ensure_dirs(cfg)
    _build_processed_artifacts(cfg)
    _build_embeddings_and_index(cfg)

    system = PARRMHQASystem(cfg)
    system.initialize()

    print("PARR-MHQA Demo Mode. Type 'exit' to quit.")
    while True:
        query = input("\nQuery> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        result = system.answer(query)
        print("\nAnswer:")
        print(result.final_answer)
        print(f"Hallucination: {result.final_hallucination_score:.4f}")
        print(f"Iterations: {result.iterations_used} | Query type: {result.policy_decision.query_type} | LLM calls: {result.num_llm_calls}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PARR-MHQA unified runner")
    parser.add_argument("command", choices=["setup", "train", "evaluate", "demo", "research"])
    parser.add_argument("--n", type=int, default=200, help="Sample size for evaluate")
    args = parser.parse_args()

    cfg = Config()
    setup_reproducibility(cfg.RANDOM_SEED)
    setup_logging(cfg.LOG_DIR)

    if args.command == "setup":
        setup_command(cfg)
    elif args.command == "train":
        train_command(cfg)
    elif args.command == "evaluate":
        evaluate_command(cfg, n=args.n)
    elif args.command == "research":
        research_command(cfg, n=args.n)
    else:
        demo_command(cfg)


if __name__ == "__main__":
    main()
