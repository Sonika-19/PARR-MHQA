import logging
import os
import random
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass(frozen=True)
class Config:
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_MODEL: str = "microsoft/phi-2"
    FAISS_INDEX_PATH: str = "embeddings/faiss.index"
    DOCS_PATH: str = "embeddings/documents.pkl"
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50
    TOP_K_DEFAULT: int = 5
    TOP_K_MULTIHOP: int = 8
    MAX_REFINE_ITERS: int = 1
    HALLUCINATION_THRESHOLD: float = 0.5
    CACHE_DIR: str = "embeddings/cache"
    LOG_DIR: str = "logs"
    RANDOM_SEED: int = 42
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    PROCESSED_DOCS_PATH: str = "embeddings/processed_documents.pkl"
    PROCESSED_QA_PATH: str = "embeddings/processed_qa_pairs.pkl"
    PROCESSED_CHUNKS_PATH: str = "embeddings/processed_chunks.pkl"
    EVALUATION_DIR: str = "evaluation"
    COMPARISON_RESULTS_PATH: str = "evaluation/comparison_results.json"
    ABLATION_RESULTS_PATH: str = "evaluation/ablation_results.json"
    HALLUCINATION_ENCODER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LEARNED_HALLUCINATION_MODEL_PATH: str = "models/learned_hallucination.pt"
    LEARNED_HALLUCINATION_BEST_PATH: str = "models/learned_hallucination_best.pt"
    LEARNED_HALLUCINATION_LAST_PATH: str = "models/learned_hallucination_last.pt"
    LEARNED_HALLUCINATION_CALIBRATION_PATH: str = "models/learned_hallucination_calibration.json"
    HALLUCINATION_SYNTHETIC_DATA_PATH: str = "data/hallucination_synthetic.jsonl"
    HALLUCINATION_TRAIN_SIZE: int = 20000
    # Optional small subset for debugging/scaling studies only; not used in main comparisons.
    HALLUCINATION_BASELINE_SIZE: int = 4000
    HALLUCINATION_TRAIN_20K_PATH: str = "data/hallucination_train_20k.json"
    HALLUCINATION_TRAIN_4K_PATH: str = "data/hallucination_train_4k.json"
    RESEARCH_RESULTS_PATH: str = "evaluation/research_results.json"
    PAPER_TABLES_PATH: str = "evaluation/paper_tables.md"
    TRAIN_FROM_SCRATCH: bool = True


def setup_reproducibility(seed: int = 42) -> None:
    if seed is None:
        seed = Config().RANDOM_SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "parr_mhqa.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

HALLUCINATION_THRESHOLD = Config().HALLUCINATION_THRESHOLD