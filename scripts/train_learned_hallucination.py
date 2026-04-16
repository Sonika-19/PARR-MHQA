import argparse
import logging
import os
from collections import defaultdict

import torch
from sklearn.metrics import f1_score

from config import Config, setup_logging, setup_reproducibility
from data.dataset_loader import HotpotQALoader
from generation.learned_hallucination_model import (
    LearnedHallucinationDetector,
    build_synthetic_hallucination_dataset,
    load_synthetic_dataset,
    save_synthetic_dataset,
    stratified_split_samples,
)


def _title_to_contexts(chunks):
    mapping = defaultdict(list)
    for chunk in chunks:
        title = str(chunk.get("title", "")).strip()
        text = str(chunk.get("text", "")).strip()
        if not title or not text:
            continue
        if len(mapping[title]) < 5:
            mapping[title].append(text)
    return dict(mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned hallucination detector")
    parser.add_argument("--dataset_size", type=int, choices=[20000], default=20000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = Config()
    setup_reproducibility(cfg.RANDOM_SEED)
    setup_logging(cfg.LOG_DIR)
    logging.getLogger().setLevel(logging.ERROR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    loader = HotpotQALoader()
    qa_pairs = loader.load_processed(cfg.PROCESSED_QA_PATH)
    chunks = loader.load_processed(cfg.PROCESSED_CHUNKS_PATH)

    if os.path.exists(cfg.HALLUCINATION_TRAIN_20K_PATH):
        full_samples = load_synthetic_dataset(cfg.HALLUCINATION_TRAIN_20K_PATH)
        print(f"Loaded fixed dataset from: {cfg.HALLUCINATION_TRAIN_20K_PATH}")
    else:
        print("Generating dataset (first run only)...")
        title_map = _title_to_contexts(chunks)
        full_samples = build_synthetic_hallucination_dataset(
            qa_pairs=qa_pairs,
            title_to_contexts=title_map,
            max_samples=cfg.HALLUCINATION_TRAIN_SIZE,
            seed=cfg.RANDOM_SEED,
        )
        save_synthetic_dataset(full_samples, cfg.HALLUCINATION_TRAIN_20K_PATH, seed=cfg.RANDOM_SEED)

    if len(full_samples) != cfg.HALLUCINATION_TRAIN_SIZE:
        raise RuntimeError(
            f"Expected {cfg.HALLUCINATION_TRAIN_SIZE} samples, got {len(full_samples)}"
        )

    baseline_samples = full_samples[: cfg.HALLUCINATION_BASELINE_SIZE]

    save_synthetic_dataset(full_samples, cfg.HALLUCINATION_TRAIN_20K_PATH, seed=cfg.RANDOM_SEED)
    save_synthetic_dataset(baseline_samples, cfg.HALLUCINATION_TRAIN_4K_PATH, seed=cfg.RANDOM_SEED)
    save_synthetic_dataset(full_samples, cfg.HALLUCINATION_SYNTHETIC_DATA_PATH, seed=cfg.RANDOM_SEED)

    samples = full_samples
    train_samples, val_samples = stratified_split_samples(samples, train_ratio=0.8, seed=cfg.RANDOM_SEED)

    grounded_train = sum(1 for sample in train_samples if sample.grounded_label == 1)
    grounded_val = sum(1 for sample in val_samples if sample.grounded_label == 1)

    detector = LearnedHallucinationDetector(cfg, train_from_scratch=cfg.TRAIN_FROM_SCRATCH)
    detector.model = detector.model.to(device)
    detector.calibrator = detector.calibrator.to(device)
    history = detector.fit(
        train_samples=train_samples,
        val_samples=val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    y_true = []
    y_pred = []
    for sample in val_samples:
        pred = detector.predict(sample.query, sample.answer, sample.contexts)
        hall_prob = float(pred.get("hallucinated_prob", 0.0))
        y_true.append(0 if int(sample.grounded_label) == 1 else 1)
        y_pred.append(1 if hall_prob >= 0.5 else 0)

    val_f1 = float(f1_score(y_true, y_pred, zero_division=0)) if y_true else 0.0
    hallucination_rate = float(sum(y_pred) / len(y_pred)) if y_pred else 0.0

    print("Learned hallucination training complete.")
    print(f"Dataset size used: {cfg.HALLUCINATION_TRAIN_SIZE}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Train grounded: {grounded_train} | Train hallucinated: {len(train_samples) - grounded_train}")
    print(f"Val grounded: {grounded_val} | Val hallucinated: {len(val_samples) - grounded_val}")
    print(f"Validation F1: {val_f1:.4f}")
    print(f"Validation hallucination rate: {hallucination_rate:.4f}")
    for key, value in history.items():
        print(f"{key}: {value:.4f}")
    best_epoch = int(history.get("best_epoch", -1.0))
    best_val_loss = float(history.get("best_val_loss", 0.0))
    best_val_ece = float(history.get("best_val_ece", 0.0))
    print(f"Best model selected from epoch {best_epoch} with val_loss={best_val_loss:.4f}")
    print(f"val_loss = {best_val_loss:.4f}")
    print(f"val_ece = {best_val_ece:.4f}")
    print(f"Model: {cfg.LEARNED_HALLUCINATION_MODEL_PATH}")
    print(f"Best checkpoint: {cfg.LEARNED_HALLUCINATION_BEST_PATH}")
    print(f"Last checkpoint: {cfg.LEARNED_HALLUCINATION_LAST_PATH}")
    print(f"Calibration: {cfg.LEARNED_HALLUCINATION_CALIBRATION_PATH}")
    print(f"Saved: {cfg.HALLUCINATION_TRAIN_20K_PATH}")
    print(f"Saved: {cfg.HALLUCINATION_TRAIN_4K_PATH}")
    print("Main results comparison is generated in research evaluation (heuristic vs learned on 20K protocol).")


if __name__ == "__main__":
    main()
