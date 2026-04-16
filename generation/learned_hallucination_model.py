import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import Config


@dataclass
class HallucinationSample:
    """Single supervision tuple used for learned grounding detection."""

    query: str
    answer: str
    contexts: List[str]
    grounded_label: int
    corruption_type: str


class HallucinationTrainingDataset(Dataset):
    def __init__(self, samples: List[HallucinationSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> HallucinationSample:
        return self.samples[index]


class LearnedHallucinationModel(nn.Module):
    """Novel learned detector: answer-context cross-attention + token support scoring.

    The model is intentionally modular so each component maps to a claimable paper
    contribution: token support attribution, calibrated grounding probability, and
    contrastive grounding separation.
    """

    def __init__(self, encoder_model: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        hidden = self.encoder.config.hidden_size
        self.cross_attention = nn.MultiheadAttention(hidden, num_heads=8, batch_first=True)
        self.token_support_head = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 6, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _masked_mean(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    def forward(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        query_h = self.encoder(input_ids=query_ids, attention_mask=query_mask).last_hidden_state
        answer_h = self.encoder(input_ids=answer_ids, attention_mask=answer_mask).last_hidden_state
        context_h = self.encoder(input_ids=context_ids, attention_mask=context_mask).last_hidden_state

        key_padding_mask = context_mask == 0
        attn_out, _ = self.cross_attention(
            query=answer_h,
            key=context_h,
            value=context_h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        token_support_logits = self.token_support_head(attn_out).squeeze(-1)
        support_mask = answer_mask == 0
        token_support_logits = token_support_logits.masked_fill(support_mask, -1e4)
        token_support_weights = torch.softmax(token_support_logits, dim=-1)

        pooled_query = self._masked_mean(query_h, query_mask)
        pooled_answer = self._masked_mean(answer_h, answer_mask)
        pooled_context = self._masked_mean(context_h, context_mask)
        pooled_cross = torch.sum(attn_out * token_support_weights.unsqueeze(-1), dim=1)

        features = torch.cat(
            [
                pooled_query,
                pooled_answer,
                pooled_context,
                pooled_cross,
                torch.abs(pooled_answer - pooled_context),
                pooled_answer * pooled_context,
            ],
            dim=-1,
        )

        grounded_logit = self.classifier(features).squeeze(-1)
        return {
            "grounded_logit": grounded_logit,
            "token_support_logits": token_support_logits,
            "pooled_answer": pooled_answer,
            "pooled_context": pooled_context,
        }


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature).clamp_min(1e-4)
        return logits / temperature

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).detach().cpu().item())


class LearnedHallucinationDetector:
    """Inference/training wrapper around the learned hallucination architecture."""

    ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

    def __init__(self, config: Config, train_from_scratch: bool = True):
        self.config = config
        self.train_from_scratch = bool(train_from_scratch)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.HALLUCINATION_ENCODER_MODEL)
        self.model = LearnedHallucinationModel(config.HALLUCINATION_ENCODER_MODEL).to(self.device)
        self.calibrator = TemperatureScaler().to(self.device)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _join_context(contexts: List[str]) -> str:
        return "\n\n".join([str(ctx) for ctx in contexts if str(ctx).strip()])

    def _encode_triplet(self, query: str, answer: str, contexts: List[str]) -> Dict[str, torch.Tensor]:
        context_text = self._join_context(contexts)
        q = self.tokenizer(query, truncation=True, max_length=96, return_tensors="pt")
        a = self.tokenizer(answer, truncation=True, max_length=128, return_tensors="pt")
        c = self.tokenizer(context_text, truncation=True, max_length=384, return_tensors="pt")

        return {
            "query_ids": q["input_ids"].to(self.device),
            "query_mask": q["attention_mask"].to(self.device),
            "answer_ids": a["input_ids"].to(self.device),
            "answer_mask": a["attention_mask"].to(self.device),
            "context_ids": c["input_ids"].to(self.device),
            "context_mask": c["attention_mask"].to(self.device),
        }

    def predict(self, query: str, answer: str, contexts: List[str]) -> Dict[str, float | List[float]]:
        self.model.eval()
        self.calibrator.eval()

        with torch.no_grad():
            encoded = self._encode_triplet(query, answer, contexts)
            out = self.model(**encoded)
            calibrated_logit = self.calibrator(out["grounded_logit"])
            grounded_prob = torch.sigmoid(calibrated_logit).item()
            hallucinated_prob = 1.0 - grounded_prob
            token_support = torch.sigmoid(out["token_support_logits"]).squeeze(0).cpu().tolist()

        # Compatibility factors for current refinement policy.
        entity_mismatch = self._estimate_entity_mismatch(answer, contexts)
        semantic_sim = self._estimate_semantic_proxy(out["pooled_answer"], out["pooled_context"])
        unsupported_ratio = float(np.mean([1.0 - min(max(score, 0.0), 1.0) for score in token_support]))

        return {
            "hallucinated_prob": float(hallucinated_prob),
            "grounded_prob": float(grounded_prob),
            "token_support": [float(v) for v in token_support],
            "entity_mismatch": float(entity_mismatch),
            "semantic_sim": float(semantic_sim),
            "unsupported_ratio": float(unsupported_ratio),
        }

    @staticmethod
    def _estimate_semantic_proxy(answer_vec: torch.Tensor, context_vec: torch.Tensor) -> float:
        num = torch.sum(answer_vec * context_vec, dim=-1)
        den = torch.norm(answer_vec, dim=-1) * torch.norm(context_vec, dim=-1)
        cos = (num / den.clamp_min(1e-6)).detach().cpu().item()
        return float((cos + 1.0) / 2.0)

    def _estimate_entity_mismatch(self, answer: str, contexts: List[str]) -> float:
        answer_entities = {m.group(0).lower() for m in self.ENTITY_PATTERN.finditer(answer or "")}
        context_entities = {m.group(0).lower() for m in self.ENTITY_PATTERN.finditer(self._join_context(contexts))}
        if not answer_entities:
            return 0.0
        missing = answer_entities - context_entities
        return float(len(missing) / len(answer_entities))

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), target)

    def save_checkpoint(self, path: str, epoch: int, val_loss: float, val_ece: float) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "epoch": int(epoch),
                "val_loss": float(val_loss),
                "val_ece": float(val_ece),
            },
            target,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)
        self.logger.info("Loaded learned hallucination model from %s", path)

    def save_calibration(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"temperature": self.calibrator.temperature}
        with open(target, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=True)

    def load_calibration(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        temperature = float(payload.get("temperature", 1.0))
        self.calibrator.log_temperature.data = torch.tensor([np.log(max(temperature, 1e-4))], device=self.device)

    def _batch_collate(self, batch: List[HallucinationSample]) -> Dict[str, torch.Tensor]:
        queries = [sample.query for sample in batch]
        answers = [sample.answer for sample in batch]
        contexts = [self._join_context(sample.contexts) for sample in batch]
        labels = torch.tensor([sample.grounded_label for sample in batch], dtype=torch.float32, device=self.device)

        q = self.tokenizer(queries, padding=True, truncation=True, max_length=96, return_tensors="pt")
        a = self.tokenizer(answers, padding=True, truncation=True, max_length=128, return_tensors="pt")
        c = self.tokenizer(contexts, padding=True, truncation=True, max_length=384, return_tensors="pt")

        return {
            "query_ids": q["input_ids"].to(self.device),
            "query_mask": q["attention_mask"].to(self.device),
            "answer_ids": a["input_ids"].to(self.device),
            "answer_mask": a["attention_mask"].to(self.device),
            "context_ids": c["input_ids"].to(self.device),
            "context_mask": c["attention_mask"].to(self.device),
            "labels": labels,
        }

    def _contrastive_loss(self, pooled_answer: torch.Tensor, pooled_context: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sims = F.cosine_similarity(pooled_answer, pooled_context, dim=-1)
        pos = sims[labels > 0.5]
        neg = sims[labels <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        margin = 0.2
        return F.relu(margin - (pos.mean() - neg.mean()))

    def fit(
        self,
        train_samples: List[HallucinationSample],
        val_samples: List[HallucinationSample],
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        contrastive_weight: float = 0.5,
    ) -> Dict[str, float]:
        if self.train_from_scratch:
            print("Training from scratch (no previous weights loaded)")

        train_loader = DataLoader(
            HallucinationTrainingDataset(train_samples),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda items: items,
        )
        val_loader = DataLoader(
            HallucinationTrainingDataset(val_samples),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda items: items,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        bce = nn.BCEWithLogitsLoss()

        best_val = float("inf")
        best_epoch = -1
        best_val_ece = 0.0
        history: Dict[str, float] = {}

        for epoch in range(epochs):
            print(f"\n[TRAIN] Epoch {epoch + 1}/{epochs}")
            self.model.train()
            train_losses: List[float] = []

            pbar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}",
                leave=True,
                dynamic_ncols=True,
            )

            for batch in train_loader:
                collated = self._batch_collate(batch)
                optimizer.zero_grad(set_to_none=True)

                out = self.model(
                    query_ids=collated["query_ids"],
                    query_mask=collated["query_mask"],
                    answer_ids=collated["answer_ids"],
                    answer_mask=collated["answer_mask"],
                    context_ids=collated["context_ids"],
                    context_mask=collated["context_mask"],
                )

                labels = collated["labels"]
                bce_loss = bce(out["grounded_logit"], labels)
                c_loss = self._contrastive_loss(out["pooled_answer"], out["pooled_context"], labels)
                loss = bce_loss + contrastive_weight * c_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_value = float(loss.detach().cpu().item())
                train_losses.append(loss_value)
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss_value:.4f}"})

            pbar.close()

            val_loss, val_ece = self._evaluate_for_calibration(val_loader)
            history[f"epoch_{epoch + 1}_train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
            history[f"epoch_{epoch + 1}_val_loss"] = val_loss
            history[f"epoch_{epoch + 1}_val_ece"] = val_ece
            print(f"[VAL] Epoch {epoch + 1} | val_loss={val_loss:.4f} | val_ece={val_ece:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                best_val_ece = val_ece
                self.save_checkpoint(
                    self.config.LEARNED_HALLUCINATION_BEST_PATH,
                    epoch=best_epoch,
                    val_loss=val_loss,
                    val_ece=val_ece,
                )

        # Save last epoch model separately.
        self.save(self.config.LEARNED_HALLUCINATION_LAST_PATH)

        # Use best model for calibration/evaluation by default.
        if Path(self.config.LEARNED_HALLUCINATION_BEST_PATH).exists():
            self.load(self.config.LEARNED_HALLUCINATION_BEST_PATH)

        # Keep canonical model path as best model weights for compatibility.
        self.save(self.config.LEARNED_HALLUCINATION_MODEL_PATH)

        self.calibrate(val_loader)
        self.save_calibration(self.config.LEARNED_HALLUCINATION_CALIBRATION_PATH)
        history["best_epoch"] = float(best_epoch)
        history["best_val_loss"] = float(best_val)
        history["best_val_ece"] = float(best_val_ece)
        return history

    def _evaluate_for_calibration(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        bce = nn.BCEWithLogitsLoss()
        losses: List[float] = []
        probs: List[float] = []
        labels: List[int] = []

        with torch.no_grad():
            for batch in loader:
                collated = self._batch_collate(batch)
                out = self.model(
                    query_ids=collated["query_ids"],
                    query_mask=collated["query_mask"],
                    answer_ids=collated["answer_ids"],
                    answer_mask=collated["answer_mask"],
                    context_ids=collated["context_ids"],
                    context_mask=collated["context_mask"],
                )
                loss = bce(out["grounded_logit"], collated["labels"])
                losses.append(float(loss.detach().cpu().item()))
                batch_probs = torch.sigmoid(out["grounded_logit"]).detach().cpu().numpy().tolist()
                probs.extend([float(p) for p in batch_probs])
                labels.extend([int(v) for v in collated["labels"].detach().cpu().numpy().tolist()])

        return (float(np.mean(losses)) if losses else 0.0, expected_calibration_error(probs, labels))

    def calibrate(self, loader: DataLoader) -> None:
        self.model.eval()
        logits: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                collated = self._batch_collate(batch)
                out = self.model(
                    query_ids=collated["query_ids"],
                    query_mask=collated["query_mask"],
                    answer_ids=collated["answer_ids"],
                    answer_mask=collated["answer_mask"],
                    context_ids=collated["context_ids"],
                    context_mask=collated["context_mask"],
                )
                logits.append(out["grounded_logit"])
                labels.append(collated["labels"])

        if not logits:
            return

        stacked_logits = torch.cat(logits)
        stacked_labels = torch.cat(labels)

        optimizer = torch.optim.LBFGS([self.calibrator.log_temperature], lr=0.1, max_iter=50)
        criterion = nn.BCEWithLogitsLoss()

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            calibrated = self.calibrator(stacked_logits)
            loss = criterion(calibrated, stacked_labels)
            loss.backward()
            return loss

        optimizer.step(_closure)


def expected_calibration_error(probs: List[float], labels: List[int], n_bins: int = 10) -> float:
    if not probs:
        return 0.0
    probs_arr = np.asarray(probs, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]
        if i == n_bins - 1:
            mask = (probs_arr >= lower) & (probs_arr <= upper)
        else:
            mask = (probs_arr >= lower) & (probs_arr < upper)
        if not np.any(mask):
            continue

        conf = float(np.mean(probs_arr[mask]))
        acc = float(np.mean(labels_arr[mask]))
        frac = float(np.mean(mask.astype(np.float32)))
        ece += abs(acc - conf) * frac

    return float(ece)


def _swap_entity(text: str, replacement_pool: List[str], rng: random.Random) -> str:
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    if not entities:
        return text + " near the Atlantic Ocean"
    target = rng.choice(entities)
    replacement = rng.choice(replacement_pool) if replacement_pool else "Unknown City"
    return text.replace(target, replacement, 1)


def _contradict_fact(text: str) -> str:
    if " is " in text:
        return text.replace(" is ", " is not ", 1)
    if " was " in text:
        return text.replace(" was ", " was not ", 1)
    return "It is false that " + text


def _unsupported_extension(text: str) -> str:
    return text + " It also won the Nobel Prize in Chemistry."


def _partial_grounding(text: str) -> str:
    half = max(1, len(text.split()) // 2)
    prefix = " ".join(text.split()[:half])
    return prefix + " and later became the capital of Mars Colony One."


def build_synthetic_hallucination_dataset(
    qa_pairs: List[dict],
    title_to_contexts: Dict[str, List[str]],
    max_samples: int = 20000,
    seed: int = 42,
) -> List[HallucinationSample]:
    """Create a balanced dataset with equal grounded/hallucinated classes.

    Hallucinated samples are balanced across corruption types to reduce synthetic bias.
    """

    if max_samples <= 0 or max_samples % 2 != 0:
        raise ValueError("max_samples must be a positive even integer")

    rng = random.Random(seed)
    all_titles = list(title_to_contexts.keys())

    target_size = max_samples // 2
    grounded_pool: List[HallucinationSample] = []
    hallucinated_by_type: Dict[str, List[HallucinationSample]] = {
        "entity_swap": [],
        "fact_contradiction": [],
        "unsupported_claim": [],
        "partial_grounding": [],
    }

    min_per_type = target_size // 4

    for item in tqdm(qa_pairs, desc="Generating synthetic hallucinations", leave=True, dynamic_ncols=True, mininterval=0.2):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue

        supporting = [str(sf[0]) for sf in item.get("supporting_facts", []) if sf]
        contexts: List[str] = []
        for title in supporting:
            contexts.extend(title_to_contexts.get(title, [])[:1])
        if not contexts:
            continue

        grounded_pool.append(
            HallucinationSample(
                query=question,
                answer=answer,
                contexts=contexts[:3],
                grounded_label=1,
                corruption_type="grounded",
            )
        )

        replacement_pool = rng.sample(all_titles, k=min(20, len(all_titles))) if all_titles else []
        corruptions = [
            ("entity_swap", _swap_entity(answer, replacement_pool, rng)),
            ("fact_contradiction", _contradict_fact(answer)),
            ("unsupported_claim", _unsupported_extension(answer)),
            ("partial_grounding", _partial_grounding(answer)),
        ]

        for corruption_type, corrupted_answer in corruptions:
            hallucinated_by_type[corruption_type].append(
                HallucinationSample(
                    query=question,
                    answer=corrupted_answer,
                    contexts=contexts[:3],
                    grounded_label=0,
                    corruption_type=corruption_type,
                )
            )

        if len(grounded_pool) >= target_size and all(len(hallucinated_by_type[key]) >= min_per_type for key in hallucinated_by_type):
            break

    if len(grounded_pool) < target_size:
        raise ValueError(
            f"Not enough grounded samples for balanced dataset: {len(grounded_pool)} available, {target_size} required"
        )

    grounded_selected = grounded_pool[:]
    rng.shuffle(grounded_selected)
    grounded_selected = grounded_selected[:target_size]

    hallucinated_selected: List[HallucinationSample] = []
    for key in ["entity_swap", "fact_contradiction", "unsupported_claim", "partial_grounding"]:
        bucket = hallucinated_by_type.get(key, [])[:]
        rng.shuffle(bucket)
        if len(bucket) < min_per_type:
            raise ValueError(f"Insufficient {key} samples: {len(bucket)} available, {min_per_type} required")
        hallucinated_selected.extend(bucket[:min_per_type])

    remaining = target_size - len(hallucinated_selected)
    if remaining > 0:
        extras: List[HallucinationSample] = []
        for key in ["entity_swap", "fact_contradiction", "unsupported_claim", "partial_grounding"]:
            extras.extend(hallucinated_by_type[key][min_per_type:])
        rng.shuffle(extras)
        if len(extras) < remaining:
            raise ValueError("Insufficient hallucinated samples to fill target class size")
        hallucinated_selected.extend(extras[:remaining])

    dataset = grounded_selected + hallucinated_selected
    rng.shuffle(dataset)
    return dataset


def stratified_split_samples(
    samples: List[HallucinationSample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[HallucinationSample], List[HallucinationSample]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    rng = random.Random(seed)
    grounded = [sample for sample in samples if int(sample.grounded_label) == 1]
    hallucinated = [sample for sample in samples if int(sample.grounded_label) == 0]

    rng.shuffle(grounded)
    rng.shuffle(hallucinated)

    g_split = int(len(grounded) * train_ratio)
    h_split = int(len(hallucinated) * train_ratio)

    train_data = grounded[:g_split] + hallucinated[:h_split]
    val_data = grounded[g_split:] + hallucinated[h_split:]

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    return train_data, val_data


def save_synthetic_dataset(samples: List[HallucinationSample], path: str, seed: int = 42) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    grounded = sum(1 for sample in samples if int(sample.grounded_label) == 1)
    hallucinated = sum(1 for sample in samples if int(sample.grounded_label) == 0)

    payload = {
        "total_samples": int(len(samples)),
        "grounded": int(grounded),
        "hallucinated": int(hallucinated),
        "seed": int(seed),
        "samples": [
            {
                "query": sample.query,
                "answer": sample.answer,
                "contexts": sample.contexts,
                "grounded_label": int(sample.grounded_label),
                "corruption_type": sample.corruption_type,
            }
            for sample in samples
        ],
    }

    with open(target, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)


def load_synthetic_dataset(path: str) -> List[HallucinationSample]:
    samples: List[HallucinationSample] = []
    with open(path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        return samples

    # Preferred format: JSON object with metadata + samples.
    if content.startswith("{"):
        parsed = json.loads(content)
        rows = parsed.get("samples", []) if isinstance(parsed, dict) else []
    else:
        # Backward compatibility: JSONL format.
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]

    for row in rows:
        samples.append(
            HallucinationSample(
                query=str(row.get("query", "")),
                answer=str(row.get("answer", "")),
                contexts=[str(v) for v in row.get("contexts", [])],
                grounded_label=int(row.get("grounded_label", 0)),
                corruption_type=str(row.get("corruption_type", "unknown")),
            )
        )
    return samples
