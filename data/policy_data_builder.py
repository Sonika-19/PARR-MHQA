import re
from typing import Any, Dict, List

import pandas as pd

from data.dataset_loader import HotpotQALoader


class PolicyDataBuilder:
    COMPARISON_CUE_PATTERN = re.compile(
        r"(which|both|compare|versus|differ|same|older|newer|larger|smaller)",
        re.IGNORECASE,
    )
    BRIDGE_CUE_PATTERN = re.compile(
        r"(who|where|when|author|director|born|located|founded)",
        re.IGNORECASE,
    )
    # Count contiguous groups of capitalized words, e.g., "New York City" as one entity group.
    ENTITY_GROUP_PATTERN = re.compile(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)")

    def build_training_data(self, qa_pairs: List[Dict[str, Any]]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for item in qa_pairs:
            support_count = len(item.get("supporting_facts", []))
            q_type = str(item.get("type", "")).lower()
            level = str(item.get("level", "")).lower()
            question = str(item.get("question", ""))

            is_bridge = q_type == "bridge"
            is_comparison = q_type == "comparison"
            is_complex_level = level in ("medium", "hard")
            is_long_question = len(question.split()) >= 12

            is_multi_hop = (
                (is_bridge and support_count >= 2 and is_complex_level)
                or (is_comparison and is_long_question)
            )

            label = 1 if is_multi_hop else 0
            q_length = len(question.split())
            has_comparison_cue = int(bool(self.COMPARISON_CUE_PATTERN.search(question)))
            has_bridge_cue = int(bool(self.BRIDGE_CUE_PATTERN.search(question)))
            entity_count = len(self.ENTITY_GROUP_PATTERN.findall(question))

            rows.append(
                {
                    "question": question,
                    "label": label,
                    "q_length": q_length,
                    "has_comparison_cue": has_comparison_cue,
                    "has_bridge_cue": has_bridge_cue,
                    "entity_count": entity_count,
                    "type": q_type,
                    "level": level,
                }
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "question",
                "label",
                "q_length",
                "has_comparison_cue",
                "has_bridge_cue",
                "entity_count",
                "type",
                "level",
            ],
        )

        print("\nLabel distribution:")
        print(df["label"].value_counts())

        assert df["label"].nunique() > 1, "ERROR: Only one class present!"

        df_0 = df[df["label"] == 0]
        df_1 = df[df["label"] == 1]
        n = min(len(df_0), len(df_1), 20000)
        if n > 0:
            df = pd.concat(
                [
                    df_0.sample(n, random_state=42),
                    df_1.sample(n, random_state=42),
                ]
            ).sample(frac=1.0, random_state=42).reset_index(drop=True)

        return df

    def analyze_distribution(self, df: pd.DataFrame) -> None:
        print("Class balance:")
        print(df["label"].value_counts(dropna=False).sort_index())
        print()

        print("Type breakdown:")
        print(df["type"].value_counts(dropna=False))
        print()

        print("Level breakdown:")
        print(df["level"].value_counts(dropna=False))
        print()

        print("Average q_length per class:")
        print(df.groupby("label")["q_length"].mean())
        print()

        print("Example questions per class (up to 3 each):")
        for label in sorted(df["label"].dropna().unique().tolist()):
            print(f"Class {label}:")
            examples = df[df["label"] == label]["question"].head(3).tolist()
            if not examples:
                print("  (no examples)")
                continue
            for idx, question in enumerate(examples, start=1):
                print(f"  {idx}. {question}")
            print()

    def save(self, df: pd.DataFrame, path: str) -> None:
        df.to_csv(path, index=False)


if __name__ == "__main__":
    loader = HotpotQALoader()
    builder = PolicyDataBuilder()

    qa_pairs = loader.load_processed("embeddings/processed_qa_pairs.pkl")
    df = builder.build_training_data(qa_pairs)
    builder.analyze_distribution(df)
    builder.save(df, "data/policy_train.csv")

    print("Saved training data to data/policy_train.csv")
