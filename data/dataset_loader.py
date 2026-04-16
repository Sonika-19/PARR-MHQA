import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm


class HotpotQALoader:
    def load(self, split: str = "train") -> Dataset:
        return load_dataset("hotpot_qa", "fullwiki", split=split)

    def extract_documents(
        self,
        dataset: Dataset,
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, str]]:
        documents: List[Dict[str, str]] = []
        doc_counter = 0

        iterator = dataset
        if show_progress:
            iterator = tqdm(dataset, desc="Extracting documents", dynamic_ncols=True, mininterval=0.2)

        for item in iterator:
            if max_samples is not None and len(documents) >= max_samples:
                break

            context = item.get("context", [])

            # Support both list-based context [[title, sentences], ...]
            # and dict-based context {"title": [...], "sentences": [[...], ...]}.
            if isinstance(context, dict):
                titles = context.get("title", [])
                sentences_groups = context.get("sentences", [])
                context_pairs = zip(titles, sentences_groups)
            else:
                context_pairs = context

            for title_idx, pair in enumerate(context_pairs):
                if max_samples is not None and len(documents) >= max_samples:
                    break

                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue

                title, sentences = pair
                if not isinstance(sentences, list):
                    continue

                text = " ".join(str(sentence) for sentence in sentences)
                documents.append(
                    {
                        "title": str(title),
                        "text": text,
                        "doc_id": f"{title}_{doc_counter}",
                    }
                )
                doc_counter += 1

        return documents

    def extract_qa_pairs(self, dataset: Dataset, show_progress: bool = True) -> List[Dict[str, Any]]:
        qa_pairs: List[Dict[str, Any]] = []

        iterator = dataset
        if show_progress:
            iterator = tqdm(dataset, desc="Extracting QA pairs", dynamic_ncols=True, mininterval=0.2)

        for item in iterator:
            supporting_facts_raw = item.get("supporting_facts", [])

            if isinstance(supporting_facts_raw, dict):
                titles = supporting_facts_raw.get("title", [])
                sent_ids = supporting_facts_raw.get("sent_id", [])
                supporting_facts = [(title, sent_id) for title, sent_id in zip(titles, sent_ids)]
            else:
                supporting_facts = []
                for fact in supporting_facts_raw:
                    if isinstance(fact, (list, tuple)) and len(fact) == 2:
                        supporting_facts.append((fact[0], fact[1]))

            qa_pairs.append(
                {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "supporting_facts": supporting_facts,
                    "type": item.get("type", ""),
                    "level": item.get("level", ""),
                }
            )

        return qa_pairs

    def chunk_documents(
        self,
        documents: List[Dict[str, str]],
        chunk_size: int = 300,
        overlap: int = 50,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        chunks: List[Dict[str, Any]] = []
        stride = chunk_size - overlap

        iterator = documents
        if show_progress:
            iterator = tqdm(documents, desc="Chunking documents", dynamic_ncols=True, mininterval=0.2)

        for doc in iterator:
            words = doc.get("text", "").split()
            doc_id = doc.get("doc_id", "")
            title = doc.get("title", "")

            if not words:
                continue

            chunk_index = 0
            for start in range(0, len(words), stride):
                end = start + chunk_size
                chunk_words = words[start:end]
                if not chunk_words:
                    continue

                chunk_text = " ".join(chunk_words)
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                        "doc_id": doc_id,
                        "title": title,
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

                if end >= len(words):
                    break

        return chunks

    def save_processed(self, data: Any, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "wb") as file:
            pickle.dump(data, file)

    def load_processed(self, path: str) -> Any:
        with open(path, "rb") as file:
            return pickle.load(file)


if __name__ == "__main__":
    loader = HotpotQALoader()

    print("Loading HotpotQA train split...")
    train_dataset = loader.load(split="train")

    print("Extracting first 50000 documents from context...")
    documents = loader.extract_documents(train_dataset, max_samples=50000)

    print("Extracting all QA pairs...")
    qa_pairs = loader.extract_qa_pairs(train_dataset)

    print("Chunking documents...")
    chunks = loader.chunk_documents(documents, chunk_size=300, overlap=50)

    docs_path = "embeddings/processed_documents.pkl"
    qa_path = "embeddings/processed_qa_pairs.pkl"
    chunks_path = "embeddings/processed_chunks.pkl"

    loader.save_processed(documents, docs_path)
    loader.save_processed(qa_pairs, qa_path)
    loader.save_processed(chunks, chunks_path)

    avg_chunk_len = (sum(len(chunk["text"].split()) for chunk in chunks) / len(chunks)) if chunks else 0.0

    type_counter = Counter(item.get("type", "") for item in qa_pairs)
    level_counter = Counter(item.get("level", "") for item in qa_pairs)

    print("\nProcessing complete.")
    print(f"Total docs: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk length: {avg_chunk_len:.2f} words")
    print(
        "Type distribution: "
        f"bridge={type_counter.get('bridge', 0)}, "
        f"comparison={type_counter.get('comparison', 0)}"
    )
    print(
        "Level distribution: "
        f"easy={level_counter.get('easy', 0)}, "
        f"medium={level_counter.get('medium', 0)}, "
        f"hard={level_counter.get('hard', 0)}"
    )
