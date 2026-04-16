import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    rank: int
    retrieval_stage: str = "initial"


class DocumentStore:
    def __init__(self) -> None:
        self.chunks: List[dict] = []
        self.index_to_chunk: Dict[int, dict] = {}

    def build(self, chunks: List[dict]) -> None:
        self.chunks = list(chunks)
        self.index_to_chunk = {idx: chunk for idx, chunk in enumerate(self.chunks)}

    def get(self, index: int) -> Optional[dict]:
        return self.index_to_chunk.get(index)

    def get_batch(self, indices: List[int]) -> List[Optional[dict]]:
        return [self.get(index) for index in indices]

    def get_text(self, index: int) -> str:
        chunk = self.get(index)
        if not chunk:
            return ""
        return str(chunk.get("text", ""))

    def get_texts(self, indices: List[int]) -> List[str]:
        return [self.get_text(index) for index in indices]

    def search_by_title(self, title: str) -> List[dict]:
        needle = title.strip().lower()
        if not needle:
            return []

        matches: List[dict] = []
        for chunk in self.chunks:
            chunk_title = str(chunk.get("title", ""))
            if needle in chunk_title.lower():
                matches.append(chunk)
        return matches

    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "index_to_chunk": self.index_to_chunk,
                },
                file,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as file:
            data = pickle.load(file)

        self.chunks = data.get("chunks", [])
        self.index_to_chunk = data.get("index_to_chunk", {})


if __name__ == "__main__":
    chunks_path = "embeddings/processed_chunks.pkl"

    with open(chunks_path, "rb") as file:
        chunks = pickle.load(file)

    store = DocumentStore()
    store.build(chunks)

    print("DocumentStore built successfully.")
    print(f"Total chunks: {len(store.chunks)}")
    print("Sample entries:")

    for sample_index in range(min(3, len(store.chunks))):
        sample = store.get(sample_index)
        print(f"\nIndex: {sample_index}")
        print(f"Chunk ID: {sample.get('chunk_id', '')}")
        print(f"Doc ID: {sample.get('doc_id', '')}")
        print(f"Title: {sample.get('title', '')}")
        print(f"Text preview: {sample.get('text', '')[:120]}...")
