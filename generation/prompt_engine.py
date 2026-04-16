from typing import List

from retrieval.document_store import RetrievalResult


class PromptEngine:
    def format_context(self, results: List[RetrievalResult], max_chars: int = 3000) -> List[str]:
        ordered = sorted(results, key=lambda item: item.rank)
        context_texts: List[str] = []
        total_chars = 0

        for idx, item in enumerate(ordered, start=1):
            passage = f"Passage [{idx}] (from: {item.title}): {item.text}"

            if total_chars + len(passage) > max_chars:
                if not context_texts:
                    context_texts.append(passage[:max_chars])
                break

            context_texts.append(passage)
            total_chars += len(passage)

        return context_texts

    def build_direct_prompt(self, query, context_texts: List[str]) -> str:
        context = "\n\n".join(context_texts)
        return (
            "You are a precise QA assistant. Answer concisely using context only. "
            "Use 1-3 words or a short phrase if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def build_chain_of_thought_prompt(self, query, context_texts: List[str]) -> str:
        context = "\n\n".join(context_texts)
        return (
            "You are a multi-hop QA assistant. Think step by step using only the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Output:\n"
            "Step 1: [find relevant fact from passage X]\n"
            "Step 2: [connect with fact from passage Y]\n"
            "Answer: [final concise answer]"
        )

    def build_critique_prompt(self, query, answer, context_texts: List[str]) -> str:
        context = "\n\n".join(context_texts)
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer: {answer}\n\n"
            "Is every claim in this answer directly supported by the context? "
            "List any unsupported claims. End with exactly one of:\n"
            "FULLY_SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED"
        )

    def build_refinement_prompt(self, query, answer, critique, context_texts) -> str:
        context = "\n\n".join(context_texts)
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Original Answer: {answer}\n"
            f"Critique: {critique}\n\n"
            "Produce an improved answer that addresses the critique and stays grounded in the context.\n"
            "Answer:"
        )
