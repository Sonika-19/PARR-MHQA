import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from generation.hallucination_detector import HallucinationResult
from generation.prompt_engine import PromptEngine


@dataclass
class CritiqueResult:
    support_level: str
    unsupported_parts: List[str]
    needs_refinement: bool
    critique_text: str


@dataclass
class RefinementAction:
    action: str
    reason: str
    new_k: Optional[int] = None


class SelfCritiqueModule:
    SUPPORT_LEVELS = {"FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"}

    def __init__(self, generator, prompt_engine: PromptEngine):
        self.generator = generator
        self.prompt_engine = prompt_engine
        self.logger = logging.getLogger(__name__)

    def critique(self, query, answer, context_texts) -> CritiqueResult:
        prompt = self.prompt_engine.build_critique_prompt(query, answer, context_texts)
        response = self.generator.generate_critique(prompt, max_tokens=500)

        parsed = self.parse_critique_response(response)
        support_level = parsed["support_level"]
        unsupported_parts = parsed["unsupported_parts"]

        result = CritiqueResult(
            support_level=support_level,
            unsupported_parts=unsupported_parts,
            needs_refinement=(support_level != "FULLY_SUPPORTED"),
            critique_text=response,
        )

        self.logger.info(
            "critique | support_level=%s | unsupported_parts=%s",
            result.support_level,
            result.unsupported_parts,
        )
        return result

    def parse_critique_response(self, response: str) -> dict:
        text = response or ""
        match = re.search(r"\b(FULLY_SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED)\b", text)
        support_level = match.group(1) if match else "PARTIALLY_SUPPORTED"

        unsupported_parts: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if (
                lowered.startswith("unsupported:")
                or lowered.startswith("not supported:")
                or lowered.startswith("missing:")
            ):
                value = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
                if value:
                    unsupported_parts.append(value)

        return {
            "support_level": support_level,
            "unsupported_parts": unsupported_parts,
        }

    def determine_refinement_action(
        self,
        h_result: HallucinationResult,
        c_result: CritiqueResult,
        current_k: int,
    ) -> RefinementAction:
        if c_result.support_level == "NOT_SUPPORTED" and h_result.score > 0.5:
            return RefinementAction(
                action="increase_k_and_rerank",
                reason=(
                    "Critique indicates unsupported answer and hallucination score is high; "
                    "expand retrieval depth and rerank."
                ),
                new_k=current_k + 4,
            )

        if c_result.support_level == "PARTIALLY_SUPPORTED" and h_result.entity_mismatch > 0.3:
            return RefinementAction(
                action="increase_k",
                reason="Entity mismatch is high under partial support; retrieve additional context.",
                new_k=current_k + 2,
            )

        if c_result.support_level == "PARTIALLY_SUPPORTED" and h_result.semantic_sim < 0.45:
            return RefinementAction(
                action="rerank_only",
                reason="Low semantic alignment under partial support; rerank current documents.",
            )

        if c_result.support_level == "NOT_SUPPORTED" or h_result.unsupported_ratio > 0.5:
            return RefinementAction(
                action="switch_to_cot",
                reason="Unsupported claims remain; switch to chain-of-thought prompting for grounded reasoning.",
            )

        return RefinementAction(
            action="none",
            reason="Evidence is sufficiently supported; no refinement action needed.",
        )
