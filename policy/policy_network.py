from dataclasses import dataclass


@dataclass
class PolicyDecision:
    query_type: str
    k: int
    prompt_style: str
    confidence: float
    expand_retrieval: bool
    k_expand: int
