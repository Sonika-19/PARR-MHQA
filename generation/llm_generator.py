import hashlib
import logging
import re
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, pipeline

from config import Config
from generation.prompt_engine import PromptEngine


class LLMGenerator:
    def __init__(self, config: Config, mode: str = "eval"):
        self.config = config
        self.prompt_engine = PromptEngine()
        self.logger = logging.getLogger(__name__)
        self.mode = str(mode).lower()

        model_name = config.HF_MODEL or "microsoft/phi-2"

        self.model = model_name
        self.temperature = 0.1
        self._cache: Dict[str, str] = {}
        self._cache_hits = 0
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            device=self.device,
            dtype=torch.float16 if self.device == 0 else torch.float32,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Using LOCAL model: {model_name} (mode={self.mode})")

    @staticmethod
    def _cache_key(prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def generate(self, prompt, max_tokens: int = 300, temperature: Optional[float] = None) -> str:
        prompt_text = str(prompt)
        inputs = self.tokenizer(
            prompt_text,
            truncation=True,
            return_tensors="pt",
        )
        if self.device == 0:
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

        key = self._cache_key(prompt_text)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        _ = temperature
        max_new_tokens = 128

        outputs = self.generator(
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=False,
            return_full_text=False,
        )
        content = str(outputs[0]["generated_text"]).strip()
        if self.device == 0:
            torch.cuda.empty_cache()
        self._cache[key] = content
        self.logger.info("generate | provider=local_hf | len=%s", len(content))
        return content

    def generate_answer(self, prompt, max_tokens: int = 500, temperature: Optional[float] = None) -> str:
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def generate_critique(self, prompt, max_tokens: int = 500, temperature: Optional[float] = None) -> str:
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def generate_refinement(self, prompt, max_tokens: int = 500, temperature: Optional[float] = None) -> str:
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def extract_final_answer(self, text: str) -> str:
        marker = "Answer:"
        idx = text.rfind(marker)
        if idx == -1:
            answer = text.strip()
        else:
            answer = text[idx + len(marker) :].strip()

        answer = re.sub(r"[\s\.,;:]+$", "", answer)
        if answer.endswith("?") or answer.endswith("!"):
            return answer
        return answer


def get_generator(config, mode: str = "eval") -> LLMGenerator:
    return LLMGenerator(config, mode=mode)


if __name__ == "__main__":
    cfg = Config()
    generator = get_generator(cfg)
    print(f"Generator initialized: {generator.__class__.__name__}")
