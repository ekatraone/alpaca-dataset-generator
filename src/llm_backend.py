"""
LLM Backend abstraction.

Supports three modes:
  - 'local'  : a local instruct model (default Qwen2.5-1.5B-Instruct) via transformers
  - 'api'    : calls any OpenAI-compatible endpoint (OpenRouter by default)
  - 'legacy' : the original GPT-2 / T5 / DistilBERT pipeline

Usage
-----
# Local (default, no extra deps)
backend = LLMBackend('local', models=models)

# OpenRouter
backend = LLMBackend(
    'api',
    api_key='sk-or-...',
    model='meta-llama/llama-3.1-8b-instruct',  # any OpenRouter model slug
    base_url='https://openrouter.ai/api/v1',    # default; change for other providers
)

output = backend.generate(prompt, max_tokens=200)
"""

import os
import time
from typing import Dict, List, Optional

from config import CONFIG


SYSTEM_PROMPT = (
    "You are a precise assistant that writes training data. "
    "Follow the instruction exactly and reply with ONLY the requested output — "
    "no preamble, no repetition of the instruction or the input text."
)


class LLMBackend:
    def __init__(
        self,
        mode: str = 'local',
        models: Optional[Dict] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            mode:     'local' or 'api'
            models:   dict returned by model_setup.setup_models() — required for 'local'
            api_key:  API key for the endpoint; falls back to OPENROUTER_API_KEY env var
            model:    Model slug (e.g. 'meta-llama/llama-3.1-8b-instruct')
            base_url: API base URL; defaults to OpenRouter
        """
        if mode not in ('local', 'api', 'legacy'):
            raise ValueError(f"mode must be 'local', 'api' or 'legacy', got '{mode}'")

        self.mode = mode

        if mode in ('local', 'legacy'):
            if models is None:
                raise ValueError(f"models dict is required for '{mode}' mode")
            self._models = models

        elif mode == 'api':
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for API mode. "
                    "Install it with: pip install openai"
                )

            resolved_key = api_key or os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENAI_API_KEY')
            if not resolved_key:
                raise ValueError(
                    "API key not found. Pass api_key=, or set the "
                    "OPENROUTER_API_KEY / OPENAI_API_KEY environment variable."
                )

            resolved_url = base_url or CONFIG.get('api_base_url', 'https://openrouter.ai/api/v1')
            resolved_model = model or CONFIG.get('api_model', 'meta-llama/llama-3.1-8b-instruct:free')

            self._client = OpenAI(api_key=resolved_key, base_url=resolved_url)
            self._model = resolved_model
            self._max_retries = CONFIG.get('api_max_retries', 4)
            self._retry_delay = CONFIG.get('api_retry_delay', 2)  # seconds, doubles each attempt

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_api(self) -> bool:
        return self.mode == 'api'

    def is_legacy(self) -> bool:
        return self.mode == 'legacy'

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text for one prompt. Returns the stripped string, or '' on failure."""
        return self.generate_batch([prompt], max_tokens)[0]

    def generate_batch(self, prompts: List[str], max_tokens: int = 200) -> List[str]:
        """Generate text for several prompts (batched on the local model)."""
        if self.mode == 'local':
            return self._generate_local_batch(prompts, max_tokens)
        if self.mode == 'api':
            return [self._generate_api(p, max_tokens) for p in prompts]
        return [self._generate_legacy(p, max_tokens) for p in prompts]

    def generate_sentiment(self, text: str) -> str:
        """Returns '<Positive|Negative|Neutral>. <one-sentence explanation>'."""
        if self.mode == 'legacy':
            from utils import generate_gpt2_output
            truncated = text[:1000]
            sentiment = self._models["sentiment_pipeline"](truncated)[0]
            label = sentiment['label']
            explanation = generate_gpt2_output(
                self._models["gpt2_tokenizer"],
                self._models["gpt2_model"],
                f"The text \"{truncated[:200]}\" has {label} sentiment because",
                CONFIG['device'],
            )
            return f"{label.capitalize()}. {explanation}"
        return self.generate(self.sentiment_prompt(text), max_tokens=80)

    @staticmethod
    def sentiment_prompt(text: str) -> str:
        return (
            "Analyze the sentiment of the following text. "
            "Start your response with exactly one of: Positive, Negative, or Neutral. "
            "Then add a period and one sentence explaining why.\n\n"
            f"Text: {text}\n\nSentiment:"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_legacy(self, prompt: str, max_tokens: int) -> str:
        from utils import generate_gpt2_output
        return generate_gpt2_output(
            self._models["gpt2_tokenizer"],
            self._models["gpt2_model"],
            prompt,
            CONFIG['device'],
            max_length=max_tokens,
        )

    def _generate_local_batch(self, prompts: List[str], max_tokens: int) -> List[str]:
        """Batched chat-style generation with the local instruct model."""
        import torch
        tok = self._models["llm_tokenizer"]
        model = self._models["llm_model"]
        texts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=CONFIG.get('local_temperature', 0.7),
                top_p=CONFIG.get('local_top_p', 0.9),
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in tok.batch_decode(new_tokens, skip_special_tokens=True)]

    def _generate_api(self, prompt: str, max_tokens: int) -> str:
        """Call the OpenAI-compatible API with exponential-backoff retry."""
        delay = self._retry_delay
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=CONFIG.get('api_temperature', 0.8),
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self._max_retries - 1:
                    print(f"API call failed after {self._max_retries} attempts: {e}")
                    return ''
                print(f"API error (attempt {attempt + 1}/{self._max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
        return ''
