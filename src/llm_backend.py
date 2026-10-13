"""
LLM Backend abstraction.

Supports two modes:
  - 'local'  : uses the existing GPT-2 / T5 / DistilBERT models loaded in memory
  - 'api'    : calls any OpenAI-compatible endpoint (OpenRouter by default)

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
from typing import Dict, Optional

from config import CONFIG


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
        if mode not in ('local', 'api'):
            raise ValueError(f"mode must be 'local' or 'api', got '{mode}'")

        self.mode = mode

        if mode == 'local':
            if models is None:
                raise ValueError("models dict is required for 'local' mode")
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

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Generate text for the given prompt.

        Returns the generated string (stripped), or '' on failure.
        """
        if self.mode == 'local':
            return self._generate_local(prompt, max_tokens)
        else:
            return self._generate_api(prompt, max_tokens)

    def generate_sentiment(self, text: str) -> str:
        """
        Special-case for sentiment: returns 'POSITIVE / NEGATIVE / NEUTRAL. <explanation>'.
        API mode produces a richer explanation; local mode falls back to DistilBERT + GPT-2.
        """
        if self.mode == 'api':
            prompt = (
                "Analyze the sentiment of the following text. "
                "Start your response with exactly one of: Positive, Negative, or Neutral. "
                "Then add a period and one sentence explaining why.\n\n"
                f"Text: {text}\n\nSentiment:"
            )
            return self.generate(prompt, max_tokens=80)
        else:
            # Local path: DistilBERT for label, GPT-2 for explanation
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_local(self, prompt: str, max_tokens: int) -> str:
        """Route to GPT-2 (local mode only — keeps existing behaviour)."""
        from utils import generate_gpt2_output
        return generate_gpt2_output(
            self._models["gpt2_tokenizer"],
            self._models["gpt2_model"],
            prompt,
            CONFIG['device'],
            max_length=max_tokens,
        )

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
