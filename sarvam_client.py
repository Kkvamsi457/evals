"""Sarvam AI client with reasoning disabled for concise evaluation outputs.

Sarvam-105B can spend the entire max_tokens budget on hidden reasoning, leaving
message.content empty and finish_reason="length". For this evaluation workload,
reasoning_effort=None disables reasoning so tokens are available for the visible
answer.
"""
import logging
import time
from typing import Any

from sarvamai import SarvamAI

logger = logging.getLogger(__name__)


class SarvamClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0,
        max_tokens: int = 1200,
        max_retries: int = 3,
    ):
        if not api_key or api_key == "replace_with_your_sarvam_api_key":
            raise ValueError(
                "SARVAM_API_KEY is missing. Copy .env.example to .env and set your key."
            )

        self.client = SarvamAI(api_subscription_key=api_key)
        self.model = model
        self.temperature = temperature
        # The starter plan has a 4096-token completion ceiling. Leave a safe cap.
        self.max_tokens = max(128, min(int(max_tokens), 4096))
        self.max_retries = max(0, int(max_retries))

    def generate(self, prompt: str) -> dict[str, Any]:
        """Return visible answer, latency, token usage and provider model."""
        last_error: Exception | None = None
        started = time.perf_counter()
        usage_data: dict[str, Any] = {}
        finish_reason = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Answer the user's request directly. Do not show "
                                "internal reasoning. Follow the requested language "
                                "and format, and keep the answer concise."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    # IMPORTANT: None disables Sarvam-105B's reasoning mode.
                    # Otherwise reasoning tokens may consume the whole budget.
                    reasoning_effort=None,
                )

                choices = getattr(response, "choices", None)
                choice = choices[0] if choices else None
                message = getattr(choice, "message", None) if choice else None
                content = getattr(message, "content", "") if message else ""
                finish_reason = (
                    getattr(choice, "finish_reason", None) if choice else None
                )

                usage = getattr(response, "usage", None)
                usage_data = {}
                if usage is not None:
                    for field in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    ):
                        value = getattr(usage, field, None)
                        if value is not None:
                            usage_data[field] = value

                if isinstance(content, str) and content.strip():
                    return {
                        "answer": content.strip(),
                        "latency_seconds": round(time.perf_counter() - started, 4),
                        "usage": usage_data,
                        "provider_model": getattr(response, "model", self.model),
                        "finish_reason": finish_reason,
                        "max_tokens_used": self.max_tokens,
                    }

                reasoning_content = (
                    getattr(message, "reasoning_content", None) if message else None
                )
                reasoning_chars = (
                    len(reasoning_content)
                    if isinstance(reasoning_content, str)
                    else 0
                )
                last_error = RuntimeError(
                    "Sarvam returned no visible answer "
                    f"(finish_reason={finish_reason!r}, "
                    f"max_tokens={self.max_tokens}, usage={usage_data}, "
                    f"reasoning_chars={reasoning_chars})."
                )
                logger.warning(
                    "Sarvam returned an empty visible answer on attempt %s/%s "
                    "(finish_reason=%r, max_tokens=%s, usage=%s, reasoning_chars=%s).",
                    attempt + 1,
                    self.max_retries + 1,
                    finish_reason,
                    self.max_tokens,
                    usage_data,
                    reasoning_chars,
                )

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Sarvam request failed on attempt %s/%s: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )

            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(
            f"Sarvam API call failed after {self.max_retries + 1} attempt(s): "
            f"{last_error}"
        ) from last_error
