"""Thin wrapper around the official Sarvam AI Python SDK."""

import time
from typing import Any

from sarvamai import SarvamAI


class SarvamClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0,
        max_tokens: int = 3000,
        max_retries: int = 3,
    ):
        if not api_key or api_key == "replace_with_your_sarvam_api_key":
            raise ValueError(
                "SARVAM_API_KEY is missing. Set it in your .env file."
            )

        self.client = SarvamAI(api_subscription_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max(0, max_retries)

    def generate(self, prompt: str) -> dict[str, Any]:
        """Generate an answer and capture diagnostic information."""

        last_error = None

        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()

            try:
                response = self.client.chat.completions(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort="low",
                )

                elapsed = time.perf_counter() - started

                choices = getattr(response, "choices", None) or []
                choice = choices[0] if choices else None
                message = getattr(choice, "message", None) if choice else None
                content = getattr(message, "content", None) if message else None

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

                finish_reason = (
                    getattr(choice, "finish_reason", None)
                    if choice
                    else None
                )

                if not isinstance(content, str) or not content.strip():
                    # Print diagnostics without exposing the prompt or API key.
                    print(
                        "\nSarvam returned no visible answer."
                        f"\nAttempt: {attempt + 1}/{self.max_retries + 1}"
                        f"\nFinish reason: {finish_reason!r}"
                        f"\nToken usage: {usage_data}"
                        f"\nResponse type: {type(response).__name__}"
                    )

                    # Do not retry a response that is consistently empty
                    # without inspecting its finish reason and token usage.
                    raise RuntimeError(
                        "Empty model response. See diagnostic information above."
                    )

                return {
                    "answer": content.strip(),
                    "latency_seconds": round(elapsed, 4),
                    "usage": usage_data,
                    "provider_model": getattr(
                        response, "model", self.model
                    ),
                    "finish_reason": finish_reason,
                }

            except Exception as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    break

                # Retry transient API failures with exponential backoff.
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(
            f"Sarvam API call failed after "
            f"{self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error