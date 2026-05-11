"""Calls the multimodalart/talkie-1930 Gradio Space for period-authentic text."""

from __future__ import annotations

import logging
import random
import time

from gradio_client import Client

from .prompts import (
    FALLBACK_LINES,
    SYSTEM_PROMPT,
    contains_anachronism,
    make_retry_prompt,
)

logger = logging.getLogger(__name__)

_SPACE = "multimodalart/talkie-1930"
_API_NAME = "/chat_fn"
_MAX_RETRIES = 3
_BASE_DELAY = 2.0


def _build_client() -> Client:
    return Client(_SPACE)


class TalkieClient:
    def __init__(self) -> None:
        logger.info("Connecting to %s …", _SPACE)
        self._client = _build_client()
        logger.info("Talkie client ready.")

    def query(self, user_text: str, history: list[dict]) -> str:
        """
        Returns Talkie's 1930-era prose reply, or a polite fallback
        if the Space is unreachable after retries.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                reply = self._predict(user_text, history, SYSTEM_PROMPT)
                if contains_anachronism(reply):
                    logger.warning("Anachronism detected; retrying with stricter prompt.")
                    reply = self._predict(
                        make_retry_prompt(user_text), history, SYSTEM_PROMPT
                    )
                return reply
            except Exception as exc:
                logger.warning("Talkie call failed (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(_BASE_DELAY * attempt)
                else:
                    return random.choice(FALLBACK_LINES)

        return random.choice(FALLBACK_LINES)

    def _predict(self, message: str, history: list[dict], system_prompt: str) -> str:
        result = self._client.predict(
            message=message,
            history=history,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=120,
            top_p=0.95,
            top_k=50,
            api_name=_API_NAME,
        )
        if isinstance(result, (list, tuple)):
            # gradio_client may return (history, response) or just response
            return str(result[-1]) if result else ""
        return str(result)
