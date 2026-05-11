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
# The Space's API is a two-step pipeline:
#   /_user_submit  → append user message to history
#   /_bot_reply    → generate assistant reply (Talkie-1930 13B-IT)
_API_USER_SUBMIT = "/_user_submit"
_API_BOT_REPLY = "/_bot_reply"
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
        # The Space stores chat history as Gradio Chatbot messages:
        #   {"role": "user"|"assistant", "content": [{"text": "...", "type": "text"}]}
        # We accept a plain {"role", "content": "..."} history from our orchestrator
        # and convert to the Chatbot shape on each call.
        chat_history = _to_chatbot_history(history)

        # Step 1: echo user message into history.
        _, chat_history = self._client.predict(
            user_msg=message,
            history=chat_history,
            api_name=_API_USER_SUBMIT,
        )

        # Step 2: generate the assistant reply.
        chat_history = self._client.predict(
            history=chat_history,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=120,
            top_p=0.95,
            top_k=50,
            api_name=_API_BOT_REPLY,
        )

        return _extract_last_assistant_text(chat_history)


def _to_chatbot_history(plain: list[dict]) -> list[dict]:
    """Convert our orchestrator's {'role','content': str} history into the
    Gradio Chatbot dict format expected by the Talkie Space."""
    out: list[dict] = []
    for turn in plain:
        content = turn.get("content", "")
        if isinstance(content, str):
            content = [{"text": content, "type": "text"}]
        out.append({"role": turn["role"], "content": content})
    return out


def _extract_last_assistant_text(chat_history: list[dict]) -> str:
    """Pull the most recent assistant message's text out of the Chatbot history."""
    for turn in reversed(chat_history):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "")
        # content may be: str | list[dict(text=..., type='text')] | list[str]
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(parts).strip()
    return ""
