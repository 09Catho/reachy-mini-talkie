"""
Dev console — type questions, hear 1930 replies through your laptop speakers.

Use this when you have no Reachy Mini hardware and the simulator's mic is silent.
Bypasses STT and the daemon entirely; only uses TalkieClient + KokoroTTS + sounddevice.

    python -m reachy_mini_talkie.dev_console
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from .llm import TalkieClient
from .tts import KokoroTTS, KOKORO_SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "ERROR: sounddevice is required for dev mode.\n"
            "Install it with:  .venv\\Scripts\\pip install sounddevice",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("  THE 1930 BROADCASTER — Dev Console")
    print("=" * 60)
    print("Type a question and press Enter.  Ctrl+C to quit.")
    print("Try:  'Tell me about the wireless.'")
    print("Try:  'What is Twitter?'")
    print("-" * 60)

    tts = KokoroTTS()
    talkie = TalkieClient()
    history: list[dict] = []

    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGood evening.")
            break

        if not user_text:
            continue

        print("(consulting the wireless …)")
        reply = talkie.query(user_text, history)
        print(f"\nBroadcaster> {reply}\n")

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        if len(history) > 12:
            history = history[-12:]

        # Synthesise + play
        chunks: list[np.ndarray] = []
        for chunk in tts.synthesise(reply, output_sample_rate=KOKORO_SAMPLE_RATE):
            chunks.append(chunk)
        if chunks:
            audio = np.concatenate(chunks)
            sd.play(audio, samplerate=KOKORO_SAMPLE_RATE)
            sd.wait()


if __name__ == "__main__":
    main()
