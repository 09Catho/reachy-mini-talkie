"""
Reachy Mini × Talkie-1930: The 1930 Broadcaster
================================================
State machine: IDLE → LISTENING → THINKING → SPEAKING → IDLE

Usage (via SDK):
    from reachy_mini_talkie import TalkieApp
    app = TalkieApp()
    # SDK calls app.run(reachy_mini, stop_event) in a thread.

Standalone test (no hardware):
    python -m reachy_mini_talkie
"""

from __future__ import annotations

import logging
import queue
import threading
from enum import Enum, auto

import numpy as np

from .choreography import Choreographer, Mood
from .llm import TalkieClient
from .stt import VADCapture, WhisperSTT
from .tts import KokoroTTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


class TalkieApp:
    """Reachy Mini app — conforms to ReachyMiniApp protocol (run + stop_event)."""

    custom_app_url: str | None = None

    def __init__(
        self,
        push_to_talk: bool = False,
        voice: str = "bm_george",
        speed: float = 0.92,
    ) -> None:
        self._push_to_talk = push_to_talk
        self._voice = voice
        self._speed = speed
        self._utterance_queue: queue.Queue[str] = queue.Queue()
        self._history: list[dict] = []

    # ------------------------------------------------------------------ #
    # SDK entry point
    # ------------------------------------------------------------------ #

    def run(self, reachy_mini, stop_event: threading.Event) -> None:
        logger.info("=== The 1930 Broadcaster starting ===")

        # Boot all subsystems
        choreographer = Choreographer()
        stt = WhisperSTT()
        tts = KokoroTTS(voice=self._voice, speed=self._speed)
        talkie = TalkieClient()
        vad = VADCapture(
            stt=stt,
            on_utterance=self._utterance_queue.put,
            push_to_talk=self._push_to_talk,
        )

        choreographer.start(reachy_mini)
        choreographer.set_mood(Mood.IDLE)

        vad.start(reachy_mini)
        reachy_mini.media.start_playing()

        logger.info("Ready. Speak to begin.")

        try:
            self._conversation_loop(
                reachy_mini, stop_event, talkie, tts, choreographer
            )
        finally:
            vad.stop()
            choreographer.stop()
            reachy_mini.media.stop_playing()
            logger.info("=== The 1930 Broadcaster stopped ===")

    # ------------------------------------------------------------------ #
    # Conversation loop
    # ------------------------------------------------------------------ #

    def _conversation_loop(
        self, robot, stop_event, talkie, tts, choreographer
    ) -> None:
        state = State.IDLE
        choreographer.set_mood(Mood.IDLE)

        while not stop_event.is_set():

            # --- IDLE / LISTENING ---
            state = State.LISTENING
            choreographer.set_mood(Mood.LISTENING)

            try:
                user_text = self._utterance_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            logger.info("User said: %r", user_text)

            # --- THINKING ---
            state = State.THINKING
            choreographer.set_mood(Mood.THINKING)

            reply = talkie.query(user_text, self._history)
            logger.info("Talkie replied: %r", reply)

            # Append to rolling history (keep last 6 turns to stay within context)
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": reply})
            if len(self._history) > 12:
                self._history = self._history[-12:]

            # --- SPEAKING ---
            state = State.SPEAKING
            choreographer.set_mood(Mood.SPEAKING)

            self._speak(robot, tts, choreographer, reply)

            if stop_event.is_set():
                break

        choreographer.set_mood(Mood.IDLE)

    # ------------------------------------------------------------------ #
    # Audio playback with choreography RMS feed
    # ------------------------------------------------------------------ #

    def _speak(self, robot, tts: KokoroTTS, choreographer: Choreographer, text: str) -> None:
        for chunk in tts.synthesise(text):
            if chunk is None or len(chunk) == 0:
                continue
            # Feed RMS to choreographer for head-bob modulation
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            choreographer.update_rms(rms)

            # Reachy expects (N, 1) or (N, 2) float32 @ 16 kHz
            stereo = np.stack([chunk, chunk], axis=1)
            robot.media.push_audio_sample(stereo)


# ------------------------------------------------------------------ #
# Standalone / simulation entry point
# ------------------------------------------------------------------ #

def _run_simulation() -> None:
    """Quick smoke-test: run against a mock robot (no hardware needed)."""
    import time

    class MockMedia:
        def start_recording(self): pass
        def stop_recording(self): pass
        def start_playing(self): pass
        def stop_playing(self): pass
        def get_audio_sample(self):
            time.sleep(0.1)
            return np.zeros((1600, 2), dtype=np.float32)
        def push_audio_sample(self, _samples): pass
        def get_DoA(self): return 0.0, False

    class MockHeadPose:
        pass

    class MockRobot:
        media = MockMedia()
        def set_target(self, **_): pass
        def goto_target(self, **_): pass

    stop = threading.Event()

    app = TalkieApp()

    # Simulate one canned utterance then stop
    def _feeder():
        time.sleep(3)
        app._utterance_queue.put("Good evening. What is electricity?")
        time.sleep(12)
        stop.set()

    t = threading.Thread(target=_feeder, daemon=True)
    t.start()

    app.run(MockRobot(), stop)


if __name__ == "__main__":
    _run_simulation()
