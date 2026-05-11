"""
Reachy Mini × Talkie-1930: The 1930 Broadcaster
================================================
State machine: IDLE → LISTENING → THINKING → SPEAKING → IDLE

The daemon runs this as: python -m reachy_mini_talkie.main
It calls app.wrapped_run() which connects to the robot then calls run().
"""

from __future__ import annotations

import logging
import queue
import threading
from enum import Enum, auto

import numpy as np

from reachy_mini import ReachyMini, ReachyMiniApp

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


class ReachyMiniTalkie(ReachyMiniApp):
    """Reachy Mini × Talkie-1930: The 1930 Broadcaster."""

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

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
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

        reachy_mini.media.start_recording()
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
            reachy_mini.media.stop_recording()
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
        import time
        out_rate = robot.media.get_output_audio_samplerate()

        for chunk in tts.synthesise(text, output_sample_rate=out_rate):
            if chunk is None or len(chunk) == 0:
                continue
            # Feed RMS to choreographer for head-bob modulation
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            choreographer.update_rms(rms)

            # push_audio_sample is non-blocking — must wait for playback to finish
            stereo = np.stack([chunk, chunk], axis=1)
            robot.media.push_audio_sample(stereo)
            time.sleep(len(chunk) / out_rate)


def main() -> None:
    """CLI entry point — also called by the daemon via python -u -m reachy_mini_talkie.main."""
    app = ReachyMiniTalkie()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
