"""faster-whisper STT with energy-based VAD, runs on CPU (Pi 5 friendly)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000        # Reachy mic native rate (confirmed in SDK docs)
CHUNK_FRAMES = 1_600        # 100 ms per chunk @ 16 kHz
ENERGY_THRESHOLD = 0.01     # RMS above this = speech
SILENCE_TAIL_S = 0.8        # seconds of silence to end an utterance
MIN_SPEECH_S = 0.4          # minimum speech length to bother transcribing


class WhisperSTT:
    def __init__(self, model_size: str = "tiny.en") -> None:
        from faster_whisper import WhisperModel  # lazy import

        logger.info("Loading faster-whisper %s (CPU int8) …", model_size)
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("Whisper ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono 16 kHz float32 array. Returns stripped text."""
        segments, _ = self._model.transcribe(
            audio, language="en", beam_size=1, best_of=1, vad_filter=True
        )
        return " ".join(seg.text for seg in segments).strip()


class VADCapture:
    """
    Reads from the Reachy mic, applies energy VAD, and calls `on_utterance`
    with a transcribed string each time the user finishes speaking.

    Pass `push_to_talk=True` to disable VAD and call manually via
    `start_speaking()` / `stop_speaking()`.
    """

    def __init__(
        self,
        stt: WhisperSTT,
        on_utterance: Callable[[str], None],
        push_to_talk: bool = False,
    ) -> None:
        self._stt = stt
        self._on_utterance = on_utterance
        self._push_to_talk = push_to_talk
        self._recording = False
        self._stop_event = threading.Event()
        self._manual_recording = threading.Event()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self, robot) -> None:
        self._robot = robot
        self._stop_event.clear()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop_event.set()

    def start_speaking(self) -> None:
        """For push-to-talk: user pressed the button."""
        self._manual_recording.set()

    def stop_speaking(self) -> None:
        """For push-to-talk: user released the button."""
        self._manual_recording.clear()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        robot = self._robot
        while not self._stop_event.is_set():
            if self._push_to_talk:
                self._manual_recording.wait()
                utterance = self._capture_until_manual_stop(robot)
            else:
                utterance = self._capture_with_vad(robot)

            if utterance is not None and len(utterance) > 0:
                text = self._stt.transcribe(utterance)
                if text:
                    logger.info("Heard: %s", text)
                    self._on_utterance(text)

    def _capture_with_vad(self, robot) -> np.ndarray | None:
        """Collect audio while speech detected; return after silence tail."""
        buffer: list[np.ndarray] = []
        silence_frames = 0
        silence_limit = int(SILENCE_TAIL_S * SAMPLE_RATE / CHUNK_FRAMES)
        speech_frames = 0
        in_speech = False

        while not self._stop_event.is_set():
            chunk = robot.media.get_audio_sample()  # (N, 2) float32 @ 16kHz
            if chunk is None:
                time.sleep(0.05)
                continue
            mono = chunk.mean(axis=1) if chunk.ndim == 2 else chunk
            rms = float(np.sqrt(np.mean(mono ** 2)))

            if rms > ENERGY_THRESHOLD:
                if not in_speech:
                    in_speech = True
                buffer.append(mono)
                speech_frames += 1
                silence_frames = 0
            elif in_speech:
                buffer.append(mono)
                silence_frames += 1
                if silence_frames >= silence_limit:
                    break
            # if not yet in speech, just discard the chunk

        if speech_frames * CHUNK_FRAMES / SAMPLE_RATE < MIN_SPEECH_S:
            return None
        return np.concatenate(buffer) if buffer else None

    def _capture_until_manual_stop(self, robot) -> np.ndarray | None:
        buffer: list[np.ndarray] = []
        while self._manual_recording.is_set() and not self._stop_event.is_set():
            chunk = robot.media.get_audio_sample()
            if chunk is None:
                time.sleep(0.05)
                continue
            mono = chunk.mean(axis=1) if chunk.ndim == 2 else chunk
            buffer.append(mono)
        return np.concatenate(buffer) if buffer else None
