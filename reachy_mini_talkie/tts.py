"""Kokoro-82M TTS wrapper — British male voice, resampled to robot's output rate."""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np
from scipy.signal import resample

logger = logging.getLogger(__name__)

# British English voices (from VOICES.md, B-grade or better):
#   bm_george — male, RP-leaning  ← default
#   bm_fable  — male, theatrical
#   bf_emma   — female, BBC-announcer
DEFAULT_VOICE = "bm_george"
DEFAULT_SPEED = 0.92      # slightly slower for 1930 broadcaster cadence
KOKORO_SAMPLE_RATE = 24_000


class KokoroTTS:
    def __init__(self, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED) -> None:
        from kokoro import KPipeline  # lazy import so startup isn't blocked if missing

        logger.info("Loading Kokoro British pipeline (voice=%s) …", voice)
        self._pipeline = KPipeline(lang_code="b")
        self._voice = voice
        self._speed = speed
        logger.info("Kokoro TTS ready.")

    def synthesise(self, text: str, output_sample_rate: int) -> Generator[np.ndarray, None, None]:
        """
        Yields float32 mono audio chunks at output_sample_rate, one per sentence.
        Streams so the robot starts playing before the full reply is synthesised.
        """
        for _gs, _ps, audio_24k in self._pipeline(
            text, voice=self._voice, speed=self._speed
        ):
            if audio_24k is None or len(audio_24k) == 0:
                continue
            audio_out = _resample(audio_24k, output_sample_rate)
            yield audio_out


def _resample(audio: np.ndarray, target_rate: int) -> np.ndarray:
    """Resample from Kokoro's 24 kHz to the robot's output sample rate."""
    n_target = int(target_rate * len(audio) / KOKORO_SAMPLE_RATE)
    resampled = resample(audio, n_target)
    return resampled.astype(np.float32)
