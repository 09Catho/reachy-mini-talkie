"""Kokoro-82M TTS wrapper — British male voice, resampled to 16 kHz for Reachy."""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)

# British English voices (from VOICES.md, B-grade or better):
#   bm_george — male, RP-leaning  ← default
#   bm_fable  — male, theatrical
#   bf_emma   — female, BBC-announcer
DEFAULT_VOICE = "bm_george"
DEFAULT_SPEED = 0.92      # slightly slower for 1930 broadcaster cadence
KOKORO_SAMPLE_RATE = 24_000
REACHY_SAMPLE_RATE = 16_000


class KokoroTTS:
    def __init__(self, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED) -> None:
        from kokoro import KPipeline  # lazy import so startup isn't blocked if missing

        logger.info("Loading Kokoro British pipeline (voice=%s) …", voice)
        self._pipeline = KPipeline(lang_code="b")
        self._voice = voice
        self._speed = speed
        logger.info("Kokoro TTS ready.")

    def synthesise(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Yields 16 kHz float32 mono audio chunks, one per sentence.
        Streams so the robot can start playing before the full text is done.
        """
        for _gs, _ps, audio_24k in self._pipeline(
            text, voice=self._voice, speed=self._speed
        ):
            if audio_24k is None or len(audio_24k) == 0:
                continue
            audio_16k = _resample(audio_24k)
            yield audio_16k

    def synthesise_to_array(self, text: str) -> np.ndarray:
        chunks = list(self.synthesise(text))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)


def _resample(audio: np.ndarray) -> np.ndarray:
    """Downsample from 24 kHz to 16 kHz (ratio 2:3)."""
    resampled = scipy.signal.resample_poly(audio, up=2, down=3)
    return resampled.astype(np.float32)
