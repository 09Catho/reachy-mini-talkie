"""
Component unit tests — no hardware, no Reachy SDK required.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pytest


# ── prompts ────────────────────────────────────────────────────── #

def test_anachronism_detection():
    from reachy_mini_talkie.prompts import contains_anachronism
    assert contains_anachronism("I checked the internet for news.") is True
    assert contains_anachronism("I read it in the Times of London.") is False
    assert contains_anachronism("The AI is remarkable.") is True
    assert contains_anachronism("Good evening, dear listeners.") is False


def test_retry_prompt_appended():
    from reachy_mini_talkie.prompts import make_retry_prompt
    original = "What year is it?"
    result = make_retry_prompt(original)
    assert result.startswith(original)
    assert "1930" in result


# ── choreography ───────────────────────────────────────────────── #

def test_pose_functions_return_dicts():
    from reachy_mini_talkie.choreography import _idle, _listening, _thinking, _speaking
    for fn in (_idle, _listening, _thinking):
        pose = fn(0.0)
        assert isinstance(pose, dict)
        assert "head_z" in pose
        assert "antenna_left" in pose

    pose = _speaking(0.0, rms=0.05)
    assert isinstance(pose, dict)
    assert "head_z" in pose


def test_choreographer_mood_switch():
    from reachy_mini_talkie.choreography import Choreographer, Mood
    c = Choreographer()
    c.set_mood(Mood.LISTENING)
    c.set_mood(Mood.THINKING)
    c.set_mood(Mood.SPEAKING)
    c.update_rms(0.02)
    # No exception means the state machine is stable


# ── tts resample ───────────────────────────────────────────────── #

def test_resample_shape():
    from reachy_mini_talkie.tts import _resample
    # 1 second of 24 kHz audio → should come out as ~16 kHz
    audio_24k = np.random.randn(24_000).astype(np.float32)
    audio_16k = _resample(audio_24k, target_rate=16_000)
    assert audio_16k.dtype == np.float32
    assert abs(len(audio_16k) - 16_000) < 10   # within 10 samples


# ── stt VAD helpers ────────────────────────────────────────────── #

def test_vad_energy_threshold():
    """Silence should not be flagged as speech."""
    silence = np.zeros(1_600, dtype=np.float32)
    rms = float(np.sqrt(np.mean(silence ** 2)))
    from reachy_mini_talkie.stt import ENERGY_THRESHOLD
    assert rms < ENERGY_THRESHOLD


def test_vad_speech_detected():
    """Loud audio should exceed the threshold."""
    loud = np.ones(1_600, dtype=np.float32) * 0.3
    rms = float(np.sqrt(np.mean(loud ** 2)))
    from reachy_mini_talkie.stt import ENERGY_THRESHOLD
    assert rms > ENERGY_THRESHOLD


# ── llm fallback lines ─────────────────────────────────────────── #

def test_fallback_lines_non_empty():
    from reachy_mini_talkie.prompts import FALLBACK_LINES
    assert len(FALLBACK_LINES) >= 3
    assert all(isinstance(line, str) and len(line) > 10 for line in FALLBACK_LINES)
