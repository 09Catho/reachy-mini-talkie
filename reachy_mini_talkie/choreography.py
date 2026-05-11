"""Head and antenna choreography — three moods keyed to the conversation state."""

from __future__ import annotations

import math
import threading
import time
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

TICK_RATE = 20      # Hz — how often we push a new target
TICK_S = 1.0 / TICK_RATE


class Mood(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


class Choreographer:
    """
    Runs a background thread that smoothly animates head + antennas.
    Call set_mood() to transition. Inject an audio_rms callback for
    the speaking mood so the head bobs to the voice.
    """

    def __init__(self) -> None:
        self._mood = Mood.IDLE
        self._mood_lock = threading.Lock()
        self._stop = threading.Event()
        self._audio_rms: float = 0.0
        self._rms_lock = threading.Lock()
        self._t = 0.0   # time counter for oscillators

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_mood(self, mood: Mood) -> None:
        with self._mood_lock:
            self._mood = mood
            self._t = 0.0   # reset oscillator phase on transition

    def update_rms(self, rms: float) -> None:
        with self._rms_lock:
            self._audio_rms = rms

    def start(self, robot: "ReachyMini") -> None:
        self._robot = robot
        self._stop.clear()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._mood_lock:
                mood = self._mood
            pose = self._compute_pose(mood)
            self._apply(pose)
            self._t += TICK_S
            time.sleep(TICK_S)

    def _compute_pose(self, mood: Mood) -> dict:
        t = self._t
        if mood == Mood.IDLE:
            return _idle(t)
        if mood == Mood.LISTENING:
            return _listening(t)
        if mood == Mood.THINKING:
            return _thinking(t)
        if mood == Mood.SPEAKING:
            with self._rms_lock:
                rms = self._audio_rms
            return _speaking(t, rms)
        return _idle(t)

    def _apply(self, pose: dict) -> None:
        try:
            from reachy_mini.utils import create_head_pose
            head = create_head_pose(
                x=pose.get("head_x", 0),
                z=pose.get("head_z", 0),
                roll=pose.get("head_roll", 0),
                degrees=True,
                mm=True,
            )
            antennas = np.deg2rad([
                pose.get("antenna_left", 0),
                pose.get("antenna_right", 0),
            ])
            self._robot.set_target(head=head, antennas=antennas)
        except Exception:
            pass   # robot may not be connected in sim; silently skip


# ------------------------------------------------------------------ #
# Pose functions — return dicts with named joint targets (degrees/mm)
# ------------------------------------------------------------------ #

def _idle(t: float) -> dict:
    sway = 0.5 * math.sin(2 * math.pi * 0.1 * t)
    return {"head_x": 0, "head_z": 0, "head_roll": sway,
            "antenna_left": 0, "antenna_right": 0}


def _listening(t: float) -> dict:
    sway = 1.0 * math.sin(2 * math.pi * 0.3 * t)
    return {"head_x": 0, "head_z": 5, "head_roll": sway,
            "antenna_left": 10, "antenna_right": 10}


def _thinking(t: float) -> dict:
    sweep = 5.0 * math.sin(2 * math.pi * 0.2 * t)
    return {"head_x": sweep, "head_z": 0, "head_roll": 8,
            "antenna_left": 20, "antenna_right": -10}


def _speaking(t: float, rms: float) -> dict:
    bob_amp = min(rms * 120, 8.0)   # scale RMS energy → degrees (cap 8°)
    bob = bob_amp * math.sin(2 * math.pi * 3.0 * t)
    # antenna twitch: wiggle both outward in sync with speech rhythm
    ant = 15 * abs(math.sin(2 * math.pi * 1.5 * t))
    return {"head_x": 0, "head_z": bob, "head_roll": 0,
            "antenna_left": ant, "antenna_right": ant}
