"""
Cooldown & Debounce Management Utility
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Prevents accidental repeated actions and stabilizes gesture activations using:
1. Refractory cooldowns (time lockout between repeated triggers).
2. Debouncing / hold-time verification (requiring a gesture condition to persist).
"""

import time
from typing import Dict, Any, Optional


class CooldownManager:
    """
    Manages per-gesture refractory cooldowns.
    """

    def __init__(self, default_cooldown: float = 0.4):
        """
        Args:
            default_cooldown: Default cooldown time in seconds.
        """
        self.default_cooldown = default_cooldown
        self._last_trigger_times: Dict[Any, float] = {}

    def is_ready(self, gesture_id: Any, cooldown_sec: Optional[float] = None) -> bool:
        """
        Checks whether the specified gesture is ready to be triggered.
        """
        cooldown = cooldown_sec if cooldown_sec is not None else self.default_cooldown
        last_time = self._last_trigger_times.get(gesture_id, 0.0)
        return (time.time() - last_time) >= cooldown

    def trigger(self, gesture_id: Any) -> float:
        """
        Records the current timestamp as the last trigger time.
        Returns the trigger timestamp.
        """
        now = time.time()
        self._last_trigger_times[gesture_id] = now
        return now

    def time_remaining(self, gesture_id: Any, cooldown_sec: Optional[float] = None) -> float:
        """
        Returns seconds remaining until gesture is ready (0.0 if already ready).
        """
        cooldown = cooldown_sec if cooldown_sec is not None else self.default_cooldown
        last_time = self._last_trigger_times.get(gesture_id, 0.0)
        elapsed = time.time() - last_time
        return max(0.0, cooldown - elapsed)

    def reset(self, gesture_id: Optional[Any] = None) -> None:
        """
        Resets cooldown for a specific gesture or all gestures.
        """
        if gesture_id is None:
            self._last_trigger_times.clear()
        elif gesture_id in self._last_trigger_times:
            del self._last_trigger_times[gesture_id]


class Debouncer:
    """
    Ensures a condition remains continuously active for a required duration
    before confirming activation (e.g. 2-second stationary open palm hold or iris dwell).
    """

    def __init__(self, required_duration_sec: float):
        self.required_duration_sec = required_duration_sec
        self._start_time: Optional[float] = None
        self._is_active: bool = False

    def update(self, condition_met: bool) -> bool:
        """
        Updates debouncer state.
        
        Args:
            condition_met: True if the gesture condition is currently detected.
            
        Returns:
            True only on the exact frame the duration threshold is reached.
        """
        now = time.time()
        if not condition_met:
            self._start_time = None
            self._is_active = False
            return False

        if self._start_time is None:
            self._start_time = now
            return False

        elapsed = now - self._start_time
        if elapsed >= self.required_duration_sec and not self._is_active:
            self._is_active = True
            return True

        return False

    def progress(self) -> float:
        """
        Returns normalized progress [0.0, 1.0] toward required duration.
        """
        if self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        return min(1.0, elapsed / self.required_duration_sec)

    def reset(self) -> None:
        self._start_time = None
        self._is_active = False
