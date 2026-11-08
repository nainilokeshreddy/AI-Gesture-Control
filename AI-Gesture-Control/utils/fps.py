"""
FPS Calculation Utility
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform
"""

import time
from collections import deque
from typing import Optional


class FPSCounter:
    """
    Real-time Frames Per Second (FPS) counter with moving-window averaging
    for stable, jitter-free display.
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of recent frame intervals to average over.
        """
        self.window_size = window_size
        self._timestamps = deque(maxlen=window_size)
        self._last_time: Optional[float] = None
        self._fps: float = 0.0

    def tick(self) -> float:
        """
        Record the arrival of a new frame and calculate current FPS.

        Returns:
            Current average FPS across the window.
        """
        current_time = time.perf_counter()

        if self._last_time is not None:
            dt = current_time - self._last_time
            if dt > 0:
                self._timestamps.append(dt)
                avg_dt = sum(self._timestamps) / len(self._timestamps)
                self._fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

        self._last_time = current_time
        return self._fps

    @property
    def fps(self) -> float:
        """Returns the most recently computed FPS."""
        return self._fps

    def reset(self) -> None:
        """Reset internal frame history."""
        self._timestamps.clear()
        self._last_time = None
        self._fps = 0.0
