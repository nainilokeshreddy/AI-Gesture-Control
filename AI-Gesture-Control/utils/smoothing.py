"""
Smoothing & Filtering Utilities
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Implements:
1. Exponential Moving Average (EMA) for scalar and 2D/3D coordinates.
2. 1-Euro Filter for adaptive jitter-free, low-latency tracking.
"""

import math
import time
from typing import Optional, Tuple, Union
import numpy as np


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) filter.
    y_t = alpha * x_t + (1 - alpha) * y_{t-1}
    """

    def __init__(self, alpha: float = 0.35):
        """
        Args:
            alpha: Smoothing factor between 0.0 (maximum smoothing, slower)
                   and 1.0 (raw input, no smoothing).
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in range [0.0, 1.0]")
        self.alpha = alpha
        self.state: Optional[np.ndarray] = None

    def filter(self, value: Union[float, Tuple[float, ...], np.ndarray]) -> np.ndarray:
        val_arr = np.asarray(value, dtype=np.float64)
        if self.state is None:
            self.state = val_arr.copy()
        else:
            self.state = self.alpha * val_arr + (1.0 - self.alpha) * self.state
        return self.state.copy()

    def reset(self) -> None:
        self.state = None


class LowPassFilter:
    """Standard first-order low-pass filter used by OneEuroFilter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.y: Optional[np.ndarray] = None

    def filter(self, value: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if alpha is not None:
            self.alpha = alpha
        if self.y is None:
            self.y = value.copy()
        else:
            self.y = self.alpha * value + (1.0 - self.alpha) * self.y
        return self.y.copy()

    def reset(self) -> None:
        self.y = None


class OneEuroFilter:
    """
    1-Euro Filter (Casiez, Roussel, Vogel - CHI 2012)
    Adaptive low-pass filter providing high jitter-reduction when stationary,
    and minimal lag during rapid movements.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        """
        Args:
            min_cutoff: Minimum cutoff frequency in Hz (controls jitter when slow).
            beta: Speed coefficient (controls lag during fast motion).
            d_cutoff: Cutoff frequency for derivative calculation in Hz.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    @staticmethod
    def _alpha(rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0 else 0.001
        return 1.0 / (1.0 + tau / te)

    def filter(
        self,
        value: Union[float, Tuple[float, ...], np.ndarray],
        timestamp: Optional[float] = None
    ) -> np.ndarray:
        val_arr = np.asarray(value, dtype=np.float64)
        t = timestamp if timestamp is not None else time.time()

        if self.last_time is None:
            self.last_time = t
            return self.x_filter.filter(val_arr, 1.0)

        dt = t - self.last_time
        self.last_time = t
        rate = 1.0 / dt if dt > 1e-5 else 30.0

        # Estimate derivative (speed)
        prev_x = self.x_filter.y
        if prev_x is None:
            dx = np.zeros_like(val_arr)
        else:
            dx = (val_arr - prev_x) * rate

        # Filter derivative
        alpha_d = self._alpha(rate, self.d_cutoff)
        edx = self.dx_filter.filter(dx, alpha_d)

        # Adaptive cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(edx)
        alpha = self._alpha(rate, cutoff)

        return self.x_filter.filter(val_arr, alpha)

    def reset(self) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None
