"""
Mouse Controller using PyAutoGUI
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Handles:
- Screen coordinate transformation from camera coordinates.
- Active tracking boundary margins (dead zones).
- Adaptive coordinate smoothing (OneEuroFilter / EMA) for jitter-free cursor tracking.
- Left-click execution with refractory cooldown.
- macOS Accessibility permission detection and non-crashing fallback.
"""

import sys
import time
from typing import Optional, Tuple
import numpy as np
import pyautogui

from config.settings import DEFAULT_CONFIG, ControlConfig, SmoothingConfig
from utils.logger import setup_logger
from utils.smoothing import OneEuroFilter, ExponentialMovingAverage


def check_macos_accessibility() -> bool:
    """
    Check if the current process has macOS Accessibility permission.
    Non-macOS platforms return True by default.
    """
    if sys.platform != "darwin":
        return True

    try:
        import ctypes
        from ctypes import util

        lib_path = util.find_library("ApplicationServices")
        if not lib_path:
            return True
        app_services = ctypes.cdll.LoadLibrary(lib_path)
        app_services.AXIsProcessTrusted.restype = ctypes.c_bool
        app_services.AXIsProcessTrusted.argtypes = []
        return bool(app_services.AXIsProcessTrusted())
    except Exception:
        # Fallback if ctypes inspection is unavailable
        return True


class MouseController:
    """
    Wraps PyAutoGUI for smooth, safe cursor movements and clicks.
    """

    def __init__(
        self,
        config: Optional[ControlConfig] = None,
        smoothing: Optional[SmoothingConfig] = None,
        screen_size: Optional[Tuple[int, int]] = None,
    ):
        self.config = config or DEFAULT_CONFIG.control
        self.smoothing = smoothing or DEFAULT_CONFIG.smoothing
        self.logger = setup_logger("MouseController")

        # PyAutoGUI Safety configuration
        pyautogui.FAILSAFE = getattr(self.config, "pyautogui_failsafe", True)
        pyautogui.PAUSE = 0.0  # Ultra-low latency for real-time tracking
        if hasattr(pyautogui, "DARWIN_CATCH_UP_TIME"):
            pyautogui.DARWIN_CATCH_UP_TIME = 0.0

        # Cached previous integer screen coordinates for redundant dispatch elimination
        self._prev_screen_pos: Optional[Tuple[int, int]] = None

        # Screen resolution
        if screen_size is not None:
            self.screen_width, self.screen_height = screen_size
        else:
            try:
                self.screen_width, self.screen_height = pyautogui.size()
            except Exception as e:
                self.logger.warning(f"Could not retrieve screen size via PyAutoGUI: {e}. Defaulting to 1920x1080.")
                self.screen_width, self.screen_height = 1920, 1080

        # Check macOS Accessibility status
        self.has_accessibility: bool = check_macos_accessibility()
        if not self.has_accessibility:
            self.logger.warning(
                "macOS Accessibility permission is not granted. "
                "PyAutoGUI cannot control mouse events. "
                "To enable: Open System Settings -> Privacy & Security -> Accessibility "
                "and grant access to your Terminal / IDE."
            )

        # Coordinate smoother (1-Euro filter for dynamic jitter reduction)
        self.smoother = OneEuroFilter(
            min_cutoff=self.smoothing.one_euro_min_cutoff,
            beta=self.smoothing.one_euro_beta,
            d_cutoff=self.smoothing.one_euro_d_cutoff,
        )

        # Click cooldown trackers
        self._last_click_time: float = 0.0
        self.click_cooldown_sec: float = getattr(self.config, "click_cooldown_sec", 0.3)
        self._last_right_click_time: float = 0.0
        self.right_click_cooldown_sec: float = getattr(self.config, "right_click_cooldown_sec", 0.4)

    def map_camera_to_screen(
        self,
        cam_x: float,
        cam_y: float,
        cam_w: int,
        cam_h: int,
    ) -> Tuple[int, int]:
        """
        Map mirrored camera coordinates to screen coordinates with margins.

        Args:
            cam_x: Mirrored camera X coordinate (0 to cam_w).
            cam_y: Mirrored camera Y coordinate (0 to cam_h).
            cam_w: Camera frame width.
            cam_h: Camera frame height.

        Returns:
            (screen_x, screen_y) tuple clamped to Mac display bounds.
        """
        margin_percent = getattr(self.config, "cursor_margin_percent", 0.15)
        margin_x = cam_w * margin_percent
        margin_y = cam_h * margin_percent

        x_min = margin_x
        x_max = cam_w - margin_x
        y_min = margin_y
        y_max = cam_h - margin_y

        # Clamp camera coordinate to active bounding region
        clamped_x = max(x_min, min(float(cam_x), x_max))
        clamped_y = max(y_min, min(float(cam_y), y_max))

        # Normalize in active box [0.0, 1.0]
        span_x = max(1.0, x_max - x_min)
        span_y = max(1.0, y_max - y_min)
        norm_x = (clamped_x - x_min) / span_x
        norm_y = (clamped_y - y_min) / span_y

        # Scale to screen resolution
        raw_screen_x = norm_x * (self.screen_width - 1)
        raw_screen_y = norm_y * (self.screen_height - 1)

        # Apply temporal smoothing
        smoothed = self.smoother.filter((raw_screen_x, raw_screen_y))
        target_x = int(round(max(0, min(self.screen_width - 1, smoothed[0]))))
        target_y = int(round(max(0, min(self.screen_height - 1, smoothed[1]))))

        return target_x, target_y

    def move_cursor(
        self,
        cam_x: float,
        cam_y: float,
        cam_w: int,
        cam_h: int,
    ) -> bool:
        """
        Move the system cursor to mapped camera coordinates.

        Returns:
            True if cursor was successfully moved, False otherwise.
        """
        target_x, target_y = self.map_camera_to_screen(cam_x, cam_y, cam_w, cam_h)

        # Optimization: Skip redundant PyAutoGUI dispatch if integer screen coordinates are identical
        if self._prev_screen_pos == (target_x, target_y):
            return True

        if not self.has_accessibility:
            # Re-check in case permission was granted at runtime
            self.has_accessibility = check_macos_accessibility()
            if not self.has_accessibility:
                return False

        try:
            pyautogui.moveTo(target_x, target_y)
            self._prev_screen_pos = (target_x, target_y)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to move cursor via PyAutoGUI: {e}")
            return False

    def left_click(self) -> bool:
        """
        Execute a single left click if cooldown has elapsed.

        Returns:
            True if click was dispatched, False otherwise.
        """
        now = time.time()
        if (now - self._last_click_time) < self.click_cooldown_sec:
            return False

        if not self.has_accessibility:
            self.has_accessibility = check_macos_accessibility()
            if not self.has_accessibility:
                return False

        self._last_click_time = now
        try:
            pyautogui.click(button="left")
            self.logger.info("Executed real OS left click.")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to execute click via PyAutoGUI: {e}")
            return False

    def right_click(self) -> bool:
        """
        Execute a single right click if cooldown has elapsed.

        Returns:
            True if right click was dispatched, False otherwise.
        """
        now = time.time()
        if (now - self._last_right_click_time) < self.right_click_cooldown_sec:
            return False

        if not self.has_accessibility:
            self.has_accessibility = check_macos_accessibility()
            if not self.has_accessibility:
                return False

        self._last_right_click_time = now
        try:
            pyautogui.rightClick()
            self.logger.info("Executed real OS right click.")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to execute right click via PyAutoGUI: {e}")
            return False

    def reset_smoother(self) -> None:
        """Reset the smoothing filter state and cached cursor position."""
        self.smoother.reset()
        self._prev_screen_pos = None
