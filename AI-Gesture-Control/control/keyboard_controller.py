"""
Keyboard Controller using PyAutoGUI
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Handles:
- Key press execution (e.g. 'space' for play/pause).
- Hotkey combinations (e.g. ('shift', 'n'), ('command', '+')).
- macOS Accessibility permission verification.
- Safe non-crashing fallback if OS automation fails.
"""

import time
from typing import Dict, Optional, Tuple
import pyautogui

from config.settings import DEFAULT_CONFIG, ControlConfig
from control.mouse_controller import check_macos_accessibility
from utils.logger import setup_logger


class KeyboardController:
    """
    Wraps PyAutoGUI for reliable, non-blocking keyboard presses and shortcuts.
    """

    def __init__(self, config: Optional[ControlConfig] = None):
        self.config = config or DEFAULT_CONFIG.control
        self.logger = setup_logger("KeyboardController")
        self.has_accessibility: bool = check_macos_accessibility()

        # Cooldown trackers per action/key to prevent stutter or key repeats
        self._last_action_times: Dict[str, float] = {}

    def press(self, key: str, cooldown_sec: float = 0.5) -> bool:
        """
        Press a single keyboard key (e.g. 'space').

        Args:
            key: PyAutoGUI key string identifier.
            cooldown_sec: Minimum cooldown duration in seconds.

        Returns:
            True if key was dispatched successfully, False otherwise.
        """
        now = time.time()
        last_t = self._last_action_times.get(key, 0.0)
        if (now - last_t) < cooldown_sec:
            return False

        if not self.has_accessibility:
            self.has_accessibility = check_macos_accessibility()
            if not self.has_accessibility:
                return False

        self._last_action_times[key] = now
        try:
            pyautogui.press(key)
            self.logger.info(f"Executed key press: '{key}'")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to press key '{key}' via PyAutoGUI: {e}")
            return False

    def hotkey(self, *keys: str, cooldown_sec: float = 0.5) -> bool:
        """
        Execute a keyboard hotkey combination (e.g. 'shift', 'n').

        Args:
            keys: Sequence of key strings pressed in order.
            cooldown_sec: Minimum cooldown duration in seconds.

        Returns:
            True if hotkey was dispatched successfully, False otherwise.
        """
        now = time.time()
        action_key = "+".join(keys)
        last_t = self._last_action_times.get(action_key, 0.0)
        if (now - last_t) < cooldown_sec:
            return False

        if not self.has_accessibility:
            self.has_accessibility = check_macos_accessibility()
            if not self.has_accessibility:
                return False

        self._last_action_times[action_key] = now
        try:
            pyautogui.hotkey(*keys)
            self.logger.info(f"Executed hotkey: {action_key}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to execute hotkey {action_key} via PyAutoGUI: {e}")
            return False

    def reset(self) -> None:
        """Reset internal cooldown buffers."""
        self._last_action_times.clear()
