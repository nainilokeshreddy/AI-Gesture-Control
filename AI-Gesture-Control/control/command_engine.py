"""
Command Engine
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Coordinates action mapping, control mode state (ON/OFF), emergency safety locks,
and dispatching real OS actions to MouseController and KeyboardController.
"""

from dataclasses import dataclass
import time
from typing import Optional, Tuple

from config.settings import DEFAULT_CONFIG, ControlConfig, SmoothingConfig
from control.action_mapper import ActionMapper, SemanticAction
from control.mouse_controller import MouseController
from control.keyboard_controller import KeyboardController
from gestures.hand_gestures import GestureRecognitionResult
from utils.logger import setup_logger


@dataclass
class CommandEngineStatus:
    """Current state of the command engine for UI reporting."""
    control_mode: bool = False
    is_cursor_active: bool = False
    last_action_name: str = "NONE"
    last_action_hand: str = ""
    last_action_time: float = 0.0
    has_accessibility: bool = True


class CommandEngine:
    """
    Central coordinator enforcing safety policies, control-mode gating,
    and OS command dispatch for mouse and keyboard actions.
    """

    def __init__(
        self,
        mouse_controller: Optional[MouseController] = None,
        keyboard_controller: Optional[KeyboardController] = None,
        action_mapper: Optional[ActionMapper] = None,
        control_config: Optional[ControlConfig] = None,
        smoothing_config: Optional[SmoothingConfig] = None,
    ):
        self.config = control_config or DEFAULT_CONFIG.control
        self.smoothing = smoothing_config or DEFAULT_CONFIG.smoothing
        self.logger = setup_logger("CommandEngine")

        self.mouse_controller = mouse_controller or MouseController(
            config=self.config, smoothing=self.smoothing
        )
        self.keyboard_controller = keyboard_controller or KeyboardController(
            config=self.config
        )
        self.action_mapper = action_mapper or ActionMapper()

        # Control Mode State: Safety first - defaults to OFF
        self.control_mode: bool = False
        self.is_cursor_active: bool = False
        self.last_action_name: str = "NONE"
        self.last_action_hand: str = ""
        self.last_action_time: float = 0.0

        # Lockout timestamp to prevent immediate gesture re-activation after ESC
        self._emergency_lockout_until: float = 0.0

    def emergency_disable(self) -> None:
        """
        Emergency safety kill switch (invoked by ESC key).
        Instantly halts cursor tracking, disables control mode, and applies lockout.
        """
        now = time.time()
        self.control_mode = False
        self.is_cursor_active = False
        lockout = getattr(self.config, "emergency_lockout_sec", 1.5)
        self._emergency_lockout_until = now + lockout
        self.last_action_name = "DISABLED (ESC)"
        self.last_action_hand = ""
        self.last_action_time = now
        self.mouse_controller.reset_smoother()
        self.action_mapper.reset()
        self.keyboard_controller.reset()
        self.logger.warning("EMERGENCY DISABLE ACTIVATED. Control mode is OFF.")

    def toggle_control_mode(self, force: Optional[bool] = None) -> bool:
        """
        Toggle or explicitly set control mode.

        Args:
            force: If provided, explicitly sets control_mode to this boolean.

        Returns:
            New control_mode state.
        """
        now = time.time()
        if now < self._emergency_lockout_until and force is None:
            self.logger.info("Toggle ignored: Emergency lockout is still active.")
            return self.control_mode

        if force is not None:
            self.control_mode = force
        else:
            self.control_mode = not self.control_mode

        if not self.control_mode:
            self.is_cursor_active = False
            self.mouse_controller.reset_smoother()

        self.last_action_name = f"MODE {'ON' if self.control_mode else 'OFF'}"
        self.last_action_time = now
        self.logger.info(f"Control Mode changed to: {'ON' if self.control_mode else 'OFF'}")
        return self.control_mode

    def process(
        self,
        gesture_result: Optional[GestureRecognitionResult],
        frame_shape: Tuple[int, ...],
    ) -> CommandEngineStatus:
        """
        Process detected gestures and dispatch OS actions if control mode is enabled.

        Args:
            gesture_result: Output from HandGestureRecognizer.
            frame_shape: Shape of camera frame (H, W, C).

        Returns:
            CommandEngineStatus with live status for HUD display.
        """
        cam_h, cam_w = frame_shape[:2]
        actions = self.action_mapper.map_gestures(gesture_result)
        now = time.time()

        cursor_moved_this_frame = False

        for req in actions:
            # 1. Control 7: Mode toggle operates regardless of current control mode
            if req.action == SemanticAction.TOGGLE_CONTROL_MODE:
                self.toggle_control_mode()
                continue

            # 2. Master Safety Gate: Block ALL computer controls when control mode is OFF
            if not self.control_mode:
                continue

            # 3. Control 1: Cursor Movement (Index Moving)
            if req.action == SemanticAction.MOVE_CURSOR and req.cursor_pos:
                cam_x, cam_y = req.cursor_pos
                moved = self.mouse_controller.move_cursor(cam_x, cam_y, cam_w, cam_h)
                if moved:
                    cursor_moved_this_frame = True

            # 4. Control 2: Left Click (Thumb + Index Pinch Down)
            elif req.action == SemanticAction.LEFT_CLICK:
                if self.mouse_controller.left_click():
                    self.last_action_name = "LEFT CLICK"
                    self.last_action_hand = req.handedness
                    self.last_action_time = now

            # 5. Control 3: Right Click (Thumb + Middle Pinch Down)
            elif req.action == SemanticAction.RIGHT_CLICK:
                if self.mouse_controller.right_click():
                    self.last_action_name = "RIGHT CLICK"
                    self.last_action_hand = req.handedness
                    self.last_action_time = now

            # 6. Control 4: Play/Pause (Fist -> Open Palm)
            elif req.action == SemanticAction.PLAY_PAUSE:
                key = getattr(self.config, "play_pause_key", "space")
                cd = getattr(self.config, "play_pause_cooldown_sec", 0.8)
                if self.keyboard_controller.press(key, cooldown_sec=cd):
                    self.last_action_name = "PLAY/PAUSE"
                    self.last_action_hand = req.handedness
                    self.last_action_time = now

            # 7. Control 5: Next (Swipe Right)
            elif req.action == SemanticAction.NEXT:
                shortcut = getattr(self.config, "next_shortcut", ("shift", "n"))
                cd = getattr(self.config, "navigation_cooldown_sec", 0.6)
                if self.keyboard_controller.hotkey(*shortcut, cooldown_sec=cd):
                    self.last_action_name = "NEXT"
                    self.last_action_hand = req.handedness
                    self.last_action_time = now

            # 8. Control 6: Previous (Swipe Left)
            elif req.action == SemanticAction.PREVIOUS:
                shortcut = getattr(self.config, "previous_shortcut", ("shift", "p"))
                cd = getattr(self.config, "navigation_cooldown_sec", 0.6)
                if self.keyboard_controller.hotkey(*shortcut, cooldown_sec=cd):
                    self.last_action_name = "PREVIOUS"
                    self.last_action_hand = req.handedness
                    self.last_action_time = now

            # 9. Control 8: Zoom In (Two-Finger Up)
            elif req.action == SemanticAction.ZOOM_IN:
                shortcut = getattr(self.config, "zoom_in_shortcut", ("command", "+"))
                cd = getattr(self.config, "zoom_cooldown_sec", 0.5)
                if self.keyboard_controller.hotkey(*shortcut, cooldown_sec=cd):
                    self.last_action_name = "ZOOM IN"
                    self.last_action_hand = req.handedness
                    self.last_action_time = now

            # 10. Control 8: Zoom Out (Two-Finger Down)
            elif req.action == SemanticAction.ZOOM_OUT:
                shortcut = getattr(self.config, "zoom_out_shortcut", ("command", "-"))
                cd = getattr(self.config, "zoom_cooldown_sec", 0.5)
                if self.keyboard_controller.hotkey(*shortcut, cooldown_sec=cd):
                    self.last_action_name = "ZOOM OUT"
                    self.last_action_hand = req.handedness
                    self.last_action_time = now

        self.is_cursor_active = cursor_moved_this_frame and self.control_mode

        has_acc = self.mouse_controller.has_accessibility and self.keyboard_controller.has_accessibility

        return CommandEngineStatus(
            control_mode=self.control_mode,
            is_cursor_active=self.is_cursor_active,
            last_action_name=self.last_action_name,
            last_action_hand=self.last_action_hand,
            last_action_time=self.last_action_time,
            has_accessibility=has_acc,
        )

    def reset(self) -> None:
        """Reset internal states."""
        self.control_mode = False
        self.is_cursor_active = False
        self.last_action_name = "NONE"
        self.last_action_hand = ""
        self.mouse_controller.reset_smoother()
        self.action_mapper.reset()
        self.keyboard_controller.reset()
