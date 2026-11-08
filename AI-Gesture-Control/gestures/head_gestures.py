"""
Head Gesture Recognition & Movement Classification Module (Stage 5)
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Simplified Stage 5 Directional Head Tracking:
- HEAD LEFT: Physical head turn/tilt to the left (Yaw < threshold)
- HEAD RIGHT: Physical head turn/tilt to the right (Yaw > threshold)
- NEUTRAL: Head facing straight ahead within deadband

IMPORTANT DESIGN PRINCIPLES:
- Pitch and roll are strictly excluded from classification (yaw is the sole signal).
- Preserves neutral baseline calibration and relative yaw tracking.
- Implements strict state debouncing and neutral return hysteresis:
  A sustained movement emits exactly ONE event upon entry and does NOT spam events while held.
- Requires returning to neutral deadband before another discrete event can fire.
- SAFETY: Zero OS actions are executed from this layer (recognition-only).
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, Any

from config.settings import GestureThresholds, CooldownConfig, DEFAULT_CONFIG
from utils.logger import setup_logger
from vision.face_tracker import FaceTrackingResult, HeadPoseData


class HeadGestureID(str, Enum):
    """Directional head movement gesture identifiers (Stage 5: Left / Right)."""
    HEAD_LEFT = "HEAD_LEFT"
    HEAD_RIGHT = "HEAD_RIGHT"
    NONE = "NONE"

    # Backward compatibility aliases
    HEAD_TILT_LEFT = "HEAD_LEFT"
    HEAD_TILT_RIGHT = "HEAD_RIGHT"
    HEAD_TURN_LEFT = "HEAD_LEFT"
    HEAD_TURN_RIGHT = "HEAD_RIGHT"


def angle_diff(target: float, baseline: float) -> float:
    """Compute circular difference (target - baseline) wrapped into (-180.0, 180.0]."""
    return ((target - baseline + 180.0) % 360.0) - 180.0


@dataclass
class HeadGestureEvent:
    """Discrete head movement event emitted upon initial entry from neutral."""
    gesture_id: HeadGestureID
    label: str
    timestamp: float
    angles: Tuple[float, float, float]  # (yaw, pitch, roll) in degrees
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadGestureState:
    """Continuous orientation and state machine status for the head."""
    current_state: str = "NO FACE"       # "NEUTRAL", "HEAD LEFT", "HEAD RIGHT", or "NO FACE"
    active_gesture: HeadGestureID = HeadGestureID.NONE
    yaw: float = 0.0                     # Smoothed Yaw in degrees
    pitch: float = 0.0                   # Smoothed Pitch in degrees
    roll: float = 0.0                    # Smoothed Roll in degrees
    delta_yaw: float = 0.0               # Calibrated relative Yaw delta in degrees
    delta_pitch: float = 0.0             # Calibrated relative Pitch delta in degrees
    delta_roll: float = 0.0              # Calibrated relative Roll delta in degrees
    face_detected: bool = False
    active_event: Optional[HeadGestureEvent] = None
    last_event: Optional[HeadGestureEvent] = None


class HeadGestureRecognizer:
    """
    Evaluates HeadPoseData relative to a calibrated neutral baseline.
    Classifies directional movement:
    - HEAD LEFT: relative yaw < head_yaw_left_threshold (e.g. < -4.5 deg)
    - HEAD RIGHT: relative yaw > head_yaw_right_threshold (e.g. > +7.0 deg)
    - NEUTRAL: within deadband around calibrated neutral baseline
    """

    def __init__(
        self,
        thresholds: Optional[GestureThresholds] = None,
        cooldown: Optional[CooldownConfig] = None,
    ):
        self.thresholds: GestureThresholds = (
            thresholds if thresholds is not None else DEFAULT_CONFIG.thresholds
        )
        self.cooldown: CooldownConfig = (
            cooldown if cooldown is not None else DEFAULT_CONFIG.cooldown
        )
        self.logger = setup_logger("HeadGestureRecognizer")

        # Calibrated Neutral Baseline (degrees)
        self.baseline_yaw: float = getattr(self.thresholds, "head_baseline_yaw", -11.1)
        self.baseline_pitch: float = getattr(self.thresholds, "head_baseline_pitch", -8.6)
        self.baseline_roll: float = getattr(self.thresholds, "head_baseline_roll", 176.6)

        # Relative Yaw Thresholds for HEAD LEFT / HEAD RIGHT
        self.yaw_left_threshold: float = getattr(self.thresholds, "head_yaw_left_threshold", -4.5)
        self.yaw_right_threshold: float = getattr(self.thresholds, "head_yaw_right_threshold", 7.0)
        self.yaw_deadzone_hysteresis: float = getattr(self.thresholds, "head_yaw_deadzone_hysteresis", 0.7)
        self.deadzone_yaw: float = getattr(self.thresholds, "head_deadzone_yaw", 4.0)
        self.cooldown_sec: float = getattr(self.cooldown, "head_cooldown", 0.6)

        # State Debouncing & Neutral Return Management
        self._in_neutral: bool = True
        self._last_event_time: float = -10.0
        self._last_event: Optional[HeadGestureEvent] = None
        self._prev_state: str = "NO FACE"

    def process(
        self,
        face_result: FaceTrackingResult,
        timestamp: Optional[float] = None,
    ) -> HeadGestureState:
        """
        Process FaceTrackingResult against calibrated neutral baseline.
        Classifies HEAD LEFT, HEAD RIGHT, or NEUTRAL using relative yaw.
        """
        now = time.time() if timestamp is None else timestamp

        if not face_result.face_detected or face_result.head_pose is None:
            self._in_neutral = True
            self._prev_state = "NO FACE"
            return HeadGestureState(
                current_state="NO FACE",
                active_gesture=HeadGestureID.NONE,
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                delta_yaw=0.0,
                delta_pitch=0.0,
                delta_roll=0.0,
                face_detected=False,
                active_event=None,
                last_event=self._last_event,
            )

        pose = face_result.head_pose
        yaw = pose.yaw
        pitch = pose.pitch
        roll = pose.roll

        # Calculate relative deltas against calibrated baseline (circular difference)
        dy = angle_diff(yaw, self.baseline_yaw)
        dp = angle_diff(pitch, self.baseline_pitch)
        dr = angle_diff(roll, self.baseline_roll)

        # Evaluate candidate state strictly from relative yaw
        detected_gesture = HeadGestureID.NONE
        state_label = "NEUTRAL"

        if dy < self.yaw_left_threshold:
            state_label = "HEAD LEFT"
            detected_gesture = HeadGestureID.HEAD_LEFT
        elif dy > self.yaw_right_threshold:
            state_label = "HEAD RIGHT"
            detected_gesture = HeadGestureID.HEAD_RIGHT
        else:
            state_label = "NEUTRAL"
            detected_gesture = HeadGestureID.NONE

        # Debounce and Neutral Return Logic
        active_event: Optional[HeadGestureEvent] = None

        if detected_gesture != HeadGestureID.NONE:
            time_since_last = now - self._last_event_time
            if self._in_neutral and (time_since_last >= self.cooldown_sec):
                # First entry into directional movement from neutral
                active_event = HeadGestureEvent(
                    gesture_id=detected_gesture,
                    label=state_label,
                    timestamp=now,
                    angles=(yaw, pitch, roll),
                    details={"delta_yaw": dy, "delta_pitch": dp, "delta_roll": dr, "state": state_label},
                )
                self._last_event = active_event
                self._last_event_time = now
                self._in_neutral = False
                self.logger.info(
                    f"Head Gesture Recognized: {detected_gesture.value} ({state_label}) "
                    f"[dYaw: {dy:+.1f}°, dPitch: {dp:+.1f}°]"
                )
            else:
                # Posture is held: state remains HEAD LEFT or HEAD RIGHT, but no repeated events
                active_event = None
        else:
            # Within neutral deadband: check hysteresis threshold to re-arm entry from neutral
            return_threshold = abs(self.yaw_left_threshold) * self.yaw_deadzone_hysteresis
            if abs(dy) <= return_threshold:
                self._in_neutral = True
            active_event = None

        self._prev_state = state_label

        return HeadGestureState(
            current_state=state_label,
            active_gesture=detected_gesture,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            delta_yaw=dy,
            delta_pitch=dp,
            delta_roll=dr,
            face_detected=True,
            active_event=active_event,
            last_event=self._last_event,
        )

    def reset(self) -> None:
        """Reset state machine and debounce timers."""
        self._in_neutral = True
        self._last_event_time = -10.0
        self._last_event = None
        self._prev_state = "NO FACE"
