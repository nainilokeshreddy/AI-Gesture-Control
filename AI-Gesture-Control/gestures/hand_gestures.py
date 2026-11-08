"""
Hand Gesture Recognition Layer
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Recognizes the 8 specified hand gestures:
1. HAND_INDEX_MOVE: Continuous index fingertip tracking with EMA coordinate smoothing.
2. HAND_PINCH_INDEX: Quick pinch between thumb (4) and index (8) with hysteresis.
3. HAND_PINCH_MIDDLE: Quick pinch between thumb (4) and middle (12) with hysteresis.
4. HAND_FIST_OPEN: Edge-triggered transition from fist to open palm.
5. HAND_SWIPE_RIGHT: Dominant horizontal rightward movement with sliding window.
6. HAND_SWIPE_LEFT: Dominant horizontal leftward movement with sliding window.
7. HAND_MODE_TOGGLE: Open palm held stationary for >= 2.0s with progress indicator.
8. HAND_VERTICAL_TWO_FINGER: Up/Down vertical motion with index + middle extended.

Supports independent multi-hand tracking (up to 2 hands) without OS level automation.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    DEFAULT_CONFIG,
    GestureThresholds,
    CooldownConfig,
    SmoothingConfig,
    GestureType,
)
from utils.cooldown import CooldownManager
from utils.logger import setup_logger
from utils.smoothing import ExponentialMovingAverage
from vision.hand_tracker import HandLandmarkData, HandTrackingResult


class HandGestureID(str, Enum):
    """Gesture identifiers for the 8 hand gestures."""
    HAND_INDEX_MOVE = "HAND_INDEX_MOVE"
    HAND_PINCH_INDEX = "HAND_PINCH_INDEX"
    HAND_PINCH_MIDDLE = "HAND_PINCH_MIDDLE"
    HAND_FIST_OPEN = "HAND_FIST_OPEN"
    HAND_SWIPE_RIGHT = "HAND_SWIPE_RIGHT"
    HAND_SWIPE_LEFT = "HAND_SWIPE_LEFT"
    HAND_MODE_TOGGLE = "HAND_MODE_TOGGLE"
    HAND_VERTICAL_TWO_FINGER = "HAND_VERTICAL_TWO_FINGER"


# Mapping from HandGestureID to 18-control specification GestureType
GESTURE_TYPE_MAP: Dict[HandGestureID, GestureType] = {
    HandGestureID.HAND_INDEX_MOVE: GestureType.INDEX_CURSOR_MOVE,
    HandGestureID.HAND_PINCH_INDEX: GestureType.THUMB_INDEX_PINCH_LEFT_CLICK,
    HandGestureID.HAND_PINCH_MIDDLE: GestureType.TWO_FINGER_PINCH_RIGHT_CLICK,
    HandGestureID.HAND_FIST_OPEN: GestureType.FIST_TO_OPEN_PALM_PLAY_PAUSE,
    HandGestureID.HAND_SWIPE_RIGHT: GestureType.SWIPE_RIGHT_NEXT,
    HandGestureID.HAND_SWIPE_LEFT: GestureType.SWIPE_LEFT_PREVIOUS,
    HandGestureID.HAND_MODE_TOGGLE: GestureType.STATIONARY_PALM_MODE_TOGGLE,
    HandGestureID.HAND_VERTICAL_TWO_FINGER: GestureType.TWO_FINGERS_VERTICAL_ZOOM,
}


@dataclass
class HandGestureEvent:
    """Discrete gesture event emitted at a specific point in time."""
    gesture_id: HandGestureID
    gesture_type: GestureType
    handedness: str
    label: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandGestureState:
    """Continuous gesture status for a single detected hand."""
    handedness: str = "Unknown"
    active_gesture: Optional[HandGestureID] = None
    active_gesture_label: str = "NONE"
    smoothed_index_pos: Optional[Tuple[int, int]] = None
    raw_index_pos: Optional[Tuple[int, int]] = None
    palm_center_pos: Optional[Tuple[int, int]] = None
    is_index_pointing: bool = False
    is_pinching_index: bool = False
    is_pinching_middle: bool = False
    is_open_palm: bool = False
    is_fist: bool = False
    is_two_finger_pointing: bool = False
    hold_progress: float = 0.0    # 0.0 to 1.0 (for 2.0s hold)
    hold_duration: float = 0.0    # elapsed seconds
    last_event: Optional[HandGestureEvent] = None
    finger_states: Dict[str, bool] = field(default_factory=dict)  # finger_name -> extended


@dataclass
class GestureRecognitionResult:
    """Aggregated gesture recognition result for all hands in the current frame."""
    hands: Dict[str, HandGestureState] = field(default_factory=dict)
    events: List[HandGestureEvent] = field(default_factory=list)
    timestamp: float = 0.0


def _euclidean_dist_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate 2D Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class SingleHandTracker:
    """
    Maintains independent temporal state, trajectory buffers, smoothing filters,
    and gesture detectors for a single hand (Left or Right).
    """

    def __init__(
        self,
        handedness: str,
        thresholds: Optional[GestureThresholds] = None,
        cooldowns: Optional[CooldownConfig] = None,
        smoothing: Optional[SmoothingConfig] = None,
    ):
        self.handedness = handedness
        self.thresholds = thresholds or DEFAULT_CONFIG.thresholds
        self.cooldowns = cooldowns or DEFAULT_CONFIG.cooldown
        self.smoothing = smoothing or DEFAULT_CONFIG.smoothing

        # Coordinate smoother for index fingertip
        self.index_smoother = ExponentialMovingAverage(alpha=self.smoothing.ema_cursor_alpha)

        # Cooldown manager for discrete events
        self.cooldown_mgr = CooldownManager(default_cooldown=self.cooldowns.default_cooldown)

        # Sliding window trajectory: deque of (timestamp, x, y)
        self.palm_trajectory: Deque[Tuple[float, float, float]] = deque()
        self.two_finger_trajectory: Deque[Tuple[float, float, float]] = deque()

        # Pinch state machines
        self._index_pinch_active: bool = False
        self._index_pinch_start_time: float = 0.0

        self._middle_pinch_active: bool = False
        self._middle_pinch_start_time: float = 0.0

        # Fist to Open Palm state tracker
        self._fist_armed: bool = False
        self._last_fist_time: Optional[float] = None

        # Stationary palm hold (Mode Toggle) state tracker
        self._hold_anchor: Optional[Tuple[float, float]] = None
        self._hold_start_time: Optional[float] = None
        self._hold_triggered: bool = False

        # Persistent last event for UI display
        self._last_event: Optional[HandGestureEvent] = None

    def reset(self) -> None:
        """Reset internal temporal buffers and state machines."""
        self.index_smoother.reset()
        self.palm_trajectory.clear()
        self.two_finger_trajectory.clear()
        self._index_pinch_active = False
        self._middle_pinch_active = False
        self._fist_armed = False
        self._last_fist_time = None
        self._hold_anchor = None
        self._hold_start_time = None
        self._hold_triggered = False
        self._last_event = None

    def classify_fingers(
        self, landmarks_px: List[Tuple[int, int]]
    ) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        """
        Classify each finger as extended or curled based on 2D landmarks.

        Returns:
            extended_dict: {'thumb': bool, 'index': bool, 'middle': bool, 'ring': bool, 'pinky': bool}
            curled_dict:   {'thumb': bool, 'index': bool, 'middle': bool, 'ring': bool, 'pinky': bool}
        """
        w = landmarks_px[0]    # Wrist
        mcp9 = landmarks_px[9] # Middle finger MCP
        palm_size = max(20.0, _euclidean_dist_2d(w, mcp9))

        extended = {}
        curled = {}

        # 4 Fingers: Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
        finger_indices = {
            "index": (5, 6, 8),
            "middle": (9, 10, 12),
            "ring": (13, 14, 16),
            "pinky": (17, 18, 20),
        }

        for name, (mcp_idx, pip_idx, tip_idx) in finger_indices.items():
            mcp = landmarks_px[mcp_idx]
            pip = landmarks_px[pip_idx]
            tip = landmarks_px[tip_idx]

            dist_tip_wrist = _euclidean_dist_2d(tip, w)
            dist_pip_wrist = _euclidean_dist_2d(pip, w)
            dist_tip_mcp = _euclidean_dist_2d(tip, mcp)
            dist_pip_mcp = _euclidean_dist_2d(pip, mcp)

            # Extended: tip is farther from wrist than PIP and farther from MCP than PIP
            is_ext = (dist_tip_wrist > dist_pip_wrist * 1.05) and (dist_tip_mcp > dist_pip_mcp)
            # Curled: tip is closer to wrist than PIP or folded inward toward MCP
            is_crl = (dist_tip_wrist < dist_pip_wrist) or (dist_tip_mcp < dist_pip_mcp * 0.85)

            extended[name] = is_ext
            curled[name] = is_crl

        # Thumb: Wrist 0, CMC 1, MCP 2, IP 3, Tip 4
        # Reference against Pinky MCP 17 and Index MCP 5
        t_tip = landmarks_px[4]
        t_ip = landmarks_px[3]
        t_mcp = landmarks_px[2]
        p_mcp = landmarks_px[17]
        i_mcp = landmarks_px[5]

        dist_t_pinky = _euclidean_dist_2d(t_tip, p_mcp)
        dist_tip_index_mcp = _euclidean_dist_2d(t_tip, i_mcp)
        dist_mcp_pinky = _euclidean_dist_2d(t_mcp, p_mcp)

        # Extended thumb: Tip is held away from palm / pinky base
        thumb_ext = (dist_t_pinky > dist_mcp_pinky * 1.05) and (dist_tip_index_mcp > palm_size * 0.45)
        # Curled thumb: Tucked inward close to palm or index MCP
        thumb_crl = (dist_t_pinky <= dist_mcp_pinky * 1.05) or (dist_tip_index_mcp < palm_size * 0.40)

        extended["thumb"] = thumb_ext
        curled["thumb"] = thumb_crl

        return extended, curled

    def update(
        self,
        hand_data: HandLandmarkData,
        current_time: Optional[float] = None,
    ) -> Tuple[HandGestureState, List[HandGestureEvent]]:
        """
        Process a new frame of landmarks for this hand and return updated state and discrete events.
        """
        now = current_time if current_time is not None else time.time()
        events: List[HandGestureEvent] = []
        pts = hand_data.landmarks_px

        if len(pts) < 21:
            return HandGestureState(handedness=self.handedness), events

        # 1. Calculate Palm Center: centroid of Wrist (0) and 4 MCP joints (5, 9, 13, 17)
        palm_pts = [pts[0], pts[5], pts[9], pts[13], pts[17]]
        palm_cx = sum(p[0] for p in palm_pts) / 5.0
        palm_cy = sum(p[1] for p in palm_pts) / 5.0
        palm_center = (int(round(palm_cx)), int(round(palm_cy)))

        # 2. Smooth Index Fingertip Coordinates (Landmark 8)
        raw_index_pos = pts[8]
        smoothed_arr = self.index_smoother.filter(np.array(raw_index_pos, dtype=np.float64))
        smoothed_index_pos = (int(round(smoothed_arr[0])), int(round(smoothed_arr[1])))

        # 3. Classify Finger Postures
        ext, crl = self.classify_fingers(pts)

        # Posture definitions
        is_open_palm = ext["index"] and ext["middle"] and ext["ring"] and ext["pinky"]
        is_fist = crl["index"] and crl["middle"] and crl["ring"] and crl["pinky"]
        is_two_finger_pointing = ext["index"] and ext["middle"] and crl["ring"] and crl["pinky"]
        is_index_pointing = ext["index"] and not ext["middle"] and not is_open_palm

        # 4. Trajectory Buffers Management
        self.palm_trajectory.append((now, palm_cx, palm_cy))
        # Prune older points beyond max swipe window
        while self.palm_trajectory and (now - self.palm_trajectory[0][0]) > self.thresholds.swipe_max_time_sec:
            self.palm_trajectory.popleft()

        # Two-finger midpoint for vertical tracking
        if is_two_finger_pointing:
            mid_x = (pts[8][0] + pts[12][0]) / 2.0
            mid_y = (pts[8][1] + pts[12][1]) / 2.0
            self.two_finger_trajectory.append((now, mid_x, mid_y))
            while self.two_finger_trajectory and (now - self.two_finger_trajectory[0][0]) > self.thresholds.vertical_max_time_sec:
                self.two_finger_trajectory.popleft()
        else:
            self.two_finger_trajectory.clear()

        # =========================================================================
        # 5. GESTURE DETECTION LOGIC
        # =========================================================================

        # --- A. PINCH DETECTION (Thumb + Index & Thumb + Middle) ---
        thumb_tip = pts[4]
        index_tip = pts[8]
        middle_tip = pts[12]

        dist_thumb_index = _euclidean_dist_2d(thumb_tip, index_tip)
        dist_thumb_middle = _euclidean_dist_2d(thumb_tip, middle_tip)

        # A closed fist curls all fingertips into the palm and should not trigger pinches
        if is_fist:
            self._index_pinch_active = False
            self._middle_pinch_active = False

        # Index Pinch with Hysteresis
        if not self._index_pinch_active:
            if not is_fist and dist_thumb_index < self.thresholds.pinch_threshold_px:
                self._index_pinch_active = True
                self._index_pinch_start_time = now
        else:
            if dist_thumb_index > self.thresholds.pinch_release_threshold_px:
                contact_duration = now - self._index_pinch_start_time
                self._index_pinch_active = False
                if (
                    contact_duration <= self.thresholds.pinch_max_duration_sec
                    and self.cooldown_mgr.is_ready(HandGestureID.HAND_PINCH_INDEX, self.cooldowns.click_cooldown)
                ):
                    self.cooldown_mgr.trigger(HandGestureID.HAND_PINCH_INDEX)
                    ev = HandGestureEvent(
                        gesture_id=HandGestureID.HAND_PINCH_INDEX,
                        gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_PINCH_INDEX],
                        handedness=self.handedness,
                        label="INDEX PINCH",
                        timestamp=now,
                        details={"duration": contact_duration, "distance": dist_thumb_index},
                    )
                    events.append(ev)
                    self._last_event = ev

        # Middle Pinch with Hysteresis
        # Disallowed if fist, middle finger is curled into palm, index pinch is active, or index is closer
        index_near = dist_thumb_index < (self.thresholds.pinch_threshold_px + 10.0)
        if not self._middle_pinch_active:
            if (
                not is_fist
                and not crl["middle"]
                and not self._index_pinch_active
                and not index_near
                and dist_thumb_middle < self.thresholds.middle_pinch_threshold_px
            ):
                self._middle_pinch_active = True
                self._middle_pinch_start_time = now
        else:
            if dist_thumb_middle > self.thresholds.middle_pinch_release_threshold_px:
                contact_duration = now - self._middle_pinch_start_time
                self._middle_pinch_active = False
                if (
                    contact_duration <= self.thresholds.pinch_max_duration_sec
                    and self.cooldown_mgr.is_ready(HandGestureID.HAND_PINCH_MIDDLE, self.cooldowns.click_cooldown)
                ):
                    self.cooldown_mgr.trigger(HandGestureID.HAND_PINCH_MIDDLE)
                    ev = HandGestureEvent(
                        gesture_id=HandGestureID.HAND_PINCH_MIDDLE,
                        gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_PINCH_MIDDLE],
                        handedness=self.handedness,
                        label="MIDDLE PINCH",
                        timestamp=now,
                        details={"duration": contact_duration, "distance": dist_thumb_middle},
                    )
                    events.append(ev)
                    self._last_event = ev

        # --- B. FIST TO OPEN PALM TRANSITION ---
        if is_fist:
            self._last_fist_time = now
            self._fist_armed = True
        elif is_open_palm:
            if self._fist_armed and self._last_fist_time is not None:
                elapsed = now - self._last_fist_time
                if (
                    elapsed <= self.thresholds.fist_to_open_max_time_sec
                    and self.cooldown_mgr.is_ready(HandGestureID.HAND_FIST_OPEN, self.cooldowns.fist_open_cooldown)
                ):
                    self.cooldown_mgr.trigger(HandGestureID.HAND_FIST_OPEN)
                    ev = HandGestureEvent(
                        gesture_id=HandGestureID.HAND_FIST_OPEN,
                        gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_FIST_OPEN],
                        handedness=self.handedness,
                        label="FIST -> OPEN PALM DETECTED",
                        timestamp=now,
                        details={"transition_time": elapsed},
                    )
                    events.append(ev)
                    self._last_event = ev
                self._fist_armed = False
                self._last_fist_time = None
        else:
            if self._last_fist_time is not None and (now - self._last_fist_time) > self.thresholds.fist_to_open_max_time_sec:
                self._fist_armed = False
                self._last_fist_time = None

        # --- C. HORIZONTAL SWIPES (RIGHT & LEFT) ---
        can_swipe_right = self.cooldown_mgr.is_ready(HandGestureID.HAND_SWIPE_RIGHT, self.cooldowns.swipe_cooldown)
        can_swipe_left = self.cooldown_mgr.is_ready(HandGestureID.HAND_SWIPE_LEFT, self.cooldowns.swipe_cooldown)

        if (can_swipe_right or can_swipe_left) and len(self.palm_trajectory) >= 2:
            curr_t, curr_x, curr_y = self.palm_trajectory[-1]
            for prev_t, prev_x, prev_y in self.palm_trajectory:
                dt = curr_t - prev_t
                if dt < 0.08:  # Minimum time delta to avoid micro-jitter
                    continue

                dx = curr_x - prev_x
                dy = curr_y - prev_y
                abs_dx = abs(dx)
                abs_dy = abs(dy)

                # Horizontal dominance check
                if abs_dx >= self.thresholds.swipe_min_distance_px and abs_dx > (self.thresholds.swipe_dominance_ratio * abs_dy):
                    if dx > 0 and can_swipe_right:
                        self.cooldown_mgr.trigger(HandGestureID.HAND_SWIPE_RIGHT)
                        ev = HandGestureEvent(
                            gesture_id=HandGestureID.HAND_SWIPE_RIGHT,
                            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_SWIPE_RIGHT],
                            handedness=self.handedness,
                            label="SWIPE RIGHT",
                            timestamp=now,
                            details={"dx": dx, "dy": dy, "duration": dt},
                        )
                        events.append(ev)
                        self._last_event = ev
                        self.palm_trajectory.clear()
                        break
                    elif dx < 0 and can_swipe_left:
                        self.cooldown_mgr.trigger(HandGestureID.HAND_SWIPE_LEFT)
                        ev = HandGestureEvent(
                            gesture_id=HandGestureID.HAND_SWIPE_LEFT,
                            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_SWIPE_LEFT],
                            handedness=self.handedness,
                            label="SWIPE LEFT",
                            timestamp=now,
                            details={"dx": dx, "dy": dy, "duration": dt},
                        )
                        events.append(ev)
                        self._last_event = ev
                        self.palm_trajectory.clear()
                        break

        # --- D. STATIONARY OPEN PALM (2-SECOND HOLD / MODE TOGGLE) ---
        hold_progress = 0.0
        hold_duration = 0.0

        if is_open_palm:
            if self._hold_triggered:
                # Already triggered for this continuous open palm posture; hold progress maxed
                hold_progress = 1.0
                hold_duration = now - (self._hold_start_time or now)
            elif self._hold_start_time is None:
                self._hold_anchor = (palm_cx, palm_cy)
                self._hold_start_time = now
            else:
                # Check jitter/movement against anchor
                anchor_dist = _euclidean_dist_2d((palm_cx, palm_cy), self._hold_anchor)
                if anchor_dist > self.thresholds.palm_max_movement_px:
                    # Moved too much! Reset hold anchor and timer
                    self._hold_anchor = (palm_cx, palm_cy)
                    self._hold_start_time = now
                    hold_progress = 0.0
                    hold_duration = 0.0
                else:
                    elapsed = now - self._hold_start_time
                    hold_duration = elapsed
                    hold_progress = min(1.0, elapsed / self.thresholds.palm_hold_duration_sec)
                    if elapsed >= self.thresholds.palm_hold_duration_sec:
                        self._hold_triggered = True
                        if self.cooldown_mgr.is_ready(HandGestureID.HAND_MODE_TOGGLE, self.cooldowns.mode_toggle_cooldown):
                            self.cooldown_mgr.trigger(HandGestureID.HAND_MODE_TOGGLE)
                            ev = HandGestureEvent(
                                gesture_id=HandGestureID.HAND_MODE_TOGGLE,
                                gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_MODE_TOGGLE],
                                handedness=self.handedness,
                                label="MODE TOGGLE READY",
                                timestamp=now,
                                details={"hold_time": elapsed},
                            )
                            events.append(ev)
                            self._last_event = ev
        else:
            # Leaving open palm resets the latch and hold timers
            self._hold_anchor = None
            self._hold_start_time = None
            self._hold_triggered = False
            hold_progress = 0.0
            hold_duration = 0.0

        # --- E. TWO-FINGER VERTICAL MOVEMENT ---
        if is_two_finger_pointing and len(self.two_finger_trajectory) >= 2:
            if self.cooldown_mgr.is_ready(HandGestureID.HAND_VERTICAL_TWO_FINGER, self.cooldowns.vertical_cooldown):
                curr_t, curr_x, curr_y = self.two_finger_trajectory[-1]
                for prev_t, prev_x, prev_y in self.two_finger_trajectory:
                    dt = curr_t - prev_t
                    if dt < 0.08:
                        continue

                    dx = curr_x - prev_x
                    dy = curr_y - prev_y
                    abs_dx = abs(dx)
                    abs_dy = abs(dy)

                    if abs_dy >= self.thresholds.vertical_two_finger_min_distance_px and abs_dy > (self.thresholds.vertical_dominance_ratio * abs_dx):
                        self.cooldown_mgr.trigger(HandGestureID.HAND_VERTICAL_TWO_FINGER)
                        direction = "UP" if dy < 0 else "DOWN"
                        ev = HandGestureEvent(
                            gesture_id=HandGestureID.HAND_VERTICAL_TWO_FINGER,
                            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_VERTICAL_TWO_FINGER],
                            handedness=self.handedness,
                            label=f"TWO-FINGER {direction}",
                            timestamp=now,
                            details={"direction": direction, "dy": dy, "dx": dx, "duration": dt},
                        )
                        events.append(ev)
                        self._last_event = ev
                        self.two_finger_trajectory.clear()
                        break

        # --- F. DETERMINE ACTIVE CONTINUOUS GESTURE ---
        active_gesture: Optional[HandGestureID] = None
        active_label: str = "NONE"

        if self._index_pinch_active:
            active_gesture = HandGestureID.HAND_PINCH_INDEX
            active_label = "INDEX PINCH"
        elif self._middle_pinch_active:
            active_gesture = HandGestureID.HAND_PINCH_MIDDLE
            active_label = "MIDDLE PINCH"
        elif is_two_finger_pointing:
            active_gesture = HandGestureID.HAND_VERTICAL_TWO_FINGER
            active_label = "TWO-FINGER POINTING"
        elif is_open_palm:
            if hold_progress > 0.0:
                active_gesture = HandGestureID.HAND_MODE_TOGGLE
                active_label = f"PALM HOLD ({int(hold_progress * 100)}%)"
            else:
                active_label = "OPEN PALM"
        elif is_fist:
            active_label = "FIST"
        elif is_index_pointing or ext["index"]:
            active_gesture = HandGestureID.HAND_INDEX_MOVE
            active_label = "INDEX MOVE"

        state = HandGestureState(
            handedness=self.handedness,
            active_gesture=active_gesture,
            active_gesture_label=active_label,
            smoothed_index_pos=smoothed_index_pos,
            raw_index_pos=raw_index_pos,
            palm_center_pos=palm_center,
            is_index_pointing=is_index_pointing or ext["index"],
            is_pinching_index=self._index_pinch_active,
            is_pinching_middle=self._middle_pinch_active,
            is_open_palm=is_open_palm,
            is_fist=is_fist,
            is_two_finger_pointing=is_two_finger_pointing,
            hold_progress=hold_progress,
            hold_duration=hold_duration,
            last_event=self._last_event,
            finger_states=ext,
        )

        return state, events


class HandGestureRecognizer:
    """
    Coordinator managing independent tracking and gesture recognition for up to 2 hands.
    Ensures Left and Right hand states never collide or cross-contaminate.
    """

    def __init__(
        self,
        thresholds: Optional[GestureThresholds] = None,
        cooldowns: Optional[CooldownConfig] = None,
        smoothing: Optional[SmoothingConfig] = None,
    ):
        self.thresholds = thresholds or DEFAULT_CONFIG.thresholds
        self.cooldowns = cooldowns or DEFAULT_CONFIG.cooldown
        self.smoothing = smoothing or DEFAULT_CONFIG.smoothing
        self.logger = setup_logger("HandGestureRecognizer")

        # Independent trackers keyed by handedness ("Left", "Right", "Unknown")
        self._trackers: Dict[str, SingleHandTracker] = {}

    def _get_tracker(self, handedness: str) -> SingleHandTracker:
        """Retrieve or create an independent SingleHandTracker instance."""
        if handedness not in self._trackers:
            self._trackers[handedness] = SingleHandTracker(
                handedness=handedness,
                thresholds=self.thresholds,
                cooldowns=self.cooldowns,
                smoothing=self.smoothing,
            )
        return self._trackers[handedness]

    def process(
        self,
        hand_result: HandTrackingResult,
        timestamp: Optional[float] = None,
    ) -> GestureRecognitionResult:
        """
        Evaluate hand tracking data from a frame and return detected gestures for all hands.

        Args:
            hand_result: HandTrackingResult from HandTracker.process().
            timestamp: Optional monotonic timestamp in seconds (defaults to time.time()).

        Returns:
            GestureRecognitionResult containing per-hand states and discrete events.
        """
        now = timestamp if timestamp is not None else time.time()
        result = GestureRecognitionResult(timestamp=now)

        if not hand_result or hand_result.hand_count == 0:
            # No hands in frame: reset temporal tracking for all existing trackers
            for tracker in self._trackers.values():
                tracker.reset()
            return result

        detected_handedness = set()

        for hand_data in hand_result.hands:
            h_label = hand_data.handedness
            detected_handedness.add(h_label)
            tracker = self._get_tracker(h_label)
            hand_state, events = tracker.update(hand_data, current_time=now)
            result.hands[h_label] = hand_state
            result.events.extend(events)

        # Reset trackers for any hand that wasn't seen in this frame
        for h_label, tracker in self._trackers.items():
            if h_label not in detected_handedness:
                tracker.reset()

        return result

    def reset(self) -> None:
        """Reset all hand trackers."""
        for tracker in self._trackers.values():
            tracker.reset()
        self._trackers.clear()
