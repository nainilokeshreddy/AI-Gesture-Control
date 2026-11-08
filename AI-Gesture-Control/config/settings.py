"""
System Configuration & Constants
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Tuple


class ControlState(Enum):
    """Global system control states."""
    DISABLED = auto()   # Gestures tracked for HUD but no OS actions triggered
    ACTIVE = auto()     # Normal gesture-to-action execution


class GestureType(Enum):
    """
    Agreed 18-Control Specification Gesture Identifiers
    """
    # Hand Gestures (1-8)
    INDEX_CURSOR_MOVE = 1
    THUMB_INDEX_PINCH_LEFT_CLICK = 2
    TWO_FINGER_PINCH_RIGHT_CLICK = 3
    FIST_TO_OPEN_PALM_PLAY_PAUSE = 4
    SWIPE_RIGHT_NEXT = 5
    SWIPE_LEFT_PREVIOUS = 6
    STATIONARY_PALM_MODE_TOGGLE = 7
    TWO_FINGERS_VERTICAL_ZOOM = 8

    # Head Gestures (Stage 5 Foundation: Left / Right)
    HEAD_LEFT = 9
    HEAD_RIGHT = 10
    HEAD_TILT_LEFT_VOL_DOWN = 9   # Legacy alias
    HEAD_TILT_RIGHT_VOL_UP = 10   # Legacy alias

    # Eye Gestures (13-17)
    IRIS_LEFT_DWELL_PREV = 13
    IRIS_RIGHT_DWELL_NEXT = 14
    IRIS_UP_DWELL_MENU = 15
    IRIS_DOWN_DWELL_BACK = 16
    DOUBLE_BLINK_CONFIRM = 17

    # Multimodal (18)
    GAZE_DOUBLE_BLINK_SELECT = 18


class AppTarget(Enum):
    """Supported target applications for context-sensitive control."""
    GLOBAL = auto()
    CHROME = auto()
    YOUTUBE = auto()
    POWERPOINT = auto()


@dataclass
class CameraConfig:
    """Video capture settings."""
    device_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    mirror: bool = True  # Mirror camera feed for intuitive control
    window_title: str = "AI Gesture Control - Camera Test"


@dataclass
class VisionConfig:
    """MediaPipe detection and tracking thresholds."""
    # Hand Tracking
    max_num_hands: int = 2
    hand_min_detection_confidence: float = 0.7
    hand_min_presence_confidence: float = 0.5
    hand_min_tracking_confidence: float = 0.6
    hand_model_path: str = "models/hand_landmarker.task"
    is_mirrored: bool = True  # Compensate handedness for horizontally mirrored camera frames

    # Face & Mesh Tracking
    max_num_faces: int = 1
    face_min_detection_confidence: float = 0.7
    face_min_presence_confidence: float = 0.5
    face_min_tracking_confidence: float = 0.6
    face_model_path: str = "models/face_landmarker.task"
    refine_landmarks: bool = True  # Required for iris landmarks


@dataclass
class SmoothingConfig:
    """Cursor and coordinate smoothing parameters."""
    # Exponential Moving Average (EMA) alpha [0.0 = full smooth/lag, 1.0 = raw]
    ema_cursor_alpha: float = 0.35
    ema_head_alpha: float = 0.40
    ema_iris_alpha: float = 0.25

    # 1-Euro Filter parameters
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.007
    one_euro_d_cutoff: float = 1.0


@dataclass
class GestureThresholds:
    """Detection thresholds for gestures."""
    # Hand
    pinch_threshold_px: float = 40.0         # Distance between thumb & index tip
    pinch_release_threshold_px: float = 55.0 # Release distance for index pinch hysteresis
    middle_pinch_threshold_px: float = 45.0  # Distance between thumb & middle tip
    middle_pinch_release_threshold_px: float = 60.0 # Release distance for middle pinch hysteresis
    pinch_max_duration_sec: float = 0.5      # Max duration for quick contact pinch
    two_finger_pinch_px: float = 45.0        # Backward compatibility alias
    swipe_min_distance_px: float = 120.0     # Minimum horizontal displacement
    swipe_max_time_sec: float = 0.5          # Maximum time window for swipe
    swipe_dominance_ratio: float = 1.5       # |dx| must exceed ratio * |dy|
    palm_hold_duration_sec: float = 2.0      # Stationary palm hold for toggle
    palm_max_movement_px: float = 30.0       # Max movement allowed during hold
    fist_to_open_max_time_sec: float = 0.8   # Max transition window from fist to open palm
    vertical_two_finger_min_distance_px: float = 50.0 # Min vertical displacement for two-finger swipe
    vertical_dominance_ratio: float = 1.5    # |dy| must exceed ratio * |dx|
    vertical_max_time_sec: float = 0.5       # Max time window for two-finger vertical motion
    zoom_min_delta_px: float = 15.0          # Vertical two-finger delta for zoom step

    # Head Calibrated Neutral Baseline (degrees, based on camera calibration)
    head_baseline_yaw: float = -11.1
    head_baseline_pitch: float = -8.6
    head_baseline_roll: float = 176.6

    # Head Relative Movement Thresholds (degrees relative to neutral baseline)
    head_yaw_left_threshold: float = -4.5       # delta yaw < -4.5 deg -> HEAD LEFT
    head_yaw_right_threshold: float = 7.0       # delta yaw > +7.0 deg -> HEAD RIGHT
    head_yaw_deadzone_hysteresis: float = 0.7   # Multiplier for returning into neutral deadband
    head_deadzone_yaw: float = 4.0              # Max |delta yaw| for neutral deadband
    head_deadzone_pitch: float = 4.5            # Max |delta pitch| for neutral deadband

    # Head Backward-Compatibility Aliases
    tilt_angle_threshold_deg: float = 15.0   # Left / Right roll angle
    tilt_roll_threshold_deg: float = 15.0    # Left / Right roll angle
    turn_yaw_threshold_deg: float = 18.0     # Left / Right yaw angle
    head_neutral_hysteresis: float = 0.6     # Multiplier for returning into neutral deadband

    # Eyes
    ear_blink_threshold: float = 0.20        # Eye Aspect Ratio for blink
    ear_open_threshold: float = 0.26         # EAR required before registering next blink
    double_blink_max_interval_sec: float = 0.45  # Window for 2 consecutive blinks
    iris_dwell_duration_sec: float = 0.8     # Dwell time before iris trigger
    iris_horizontal_ratio: float = 0.35      # Ratio threshold away from eye center
    iris_vertical_ratio: float = 0.30        # Ratio threshold away from eye center


@dataclass
class CooldownConfig:
    """Debounce and cooldown durations (seconds) to prevent accidental repeated triggers."""
    default_cooldown: float = 0.4
    click_cooldown: float = 0.3
    mode_toggle_cooldown: float = 2.0
    swipe_cooldown: float = 0.6
    fist_open_cooldown: float = 0.8
    vertical_cooldown: float = 0.6
    volume_cooldown: float = 0.25
    head_cooldown: float = 0.6
    dwell_cooldown: float = 1.0
    blink_cooldown: float = 0.5


@dataclass
class ControlConfig:
    """OS level mouse and keyboard automation settings."""
    pyautogui_failsafe: bool = True
    pyautogui_pause: float = 0.01
    mouse_speed_factor_x: float = 1.6
    mouse_speed_factor_y: float = 1.6
    cursor_margin_percent: float = 0.15      # Boundary margin (inset box) for active tracking

    # Click settings & cooldowns
    click_cooldown_sec: float = 0.3
    right_click_cooldown_sec: float = 0.4

    # Media & Navigation key mappings
    play_pause_key: str = "space"
    next_shortcut: Tuple[str, ...] = ("shift", "n")
    previous_shortcut: Tuple[str, ...] = ("shift", "p")

    # Zoom key mappings (macOS standard: command + / command -)
    zoom_in_shortcut: Tuple[str, ...] = ("command", "+")
    zoom_out_shortcut: Tuple[str, ...] = ("command", "-")

    # Cooldown durations for discrete OS actions
    play_pause_cooldown_sec: float = 0.8
    navigation_cooldown_sec: float = 0.6
    zoom_cooldown_sec: float = 0.5
    emergency_lockout_sec: float = 1.5


@dataclass
class AppConfig:
    """Master Application Configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    thresholds: GestureThresholds = field(default_factory=GestureThresholds)
    cooldown: CooldownConfig = field(default_factory=CooldownConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    log_level: str = "INFO"


# Global default configuration instance
DEFAULT_CONFIG = AppConfig()
