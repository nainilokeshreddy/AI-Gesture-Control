"""
Hand Tracking Foundation using MediaPipe 1.0.1 Tasks API
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Detects up to 2 hands, extracts 21 3D landmarks per hand, determines handedness,
and returns structured tracking data for downstream gesture recognition.
"""

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.settings import DEFAULT_CONFIG, VisionConfig
from utils.logger import setup_logger

# Module-level variable to hold the Objective-C callback reference and avoid garbage collection
_objc_hook_holder = None


def _apply_macos_compatibility_hook() -> None:
    """
    On macOS Apple Silicon with MediaPipe 1.0.1, the internal DrishtiMetalHelper
    attempts to query GPU resources during CPU graph initialization, resulting
    in an unhandled SIGABRT ('Check failed: service_ Service is unavailable').

    This compatibility hook intercepts DrishtiMetalHelper.initWithCalculatorContext:
    via the native macOS Objective-C runtime (/usr/lib/libobjc.A.dylib) using ctypes,
    allowing the optimized TensorFlow Lite CPU pipeline to initialize smoothly.
    """
    if sys.platform != "darwin":
        return

    global _objc_hook_holder
    if _objc_hook_holder is not None:
        return  # Hook already installed

    import ctypes

    try:
        from mediapipe.tasks.python.core import mediapipe_c_bindings

        mediapipe_c_bindings.load_raw_library()
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.class_replaceMethod.restype = ctypes.c_void_p
        objc.class_replaceMethod.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]

        cls = objc.objc_getClass(b"DrishtiMetalHelper")
        if not cls:
            return
        sel = objc.sel_registerName(b"initWithCalculatorContext:")
        if not sel:
            return

        IMP_TYPE = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        )

        def _safe_init(self_ptr, _cmd, cc):
            return self_ptr

        _objc_hook_holder = IMP_TYPE(_safe_init)
        objc.class_replaceMethod(
            cls, sel, ctypes.cast(_objc_hook_holder, ctypes.c_void_p), b"@:@^v"
        )
    except Exception:
        # Fall back to standard initialization if patching is not applicable
        pass


class HandTrackerError(Exception):
    """Base exception for HandTracker operations."""
    pass


class ModelNotFoundError(HandTrackerError, FileNotFoundError):
    """Raised when the MediaPipe Hand Landmarker model file is missing."""
    pass


@dataclass
class HandLandmarkData:
    """
    Data container for a single detected hand.

    Attributes:
        landmarks: 21 normalized 3D landmarks [(x, y, z), ...], x,y in [0.0, 1.0].
        landmarks_px: 21 pixel coordinates [(px, py), ...] scaled to frame resolution.
        handedness: Classification label ('Left' or 'Right').
        score: Confidence score in [0.0, 1.0].
    """
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    landmarks_px: List[Tuple[int, int]] = field(default_factory=list)
    handedness: str = "Unknown"
    score: float = 0.0


@dataclass
class HandTrackingResult:
    """
    Aggregated tracking results for all hands detected in a single video frame.

    Attributes:
        hands: List of detected hands (up to max_num_hands).
        hand_count: Number of hands detected in the frame.
        timestamp_ms: Monotonic frame timestamp in milliseconds.
    """
    hands: List[HandLandmarkData] = field(default_factory=list)
    hand_count: int = 0
    timestamp_ms: int = 0


class HandTracker:
    """
    High-performance, reusable hand tracking abstraction powered by MediaPipe Tasks API.
    Detects up to 2 hands in real-time, extracts 21 landmarks, and determines handedness.
    """

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        model_path: Optional[str] = None,
        is_mirrored: Optional[bool] = None,
    ):
        """
        Initialize the HandTracker.

        Args:
            config: Optional VisionConfig instance.
            model_path: Optional path to the hand_landmarker.task model file.
            is_mirrored: Whether input frames are horizontally mirrored. If True,
                         compensates handedness inversion so physical Right is 'Right'.
        """
        self.config: VisionConfig = (
            config if config is not None else DEFAULT_CONFIG.vision
        )
        self.model_path: str = (
            model_path if model_path is not None else self.config.hand_model_path
        )
        self.is_mirrored: bool = (
            is_mirrored if is_mirrored is not None else getattr(self.config, "is_mirrored", True)
        )
        self.logger = setup_logger("HandTracker")
        self._landmarker: Optional[vision.HandLandmarker] = None
        self._is_initialized: bool = False
        self._last_timestamp_ms: int = 0
        self._active_tracks: List[Dict[str, Any]] = []

    def start(self) -> bool:
        """
        Instantiate the MediaPipe HandLandmarker task in VIDEO mode.

        Returns:
            True if successfully initialized.

        Raises:
            ModelNotFoundError: If the model file cannot be located.
            HandTrackerError: If task instantiation fails.
        """
        if self._is_initialized and self._landmarker is not None:
            self.logger.warning("HandTracker is already running.")
            return True

        resolved_model_path = os.path.abspath(self.model_path)
        if not os.path.isfile(resolved_model_path):
            err_msg = (
                f"Hand Landmarker model file not found at: '{resolved_model_path}'.\n"
                "Please ensure the official Google MediaPipe task model 'hand_landmarker.task' "
                "is downloaded to the 'models/' directory:\n"
                "  URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
            self.logger.error(err_msg)
            raise ModelNotFoundError(err_msg)

        # Apply macOS compatibility hook for MediaPipe 1.0.1
        _apply_macos_compatibility_hook()

        self.logger.info(
            f"Initializing HandLandmarker from model: {resolved_model_path} "
            f"(max_hands={self.config.max_num_hands}, min_det={self.config.hand_min_detection_confidence})"
        )

        try:
            base_options = python.BaseOptions(model_asset_path=resolved_model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.config.max_num_hands,
                min_hand_detection_confidence=self.config.hand_min_detection_confidence,
                min_hand_presence_confidence=self.config.hand_min_presence_confidence,
                min_tracking_confidence=self.config.hand_min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self._is_initialized = True
            self.logger.info("HandLandmarker initialized successfully.")
            return True
        except Exception as e:
            err_msg = f"Failed to instantiate MediaPipe HandLandmarker: {e}"
            self.logger.error(err_msg, exc_info=True)
            raise HandTrackerError(err_msg) from e

    def process(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
        rgb_frame: Optional[np.ndarray] = None,
    ) -> HandTrackingResult:
        """
        Process a single BGR camera frame and return detected hand landmarks.

        Args:
            frame_bgr: Input camera frame in BGR format (e.g. from Camera.read()).
            timestamp_ms: Optional monotonic timestamp in milliseconds. If omitted or <= previous,
                          a strictly increasing timestamp will be generated.
            rgb_frame: Optional pre-converted RGB frame to avoid redundant conversion.

        Returns:
            HandTrackingResult containing detected hands, 21 landmarks each, and handedness.
        """
        if not self._is_initialized or self._landmarker is None:
            self.logger.error("Attempted to process frame on uninitialized HandTracker.")
            return HandTrackingResult()

        if frame_bgr is None or frame_bgr.size == 0:
            return HandTrackingResult()

        h, w = frame_bgr.shape[:2]

        # Convert BGR frame to RGB for MediaPipe (re-use if pre-computed)
        if rgb_frame is not None:
            rgb_data = rgb_frame
        else:
            rgb_data = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_data)

        # Ensure monotonically increasing timestamp required by detect_for_video
        if timestamp_ms is None:
            current_ms = int(time.time() * 1000)
        else:
            current_ms = timestamp_ms

        if current_ms <= self._last_timestamp_ms:
            current_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current_ms

        try:
            detection_result = self._landmarker.detect_for_video(mp_image, current_ms)
        except Exception as e:
            self.logger.warning(f"Error during HandLandmarker video detection: {e}")
            return HandTrackingResult(timestamp_ms=current_ms)

        detected_hands: List[HandLandmarkData] = []

        if detection_result and detection_result.hand_landmarks:
            num_detected = len(detection_result.hand_landmarks)

            for idx in range(num_detected):
                raw_landmarks = detection_result.hand_landmarks[idx]
                landmarks_norm: List[Tuple[float, float, float]] = []
                landmarks_px: List[Tuple[int, int]] = []

                for lm in raw_landmarks:
                    landmarks_norm.append((lm.x, lm.y, lm.z))
                    px = max(0, min(int(lm.x * w), w - 1))
                    py = max(0, min(int(lm.y * h), h - 1))
                    landmarks_px.append((px, py))

                # Extract handedness if available
                handedness_label = "Unknown"
                handedness_score = 0.0
                if (
                    detection_result.handedness
                    and idx < len(detection_result.handedness)
                    and len(detection_result.handedness[idx]) > 0
                ):
                    top_category = detection_result.handedness[idx][0]
                    raw_label = (
                        top_category.category_name
                        or top_category.display_name
                        or "Unknown"
                    )
                    handedness_score = float(top_category.score)

                    # When camera preview is horizontally mirrored (selfie mode),
                    # MediaPipe's unmirrored coordinate model inverts physical handedness
                    # ('Left' <-> 'Right'). Compensate so reported handedness accurately
                    # reflects the user's physical hand:
                    if self.is_mirrored:
                        if raw_label.lower() == "left":
                            handedness_label = "Right"
                        elif raw_label.lower() == "right":
                            handedness_label = "Left"
                        else:
                            handedness_label = raw_label
                    else:
                        handedness_label = raw_label

                detected_hands.append(
                    HandLandmarkData(
                        landmarks=landmarks_norm,
                        landmarks_px=landmarks_px,
                        handedness=handedness_label,
                        score=handedness_score,
                    )
                )

        # Filter duplicate phantom detections on the same physical hand
        detected_hands = self._filter_phantom_duplicates(detected_hands)

        # Stabilize handedness labels across frames to prevent spurious flips
        detected_hands = self._stabilize_handedness(detected_hands, current_ms)

        return HandTrackingResult(
            hands=detected_hands,
            hand_count=len(detected_hands),
            timestamp_ms=current_ms,
        )

    def _filter_phantom_duplicates(
        self, detected_hands: List[HandLandmarkData]
    ) -> List[HandLandmarkData]:
        """
        Suppress phantom duplicate detections caused by MediaPipe detecting the same
        physical hand twice during pinches or finger articulations.
        """
        if len(detected_hands) <= 1:
            return detected_hands

        # Helper to compute approximate bounding box area
        def get_bbox_area(landmarks_px: List[Tuple[int, int]]) -> float:
            if not landmarks_px:
                return 0.0
            xs = [p[0] for p in landmarks_px]
            ys = [p[1] for p in landmarks_px]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))

        kept_hands: List[HandLandmarkData] = []
        suppressed_indices = set()

        for i in range(len(detected_hands)):
            if i in suppressed_indices:
                continue
            hand_i = detected_hands[i]
            pts_i = hand_i.landmarks_px
            if len(pts_i) < 21 or get_bbox_area(pts_i) < 400.0:
                kept_hands.append(hand_i)
                continue

            cx_i = sum(pts_i[k][0] for k in (0, 5, 9, 13, 17)) / 5.0
            cy_i = sum(pts_i[k][1] for k in (0, 5, 9, 13, 17)) / 5.0
            w_i = pts_i[0]

            for j in range(i + 1, len(detected_hands)):
                if j in suppressed_indices:
                    continue
                hand_j = detected_hands[j]
                pts_j = hand_j.landmarks_px
                if len(pts_j) < 21 or get_bbox_area(pts_j) < 400.0:
                    continue

                cx_j = sum(pts_j[k][0] for k in (0, 5, 9, 13, 17)) / 5.0
                cy_j = sum(pts_j[k][1] for k in (0, 5, 9, 13, 17)) / 5.0
                w_j = pts_j[0]

                dist_palm = math.hypot(cx_i - cx_j, cy_i - cy_j)
                dist_wrist = math.hypot(w_i[0] - w_j[0], w_i[1] - w_j[1])

                # If wrists or palms are very close, they are the same physical hand
                if dist_palm < 85.0 or dist_wrist < 75.0:
                    # Suppress the one with lower confidence score
                    if hand_i.score >= hand_j.score:
                        suppressed_indices.add(j)
                        self.logger.debug(
                            f"Suppressed phantom hand {j} ({hand_j.handedness} {hand_j.score:.2f}) "
                            f"in favor of hand {i} ({hand_i.handedness} {hand_i.score:.2f})"
                        )
                    else:
                        suppressed_indices.add(i)
                        self.logger.debug(
                            f"Suppressed phantom hand {i} ({hand_i.handedness} {hand_i.score:.2f}) "
                            f"in favor of hand {j} ({hand_j.handedness} {hand_j.score:.2f})"
                        )
                        break

            if i not in suppressed_indices:
                kept_hands.append(hand_i)

        return kept_hands

    def _stabilize_handedness(
        self, detected_hands: List[HandLandmarkData], current_ms: int
    ) -> List[HandLandmarkData]:
        """
        Stabilize handedness labels using spatial proximity and temporal consistency
        to prevent spurious 1-frame Left <-> Right flips during pinch gestures.
        """
        # Expire stale tracks (> 500 ms)
        self._active_tracks = [t for t in self._active_tracks if (current_ms - t["last_seen_ms"]) < 500]

        if not detected_hands:
            return detected_hands

        updated_hands: List[HandLandmarkData] = []

        for hand in detected_hands:
            pts = hand.landmarks_px
            if len(pts) < 21:
                updated_hands.append(hand)
                continue

            cx = sum(pts[j][0] for j in (0, 5, 9, 13, 17)) / 5.0
            cy = sum(pts[j][1] for j in (0, 5, 9, 13, 17)) / 5.0

            # Find matching active track within 140px
            best_match = None
            best_dist = 140.0
            for trk in self._active_tracks:
                d = math.hypot(cx - trk["palm_pos"][0], cy - trk["palm_pos"][1])
                if d < best_dist:
                    best_dist = d
                    best_match = trk

            if best_match is not None:
                # Update track position and timestamp
                best_match["palm_pos"] = (cx, cy)
                best_match["last_seen_ms"] = current_ms

                current_raw = hand.handedness
                stable_label = best_match["handedness"]

                if current_raw != stable_label:
                    if best_match["streak"] >= 3 and hand.score < 0.85:
                        # Reject transient flip during pinch gesture
                        hand.handedness = stable_label
                        best_match["opposite_count"] = best_match.get("opposite_count", 0) + 1
                        if best_match["opposite_count"] >= 4:
                            # Genuine sustained hand switch
                            best_match["handedness"] = current_raw
                            best_match["streak"] = 1
                            best_match["opposite_count"] = 0
                            hand.handedness = current_raw
                    else:
                        best_match["handedness"] = current_raw
                        best_match["streak"] = 1
                        best_match["opposite_count"] = 0
                else:
                    best_match["streak"] = min(20, best_match["streak"] + 1)
                    best_match["opposite_count"] = 0
                    hand.handedness = stable_label
            else:
                # New track
                self._active_tracks.append({
                    "handedness": hand.handedness,
                    "palm_pos": (cx, cy),
                    "streak": 1,
                    "opposite_count": 0,
                    "last_seen_ms": current_ms,
                })

            updated_hands.append(hand)

        return updated_hands

    def close(self) -> None:
        """Release MediaPipe native landmarker resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as e:
                self.logger.warning(f"Error closing HandLandmarker: {e}")
            finally:
                self._landmarker = None

        self._is_initialized = False
        self.logger.info("HandTracker closed cleanly.")

    @property
    def is_initialized(self) -> bool:
        """Check if HandTracker is active."""
        return self._is_initialized and self._landmarker is not None

    def __enter__(self) -> "HandTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        if self._is_initialized:
            self.close()
