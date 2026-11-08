"""
Face Tracker & Head Pose Estimation Module (Stage 5)
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Utilizes MediaPipe 1.0.1 FaceLandmarker Task in VIDEO mode to extract 478 3D facial
landmarks, and estimates 3D head pose orientation (Yaw, Pitch, Roll in degrees)
via Perspective-n-Point (cv2.solvePnP) and Euler angle decomposition (cv2.RQDecomp3x3).
"""

import os
import time
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
    RunningMode,
)

from config.settings import VisionConfig, SmoothingConfig, DEFAULT_CONFIG
from utils.logger import setup_logger
from utils.smoothing import ExponentialMovingAverage


class FaceTrackerError(Exception):
    """Base exception for FaceTracker pipeline errors."""
    pass


class ModelNotFoundError(FaceTrackerError):
    """Raised when the MediaPipe Face Landmarker model asset is not found."""
    pass


@dataclass
class FaceLandmarkData:
    """Detailed facial landmark data for a single detected face."""
    landmarks: List[Tuple[float, float, float]]      # 478 Normalized landmarks (x, y, z)
    landmarks_px: List[Tuple[int, int]]              # 478 Pixel landmarks (x, y)
    bbox: Tuple[int, int, int, int]                  # (min_x, min_y, max_x, max_y)
    center_px: Tuple[int, int]                       # Approximate face center (x, y)
    confidence: float = 1.0                          # Detection confidence

    @property
    def nose_tip_px(self) -> Tuple[int, int]:
        """Landmark 1: Nose tip."""
        return self.landmarks_px[1] if len(self.landmarks_px) > 1 else (0, 0)

    @property
    def chin_px(self) -> Tuple[int, int]:
        """Landmark 152: Chin."""
        return self.landmarks_px[152] if len(self.landmarks_px) > 152 else (0, 0)

    @property
    def screen_right_eye_px(self) -> Tuple[int, int]:
        """Landmark 263: Outer corner of eye on screen right (subject's right in mirrored view)."""
        return self.landmarks_px[263] if len(self.landmarks_px) > 263 else (0, 0)

    @property
    def screen_left_eye_px(self) -> Tuple[int, int]:
        """Landmark 33: Outer corner of eye on screen left (subject's left in mirrored view)."""
        return self.landmarks_px[33] if len(self.landmarks_px) > 33 else (0, 0)

    @property
    def screen_right_mouth_px(self) -> Tuple[int, int]:
        """Landmark 291: Mouth corner on screen right."""
        return self.landmarks_px[291] if len(self.landmarks_px) > 291 else (0, 0)

    @property
    def screen_left_mouth_px(self) -> Tuple[int, int]:
        """Landmark 61: Mouth corner on screen left."""
        return self.landmarks_px[61] if len(self.landmarks_px) > 61 else (0, 0)


@dataclass
class HeadPoseData:
    """Estimated head orientation angles in degrees and 3D PnP pose vectors."""
    yaw: float              # Smoothed Yaw in degrees (positive = Turn Right, negative = Turn Left)
    pitch: float            # Smoothed Pitch in degrees (positive = Nod Down, negative = Nod Up)
    roll: float             # Smoothed Roll in degrees (positive = Tilt Right, negative = Tilt Left)
    raw_yaw: float          # Unfiltered Yaw
    raw_pitch: float        # Unfiltered Pitch
    raw_roll: float         # Unfiltered Roll
    rvec: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    tvec: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))


@dataclass
class FaceTrackingResult:
    """Aggregated result of face detection, landmark extraction, and head pose estimation."""
    face_detected: bool
    face_data: Optional[FaceLandmarkData] = None
    head_pose: Optional[HeadPoseData] = None
    timestamp_ms: int = 0


# 3D Anthropometric Facial Model Points in millimetres (head reference coordinate frame)
# Reference: Canonical 3D Face Model centered at the nose tip with OpenCV image-aligned frame
# (X: Right positive, Y: Down positive, Z: Negative behind nose tip)
CANONICAL_FACE_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
    (0.0, 330.0, -65.0),         # Chin (landmark 152) -> Y positive DOWN
    (225.0, -170.0, -135.0),     # Screen Right eye outer corner (landmark 263 in mirrored selfie) -> Y negative UP
    (-225.0, -170.0, -135.0),    # Screen Left eye outer corner (landmark 33 in mirrored selfie) -> Y negative UP
    (150.0, 150.0, -125.0),      # Screen Right mouth corner (landmark 291 in mirrored selfie) -> Y positive DOWN
    (-150.0, 150.0, -125.0),     # Screen Left mouth corner (landmark 61 in mirrored selfie) -> Y positive DOWN
], dtype=np.float64)


class FaceTracker:
    """
    Reusable MediaPipe FaceLandmarker pipeline for face detection and 3D head pose tracking.
    """

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        smoothing_config: Optional[SmoothingConfig] = None,
        model_path: Optional[str] = None,
    ):
        self.config: VisionConfig = config if config is not None else DEFAULT_CONFIG.vision
        self.smoothing_config: SmoothingConfig = (
            smoothing_config if smoothing_config is not None else DEFAULT_CONFIG.smoothing
        )
        self.model_path: str = model_path if model_path is not None else self.config.face_model_path
        self.logger = setup_logger("FaceTracker")

        self._landmarker: Optional[FaceLandmarker] = None
        self._is_initialized: bool = False
        self._last_timestamp_ms: int = 0

        # Temporal smoother for head pose angles (yaw, pitch, roll)
        self._pose_smoother = ExponentialMovingAverage(alpha=self.smoothing_config.ema_head_alpha)

    @property
    def is_initialized(self) -> bool:
        """True if the MediaPipe landmarker is running."""
        return self._is_initialized

    def start(self) -> bool:
        """
        Instantiate MediaPipe FaceLandmarker in VIDEO running mode.
        """
        if self._is_initialized and self._landmarker is not None:
            self.logger.warning("FaceTracker is already running.")
            return True

        resolved_model_path = os.path.abspath(self.model_path)
        if not os.path.isfile(resolved_model_path):
            err_msg = (
                f"Face Landmarker model file not found at: '{resolved_model_path}'.\n"
                "Please ensure the official Google MediaPipe task model 'face_landmarker.task' "
                "is downloaded to the 'models/' directory:\n"
                "  URL: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
            self.logger.error(err_msg)
            raise ModelNotFoundError(err_msg)

        # Apply macOS compatibility hook for MediaPipe 1.0.1
        try:
            from vision.hand_tracker import _apply_macos_compatibility_hook
            _apply_macos_compatibility_hook()
        except Exception:
            pass

        try:
            self.logger.info(
                f"Initializing FaceLandmarker from model: {resolved_model_path} "
                f"(max_faces={self.config.max_num_faces}, min_det={self.config.face_min_detection_confidence})"
            )
            base_options = BaseOptions(model_asset_path=resolved_model_path)
            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.VIDEO,
                num_faces=self.config.max_num_faces,
                min_face_detection_confidence=self.config.face_min_detection_confidence,
                min_face_presence_confidence=self.config.face_min_presence_confidence,
                min_tracking_confidence=self.config.face_min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)
            self._is_initialized = True
            self.logger.info("FaceLandmarker initialized successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize FaceLandmarker: {e}", exc_info=True)
            raise FaceTrackerError(f"FaceLandmarker initialization failed: {e}") from e

    def process(
        self,
        frame: Optional[np.ndarray],
        timestamp_ms: Optional[int] = None,
        rgb_frame: Optional[np.ndarray] = None,
    ) -> FaceTrackingResult:
        """
        Process a single video frame for face detection and head pose estimation.

        Args:
            frame: BGR numpy image frame from camera.
            timestamp_ms: Monotonically increasing timestamp in milliseconds.
            rgb_frame: Optional pre-converted RGB frame to avoid redundant conversion.

        Returns:
            FaceTrackingResult containing face landmarks and 3D head pose data.
        """
        if frame is None or frame.size == 0:
            return FaceTrackingResult(face_detected=False)

        if not self._is_initialized or self._landmarker is None:
            self.logger.warning("FaceTracker.process called before start(). Auto-starting.")
            self.start()

        # Ensure monotonically increasing timestamp for MediaPipe VIDEO mode
        if timestamp_ms is None:
            current_ms = int(time.time() * 1000)
        else:
            current_ms = timestamp_ms

        if current_ms <= self._last_timestamp_ms:
            current_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current_ms

        # Re-use pre-converted RGB frame if provided (avoids duplicate cvtColor)
        if rgb_frame is not None:
            rgb_data = rgb_frame
        else:
            rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_data)

        # Run inference
        raw_result: FaceLandmarkerResult = self._landmarker.detect_for_video(mp_image, current_ms)

        if not raw_result.face_landmarks or len(raw_result.face_landmarks) == 0:
            self._pose_smoother.reset()
            return FaceTrackingResult(face_detected=False, timestamp_ms=current_ms)

        # Process the primary face
        h, w = frame.shape[:2]
        face_lms = raw_result.face_landmarks[0]

        norm_pts = [(lm.x, lm.y, lm.z) for lm in face_lms]
        pixel_pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms]

        xs = [p[0] for p in pixel_pts]
        ys = [p[1] for p in pixel_pts]
        bbox = (max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys)))
        center_px = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

        face_data = FaceLandmarkData(
            landmarks=norm_pts,
            landmarks_px=pixel_pts,
            bbox=bbox,
            center_px=center_px,
        )

        # Estimate Head Pose via solvePnP
        head_pose = self._estimate_head_pose(pixel_pts, (w, h))

        return FaceTrackingResult(
            face_detected=True,
            face_data=face_data,
            head_pose=head_pose,
            timestamp_ms=current_ms,
        )

    def _estimate_head_pose(
        self,
        pixel_pts: List[Tuple[int, int]],
        frame_size: Tuple[int, int],
    ) -> Optional[HeadPoseData]:
        """
        Estimate 3D head pose orientation (Yaw, Pitch, Roll in degrees) from facial landmarks.
        """
        w, h = frame_size
        if len(pixel_pts) < 468:
            return None

        # 2D image points corresponding to CANONICAL_FACE_3D:
        # Landmark 1: Nose tip
        # Landmark 152: Chin
        # Landmark 263: Screen Right eye corner
        # Landmark 33: Screen Left eye corner
        # Landmark 291: Screen Right mouth corner
        # Landmark 61: Screen Left mouth corner
        image_points = np.array([
            pixel_pts[1],     # Nose tip
            pixel_pts[152],   # Chin
            pixel_pts[263],   # Screen Right eye
            pixel_pts[33],    # Screen Left eye
            pixel_pts[291],   # Screen Right mouth
            pixel_pts[61],    # Screen Left mouth
        ], dtype=np.float64)

        # Camera intrinsic matrix approximation
        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            CANONICAL_FACE_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # Convert rotation vector to 3x3 rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles in degrees
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        def _normalize_angle(angle: float) -> float:
            """Normalize Euler angle into standard (-180.0, 180.0] interval."""
            return ((angle + 180.0) % 360.0) - 180.0

        # Pitch: rotation about lateral X-axis.
        # Negated so that physical Nod Down is positive (> 0), Nod Up is negative (< 0).
        raw_pitch = -_normalize_angle(float(angles[0]))
        # Yaw: rotation about vertical Y-axis (positive = Turn Right, negative = Turn Left).
        raw_yaw = _normalize_angle(float(angles[1]))
        # Roll: rotation about optical Z-axis (positive = Tilt Right, negative = Tilt Left).
        raw_roll = _normalize_angle(float(angles[2]))

        # Smooth angles using Exponential Moving Average
        smoothed = self._pose_smoother.filter(np.array([raw_yaw, raw_pitch, raw_roll]))
        smooth_yaw, smooth_pitch, smooth_roll = float(smoothed[0]), float(smoothed[1]), float(smoothed[2])

        return HeadPoseData(
            yaw=smooth_yaw,
            pitch=smooth_pitch,
            roll=smooth_roll,
            raw_yaw=raw_yaw,
            raw_pitch=raw_pitch,
            raw_roll=raw_roll,
            rvec=rvec,
            tvec=tvec,
        )

    def reset_smoother(self) -> None:
        """Reset the temporal pose smoothing filter."""
        self._pose_smoother.reset()

    def close(self) -> None:
        """Release MediaPipe native landmarker resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as e:
                self.logger.warning(f"Error closing FaceLandmarker: {e}")
            finally:
                self._landmarker = None
        self._is_initialized = False
        self._pose_smoother.reset()
        self.logger.info("FaceTracker closed cleanly.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
