"""
Camera Capture Abstraction
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Provides a robust, reusable abstraction over OpenCV VideoCapture with:
- macOS AVFoundation backend prioritization
- Automatic horizontal frame mirroring
- Graceful initialization and frame-read error handling
- Configurable index and resolution
- Context manager support
"""

import sys
from typing import Optional, Tuple
import cv2
import numpy as np

from config.settings import CameraConfig, DEFAULT_CONFIG
from utils.logger import setup_logger


class CameraError(Exception):
    """Base exception for camera operations."""
    pass


class CameraInitError(CameraError):
    """Raised when the camera fails to initialize or open."""
    pass


class Camera:
    """
    Reusable camera manager encapsulating device capture, hardware backend selection,
    frame acquisition, and stream lifecycle.
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        """
        Initialize the Camera instance.

        Args:
            config: CameraConfig specifying device index, resolution, fps, and mirroring.
        """
        self.config: CameraConfig = config if config is not None else DEFAULT_CONFIG.camera
        self.logger = setup_logger("Camera")
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_fps: float = 0.0

    def start(self) -> bool:
        """
        Open the camera device using the optimal backend (cv2.CAP_AVFOUNDATION on macOS).
        Configures frame resolution and verified readiness.

        Returns:
            True if camera opened successfully.

        Raises:
            CameraInitError: If camera cannot be opened or accessed.
        """
        if self._is_opened and self._cap is not None and self._cap.isOpened():
            self.logger.warning("Camera is already opened.")
            return True

        device_index = self.config.device_index

        # Select backend: enforce AVFoundation on macOS
        if sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION
            backend_name = "AVFoundation"
        else:
            backend = cv2.CAP_ANY
            backend_name = "Default/Auto"

        self.logger.info(
            f"Opening camera index {device_index} with backend {backend_name} (sys.platform={sys.platform})..."
        )

        try:
            self._cap = cv2.VideoCapture(device_index, backend)
        except Exception as e:
            err_msg = f"Exception while invoking cv2.VideoCapture({device_index}, {backend_name}): {e}"
            self.logger.error(err_msg)
            raise CameraInitError(err_msg) from e

        if not self._cap.isOpened():
            err_msg = (
                f"Failed to open camera device at index {device_index} using {backend_name}.\n"
                "Troubleshooting steps:\n"
                "  1. Verify macOS Camera permission in System Settings -> Privacy & Security -> Camera.\n"
                "  2. Check if another application (FaceTime, Zoom, Teams, Photo Booth) is using the camera.\n"
                "  3. Verify the device index (built-in camera is typically index 0)."
            )
            self.logger.error(err_msg)
            self._cap.release()
            self._cap = None
            self._is_opened = False
            raise CameraInitError(err_msg)

        # Set requested resolution & frame rate
        self._cap.set(cv2.CAP_PROP_FPS, float(self.config.fps))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.frame_width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.frame_height))
        # Ensure single-frame buffer to prevent frame queue accumulation and latency drift
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Query actual negotiated properties
        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._is_opened = True

        self.logger.info(
            f"Camera opened successfully: {self._actual_width}x{self._actual_height} "
            f"@ {self._actual_fps:.1f} FPS (Requested: {self.config.frame_width}x{self.config.frame_height} "
            f"@ {self.config.fps} FPS, Mirror: {self.config.mirror})"
        )
        return True

    def read(self, raw: bool = False) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the camera.

        Args:
            raw: If True, do not apply horizontal flip even if config.mirror is True.

        Returns:
            Tuple of (success: bool, frame: Optional[np.ndarray]).
            If frame capture fails, returns (False, None).
        """
        if not self._is_opened or self._cap is None:
            self.logger.error("Attempted to read from an uninitialized or closed camera.")
            return False, None

        ret, frame = self._cap.read()

        if not ret or frame is None:
            self.logger.warning("Camera failed to deliver frame (ret=False or empty frame).")
            return False, None

        # Mirror frame horizontally if configured
        if self.config.mirror and not raw:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self) -> None:
        """
        Safely release the camera hardware resource.
        """
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                self.logger.warning(f"Error during VideoCapture release: {e}")
            finally:
                self._cap = None

        self._is_opened = False
        self.logger.info("Camera capture released cleanly.")

    @property
    def is_opened(self) -> bool:
        """Check if camera is currently opened and available."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return actual (width, height) of captured frames."""
        return self._actual_width, self._actual_height

    @property
    def width(self) -> int:
        """Return actual frame width."""
        return self._actual_width

    @property
    def height(self) -> int:
        """Return actual frame height."""
        return self._actual_height

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        if self._is_opened:
            self.release()
