#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera capture module for webcam access and frame processing.

This module provides the CameraCapture class for interfacing with webcams,
configuring camera settings, capturing frames, and basic image operations.
"""

import time
from typing import Tuple, Dict, Optional, Any, List

import cv2
import numpy as np

from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class CameraCapture:
    """Handles webcam capture with configuration and frame operations.

    Provides interface for camera configuration, frame capture, and basic
    image operations like flipping and brightness adjustment.

    Attributes:
        camera_id: Camera device identifier
        width: Preferred frame width in pixels
        height: Preferred frame height in pixels
        fps: Target frames per second
        logger: Logger instance for diagnostics
        cap: OpenCV VideoCapture object
        is_open: Whether camera is currently open
        frame_count: Total frames captured
        last_frame: Most recently captured frame
        camera_info: Dictionary of camera parameters

    Example:
        >>> camera = CameraCapture(camera_id=0, width=640, height=480)
        >>> success, frame = camera.read()
        >>> if success:
        ...     processed = camera.flip_horizontal(frame)
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        logger: Optional[CustomLogger] = None
    ):
        """Initialize camera capture module.

        Args:
            camera_id: Camera device ID (usually 0 for default camera)
            width: Preferred frame width in pixels
            height: Preferred frame height in pixels
            fps: Target frames per second
            logger: Optional logger for diagnostics
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.logger = logger or CustomLogger()

        self.cap = None
        self.is_open = False
        self.frame_count = 0
        self.last_frame = None
        self.last_frame_time = 0
        self.performance = PerformanceMonitor("CameraCapture")

        # Camera information dictionary
        self.camera_info = {
            "id": camera_id,
            "name": "Unknown",
            "resolution": (width, height),
            "fps": fps,
            "real_fps": 0.0
        }

        # Automatically open camera on initialization
        self.open()

    def open(self) -> bool:
        """Open camera connection and configure parameters.

        Returns:
            True if camera opened successfully, False otherwise
        """
        try:
            self.logger.debug("CameraCapture", f"Attempting to open camera ID: {self.camera_id}", log_type="CAMERA")
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                self.logger.error("CameraCapture", f"Failed to open camera ID: {self.camera_id}",
                                  log_type="CAMERA")
                self.is_open = False
                return False

            # Configure camera parameters
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Read actual parameters
            real_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            real_fps = self.cap.get(cv2.CAP_PROP_FPS)

            # Update camera information
            self.camera_info.update({
                "resolution": (real_width, real_height),
                "fps": real_fps,
                "backend": self.cap.getBackendName()
            })

            self.width, self.height = real_width, real_height
            self.is_open = True

            self.logger.info(
                "CameraCapture",
                f"Camera opened: {self.width}x{self.height} @ {real_fps:.1f} FPS",
                log_type="CAMERA",
                camera_info=self.camera_info
            )

            # Notify logger of camera status
            self.logger.camera_status(True, self.camera_info)

            # Initialize first frame
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                self.last_frame_time = time.time()
                self.frame_count = 1

            return True

        except Exception as e:
            self.is_open = False
            error_info = {"error": str(e)}
            self.logger.critical(
                "CameraCapture",
                f"Error opening camera: {str(e)}",
                log_type="CAMERA",
                error=error_info
            )
            self.logger.camera_status(False, error_info)
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera.

        Returns:
            Tuple containing:
                - bool: True if frame read successfully
                - Optional[np.ndarray]: Frame as NumPy array or None on error
        """
        if not self.is_open or self.cap is None:
            self.logger.warning("CameraCapture", "Attempting to read from closed camera", log_type="CAMERA")
            return False, None

        self.performance.start_timer()

        try:
            ret, frame = self.cap.read()

            if ret:
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_frame_time

                if elapsed > 0:
                    # Update real FPS with exponential smoothing
                    alpha = 0.3  # Smoothing coefficient
                    current_fps = 1.0 / elapsed
                    if self.camera_info["real_fps"] == 0:
                        self.camera_info["real_fps"] = current_fps
                    else:
                        self.camera_info["real_fps"] = (1 - alpha) * self.camera_info["real_fps"] + alpha * current_fps

                self.last_frame = frame
                self.last_frame_time = current_time

                # Log statistics every 100 frames
                if self.frame_count % 100 == 0:
                    self.logger.debug(
                        "CameraCapture",
                        f"Read {self.frame_count} frames, current FPS: {self.camera_info['real_fps']:.1f}",
                        log_type="CAMERA"
                    )

                self.performance.stop_timer()
                processing_time = self.performance.get_last_execution_time() * 1000  # ms

                # Log performance every 500 frames
                if self.frame_count % 500 == 0:
                    self.logger.performance_metrics(
                        self.camera_info["real_fps"],
                        processing_time,
                        "CameraCapture"
                    )

                return True, frame
            else:
                self.logger.warning(
                    "CameraCapture",
                    "Failed to read frame from camera",
                    log_type="CAMERA"
                )
                return False, None

        except Exception as e:
            self.logger.error(
                "CameraCapture",
                f"Error reading from camera: {str(e)}",
                log_type="CAMERA",
                error={"error": str(e)}
            )
            return False, None

    def get_latest_frame(self) -> np.ndarray:
        """Return most recently captured frame.

        Returns:
            Last successfully captured frame, or black frame if none available
        """
        if self.last_frame is not None:
            return self.last_frame.copy()
        else:
            # Return empty (black) frame
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information.

        Returns:
            Dictionary containing camera parameters
        """
        return self.camera_info.copy()

    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution.

        Args:
            width: New width in pixels
            height: New height in pixels

        Returns:
            True if resolution changed successfully
        """
        if not self.is_open or self.cap is None:
            self.logger.warning(
                "CameraCapture",
                "Attempting to change resolution on closed camera",
                log_type="CAMERA"
            )
            return False

        try:
            self.logger.debug(
                "CameraCapture",
                f"Changing resolution to {width}x{height}",
                log_type="CAMERA"
            )

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Check if resolution actually changed
            real_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.width, self.height = real_width, real_height
            self.camera_info["resolution"] = (real_width, real_height)

            self.logger.info(
                "CameraCapture",
                f"New resolution: {real_width}x{real_height}",
                log_type="CAMERA",
                camera_info=self.camera_info
            )

            # Warn if dimensions differ from requested
            if real_width != width or real_height != height:
                self.logger.warning(
                    "CameraCapture",
                    f"Requested resolution {width}x{height} not supported. "
                    f"Using closest available: {real_width}x{real_height}",
                    log_type="CAMERA"
                )

            return True

        except Exception as e:
            self.logger.error(
                "CameraCapture",
                f"Error changing resolution: {str(e)}",
                log_type="CAMERA",
                error={"error": str(e)}
            )
            return False

    def set_fps(self, fps: int) -> bool:
        """Set target frames per second.

        Args:
            fps: New FPS value

        Returns:
            True if FPS changed successfully
        """
        if not self.is_open or self.cap is None:
            self.logger.warning(
                "CameraCapture",
                "Attempting to change FPS on closed camera",
                log_type="CAMERA"
            )
            return False

        try:
            self.logger.debug("CameraCapture", f"Changing FPS to {fps}", log_type="CAMERA")

            self.cap.set(cv2.CAP_PROP_FPS, fps)

            # Check actual FPS
            real_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = real_fps
            self.camera_info["fps"] = real_fps

            self.logger.info(
                "CameraCapture",
                f"New FPS: {real_fps:.1f}",
                log_type="CAMERA"
            )

            if abs(real_fps - fps) > 0.1:
                self.logger.warning(
                    "CameraCapture",
                    f"Requested FPS {fps} not supported. "
                    f"Using closest available: {real_fps:.1f}",
                    log_type="CAMERA"
                )

            return True

        except Exception as e:
            self.logger.error(
                "CameraCapture",
                f"Error changing FPS: {str(e)}",
                log_type="CAMERA",
                error={"error": str(e)}
            )
            return False

    def list_available_cameras(self) -> List[Dict[str, Any]]:
        """Detect available cameras on system.

        Returns:
            List of dictionaries with information about available cameras
        """
        available_cameras = []
        max_cameras = 10  # Limit to 10 cameras

        self.logger.debug("CameraCapture", "Searching for available cameras...", log_type="CAMERA")

        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_info = {
                        "id": i,
                        "resolution": (
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        ),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                        "backend": cap.getBackendName()
                    }

                    # Capture one frame for verification
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(camera_info)
                        self.logger.debug(
                            "CameraCapture",
                            f"Found camera ID: {i} - {camera_info['resolution']} @ {camera_info['fps']} FPS",
                            log_type="CAMERA"
                        )

                cap.release()

            except Exception as e:
                self.logger.trace(
                    "CameraCapture",
                    f"Error checking camera ID: {i}: {str(e)}",
                    log_type="CAMERA"
                )

        self.logger.info(
            "CameraCapture",
            f"Found {len(available_cameras)} available cameras",
            log_type="CAMERA",
            cameras=available_cameras
        )

        return available_cameras

    def flip_horizontal(self, frame: np.ndarray) -> np.ndarray:
        """Flip frame horizontally.

        Args:
            frame: Input frame

        Returns:
            Horizontally flipped frame
        """
        return cv2.flip(frame, 1)

    def adjust_brightness_contrast(
        self,
        frame: np.ndarray,
        brightness: float = 0,
        contrast: float = 1.0
    ) -> np.ndarray:
        """Adjust frame brightness and contrast.

        Args:
            frame: Input frame
            brightness: Brightness adjustment (-1.0 to 1.0)
            contrast: Contrast multiplier (0.0 to 3.0)

        Returns:
            Adjusted frame
        """
        # Convert brightness from range -1.0:1.0 to pixel offset value
        brightness_value = int(brightness * 255)

        # Apply contrast and brightness
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness_value)

        return adjusted

    def close(self) -> None:
        """Close camera connection."""
        if self.is_open and self.cap is not None:
            self.cap.release()
            self.is_open = False
            self.logger.info("CameraCapture", "Camera closed", log_type="CAMERA")

    def __del__(self):
        """Destructor ensuring camera is closed."""
        self.close()
