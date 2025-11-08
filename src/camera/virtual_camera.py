#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/camera/virtual_camera.py

import platform
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pyvirtualcam

from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class VirtualCamera:
    """
    Class for creating a virtual camera accessible to other applications.
    Allows passing generated images as a video source.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        device_name: str = None,  # Default None for auto-detection
        logger: Optional[CustomLogger] = None,
        max_retries: int = 3,  # Maximum number of attempts
        retry_delay: float = 1.0,  # Delay between attempts
    ):
        """
        Initialize virtual camera.

        Args:
            width (int): Width of the virtual camera image
            height (int): Height of the virtual camera image
            fps (int): Frames per second
            device_name (str, optional): Display name for device (default: auto-detected)
            logger (CustomLogger, optional): Logger for recording messages
            max_retries (int): Maximum number of initialization attempts
            retry_delay (float): Delay between attempts in seconds
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.device_name = device_name
        self.logger = logger or CustomLogger()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.cam = None
        self.is_initialized = False
        self.initialization_failed = False  # Flag indicating definitive failure
        self.frame_count = 0
        self.last_frame_time = 0
        self.performance = PerformanceMonitor("VirtualCamera")
        self.retry_count = 0  # Initialization attempt counter

        # Virtual camera information
        self.camera_info = {
            "name": device_name if device_name else "Auto-detected",
            "resolution": (width, height),
            "fps": fps,
            "backend": "unknown",
            "system": platform.system(),
            "available_devices": [],
        }

        self.fps_sleep_time = 1.0 / fps if fps > 0 else 0

        # Automatically detect available devices
        self._detect_available_devices()

    def _detect_available_devices(self) -> None:
        """
        Detects available virtual camera devices.
        """
        try:
            # Try to get available backends
            available_backends = []
            try:
                if hasattr(pyvirtualcam, "get_available_backends"):
                    available_backends = pyvirtualcam.get_available_backends()
                    self.logger.debug(
                        "VirtualCamera",
                        f"Available backends: {available_backends}",
                        log_type="VIRTUAL_CAM",
                    )
            except Exception as e:
                self.logger.warning(
                    "VirtualCamera",
                    f"Error getting available backends: {str(e)}",
                    log_type="VIRTUAL_CAM",
                )

            # Try automatic detection of device names
            system = platform.system()
            self.camera_info["available_devices"] = self._get_system_specific_device_names(system)

        except Exception as e:
            self.logger.warning(
                "VirtualCamera",
                f"Error detecting available devices: {str(e)}",
                log_type="VIRTUAL_CAM",
            )

    def _get_system_specific_device_names(self, system: str) -> List[str]:
        """
        Returns list of possible virtual camera device names for given system.

        Args:
            system (str): Operating system name

        Returns:
            List[str]: List of possible device names
        """
        if system == "Windows":
            return [
                "OBS Virtual Camera",
                "OBS Camera",
                "OBS-Camera",
                "Unity Video Capture",
                "XSplit VCam",
                "e2eSoft VCam",
                "Stick Figure Webcam",
            ]
        elif system == "Darwin":  # macOS
            return ["OBS Virtual Camera", "NDI Video", "CamTwist", "Stick Figure Webcam"]
        elif system == "Linux":
            return [
                "/dev/video0",
                "/dev/video1",
                "/dev/video2",
                "/dev/video20",
                "Stick Figure Webcam",
            ]
        else:
            return ["Stick Figure Webcam"]

    def initialize(self) -> bool:
        """
        Initializes virtual camera.

        Returns:
            bool: True if camera initialization succeeded, False otherwise
        """
        if self.is_initialized:
            return True

        # If initialization already definitively failed and attempts exhausted,
        # don't try again
        if self.initialization_failed:
            return False

        try:
            self.retry_count += 1
            self.logger.debug(
                "VirtualCamera",
                f"Initializing virtual camera (attempt {self.retry_count}/{self.max_retries}): "
                f"{self.width}x{self.height} @ {self.fps} FPS",
                log_type="VIRTUAL_CAM",
            )

            # Determine pixel format - OpenCV uses BGR
            fmt = pyvirtualcam.PixelFormat.BGR

            # Different operating systems may require different settings
            system = platform.system()
            backend_params = {}

            # Choose device name:
            # 1. Priority goes to user-provided name
            # 2. Use default name for given system
            if self.device_name:
                if system == "Windows":
                    backend_params["device"] = self.device_name
            else:
                # For Windows try OBS Virtual Camera first
                if system == "Windows":
                    backend_params["device"] = "OBS Virtual Camera"

            # Try creating virtual camera with automatic backend detection
            try:
                self.cam = pyvirtualcam.Camera(
                    width=self.width, height=self.height, fps=self.fps, fmt=fmt, **backend_params
                )

                # Update backend and actual device name information
                self.camera_info["backend"] = self.cam.backend
                if hasattr(self.cam, "device"):
                    self.camera_info["name"] = self.cam.device

                self.is_initialized = True
                self.retry_count = 0  # Reset attempt counter

                self.logger.info(
                    "VirtualCamera",
                    f"Virtual camera started with backend: {self.cam.backend}, "
                    f"device: {getattr(self.cam, 'device', 'default')}",
                    log_type="VIRTUAL_CAM",
                )

                self.last_frame_time = time.time()

                # Notify logger of status
                self.logger.virtual_camera_status(True, self.camera_info)

                return True

            except Exception as first_e:
                # If first attempt fails, try with alternative device names
                if system == "Windows" and self.retry_count <= 1:
                    device_names = self._get_system_specific_device_names(system)

                    for device_name in device_names:
                        try:
                            self.logger.debug(
                                "VirtualCamera",
                                f"Trying initialization with alternative device name: {device_name}",
                                log_type="VIRTUAL_CAM",
                            )

                            self.cam = pyvirtualcam.Camera(
                                width=self.width,
                                height=self.height,
                                fps=self.fps,
                                fmt=fmt,
                                device=device_name,
                            )

                            # Success - save information
                            self.camera_info["backend"] = self.cam.backend
                            self.camera_info["name"] = device_name
                            self.is_initialized = True
                            self.retry_count = 0

                            self.logger.info(
                                "VirtualCamera",
                                f"Virtual camera started with alternative name: {device_name}, "
                                f"backend: {self.cam.backend}",
                                log_type="VIRTUAL_CAM",
                            )

                            self.last_frame_time = time.time()
                            self.logger.virtual_camera_status(True, self.camera_info)
                            return True

                        except Exception as e:
                            # Continue trying with other names
                            self.logger.debug(
                                "VirtualCamera",
                                f"Attempt with device {device_name} failed: {str(e)}",
                                log_type="VIRTUAL_CAM",
                            )

                    # No alternative name worked, report original error
                    raise first_e
                else:
                    # For other systems or after exhausting alternatives, report original error
                    raise first_e

        except Exception as e:
            error_info = {"error": str(e)}
            self.logger.error(
                "VirtualCamera",
                f"Error during virtual camera initialization: {str(e)}",
                log_type="VIRTUAL_CAM",
                error=error_info,
            )
            self.logger.virtual_camera_status(False, error_info)

            # Additional troubleshooting advice
            self._provide_troubleshooting_info()

            # Check if we've exhausted all attempts
            if self.retry_count >= self.max_retries:
                self.initialization_failed = True
                self.logger.warning(
                    "VirtualCamera",
                    f"Exhausted limit of {self.max_retries} virtual camera initialization attempts. "
                    "Giving up on further attempts.",
                    log_type="VIRTUAL_CAM",
                )
            else:
                # Wait before next attempt
                time.sleep(self.retry_delay)

            return False

    def _provide_troubleshooting_info(self) -> None:
        """
        Provides information helpful in troubleshooting virtual camera issues.
        """
        system = platform.system()

        if system == "Windows":
            self.logger.info(
                "VirtualCamera",
                "Troubleshooting (Windows): Make sure OBS Studio "
                "is installed and virtual camera is running. "
                "1. Open OBS Studio. "
                "2. In menu select Tools -> Virtual Camera. "
                "3. Click 'Start Virtual Camera'.",
                log_type="VIRTUAL_CAM",
            )
        elif system == "Darwin":  # macOS
            self.logger.info(
                "VirtualCamera",
                "Troubleshooting (macOS): Install OBS Studio and plugin "
                "obs-mac-virtualcam. Alternatively you can use CamTwist. "
                "Detailed instructions: "
                "1. Install OBS from https://obsproject.com "
                "2. Install obs-mac-virtualcam plugin "
                "3. Run OBS and enable Virtual Camera",
                log_type="VIRTUAL_CAM",
            )
        elif system == "Linux":
            self.logger.info(
                "VirtualCamera",
                "Troubleshooting (Linux): Make sure "
                "v4l2loopback is installed and loaded:\n"
                "sudo apt-get install v4l2loopback-dkms\n"
                "sudo modprobe v4l2loopback\n"
                "After loading module, check available devices: ls -l /dev/video*",
                log_type="VIRTUAL_CAM",
            )

    def send_frame(self, frame: np.ndarray) -> bool:
        """
        Sends frame to virtual camera.

        Args:
            frame (np.ndarray): Frame (image) to send

        Returns:
            bool: True if frame was sent successfully, False otherwise
        """
        # If initialization definitively failed, don't try again
        if self.initialization_failed:
            return False

        # Attempt initialization, but only if there wasn't definitive failure
        if not self.is_initialized and not self.initialize():
            # Return False but don't report error - we already did in initialize()
            return False

        if self.cam is None:
            self.logger.warning(
                "VirtualCamera",
                "Attempting to send frame to uninitialized camera",
                log_type="VIRTUAL_CAM",
            )
            return False

        self.performance.start_timer()

        try:
            # Check if frame size matches camera settings
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))

            # Send frame
            self.cam.send(frame)

            # Calculate time to next frame to maintain constant FPS
            current_time = time.time()

            # Protection against division by zero - use epsilon
            elapsed = max(0.000001, current_time - self.last_frame_time)  # Minimum value 1µs

            # If needed, wait to maintain constant FPS
            if self.fps_sleep_time > 0 and elapsed < self.fps_sleep_time:
                time.sleep(self.fps_sleep_time - elapsed)

            self.last_frame_time = time.time()
            self.frame_count += 1

            # Log information every 100 frames
            if self.frame_count % 100 == 0:
                # Calculate actual FPS (with protection against division by zero)
                # Use self.fps value if elapsed is too small
                real_fps = 1.0 / elapsed if elapsed > 0.001 else self.fps

                self.logger.debug(
                    "VirtualCamera",
                    f"Sent {self.frame_count} frames, current FPS: {real_fps:.1f}",
                    log_type="VIRTUAL_CAM",
                )

            self.performance.stop_timer()
            processing_time = self.performance.get_last_execution_time() * 1000  # ms

            # Log performance information every 300 frames
            if self.frame_count % 300 == 0:
                # Protection against division by zero
                real_fps = 1.0 / max(0.001, elapsed)
                self.logger.performance_metrics(real_fps, processing_time, "VirtualCamera")

            return True

        except Exception as e:
            self.performance.stop_timer()
            self.logger.error(
                "VirtualCamera",
                f"Error sending frame: {str(e)}",
                log_type="VIRTUAL_CAM",
                error={"error": str(e)},
            )

            # Reset camera state to attempt re-initialization
            self.is_initialized = False
            if self.cam is not None:
                try:
                    self.cam.close()
                except:
                    pass
                self.cam = None

            return False

    def send_black_frame(self) -> bool:
        """
        Sends black frame to virtual camera.

        Returns:
            bool: True if frame was sent successfully, False otherwise
        """
        if self.initialization_failed:
            return False

        black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.send_frame(black_frame)

    def send_white_frame(self) -> bool:
        """
        Sends white frame to virtual camera.

        Returns:
            bool: True if frame was sent successfully, False otherwise
        """
        if self.initialization_failed:
            return False

        white_frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        return self.send_frame(white_frame)

    def send_test_pattern(self) -> bool:
        """
        Sends test pattern to virtual camera.

        Returns:
            bool: True if frame was sent successfully, False otherwise
        """
        if self.initialization_failed:
            return False

        # Create test pattern - colored bars, grid, text
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Horizontal colored bars
        bar_height = self.height // 7
        colors = [
            (255, 0, 0),  # Blue (BGR)
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 255, 255),  # White
        ]

        for i, color in enumerate(colors):
            start_y = i * bar_height
            end_y = (i + 1) * bar_height if i < len(colors) - 1 else self.height
            frame[start_y:end_y, :] = color

        # Add grid
        grid_step = 50
        grid_color = (128, 128, 128)
        grid_thickness = 1

        for x in range(0, self.width, grid_step):
            cv2.line(frame, (x, 0), (x, self.height), grid_color, grid_thickness)
        for y in range(0, self.height, grid_step):
            cv2.line(frame, (0, y), (self.width, y), grid_color, grid_thickness)

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Stick Figure Webcam {self.width}x{self.height} @ {self.fps}FPS"
        text_size, _ = cv2.getTextSize(text, font, 1, 2)
        text_x = (self.width - text_size[0]) // 2
        text_y = (self.height + text_size[1]) // 2

        # Initialization status - red if not working, green if working
        status_color = (0, 255, 0) if self.is_initialized else (0, 0, 255)  # Green/Red
        status_text = (
            "VIRTUAL CAMERA READY" if self.is_initialized else "VIRTUAL CAMERA NOT WORKING"
        )
        status_text_size, _ = cv2.getTextSize(status_text, font, 0.8, 2)
        status_x = (self.width - status_text_size[0]) // 2
        status_y = (self.height + status_text_size[1]) // 2 + 30

        # Text border for better visibility
        cv2.putText(frame, text, (text_x - 1, text_y - 1), font, 1, (0, 0, 0), 2)
        cv2.putText(frame, text, (text_x + 1, text_y + 1), font, 1, (0, 0, 0), 2)
        cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

        # Status
        cv2.putText(frame, status_text, (status_x, status_y), font, 0.8, status_color, 2)

        return self.send_frame(frame)

    def get_camera_info(self) -> Dict[str, Any]:
        """
        Returns information about virtual camera.

        Returns:
            Dict[str, Any]: Dictionary with camera information
        """
        return self.camera_info.copy()

    def set_resolution(self, width: int, height: int) -> bool:
        """
        Changes virtual camera resolution.
        Requires camera re-initialization!

        Args:
            width (int): New width
            height (int): New height

        Returns:
            bool: True if resolution change succeeded
        """
        if self.is_initialized:
            self.logger.warning(
                "VirtualCamera",
                "Resolution change requires camera re-initialization. "
                "Closing current instance...",
                log_type="VIRTUAL_CAM",
            )
            self.close()

        self.width = width
        self.height = height
        self.camera_info["resolution"] = (width, height)

        # Reset initialization failure flag
        self.initialization_failed = False
        self.retry_count = 0

        self.logger.info(
            "VirtualCamera",
            f"Set new resolution: {width}x{height}. "
            "Camera will be initialized on next use.",
            log_type="VIRTUAL_CAM",
        )

        return True

    def set_fps(self, fps: int) -> bool:
        """
        Changes target FPS of virtual camera.
        Requires camera re-initialization!

        Args:
            fps (int): New FPS

        Returns:
            bool: True if FPS change succeeded
        """
        if self.is_initialized:
            self.logger.warning(
                "VirtualCamera",
                "FPS change requires camera re-initialization. "
                "Closing current instance...",
                log_type="VIRTUAL_CAM",
            )
            self.close()

        self.fps = fps
        self.camera_info["fps"] = fps
        self.fps_sleep_time = 1.0 / fps if fps > 0 else 0

        # Reset initialization failure flag
        self.initialization_failed = False
        self.retry_count = 0

        self.logger.info(
            "VirtualCamera",
            f"Set new FPS: {fps}. "
            "Camera will be initialized on next use.",
            log_type="VIRTUAL_CAM",
        )

        return True

    def reset(self) -> bool:
        """
        Resets virtual camera, closing current instance and resetting state flags.

        Returns:
            bool: True if reset succeeded
        """
        try:
            self.close()
            self.is_initialized = False
            self.initialization_failed = False
            self.retry_count = 0

            self.logger.info(
                "VirtualCamera", "Virtual camera has been reset", log_type="VIRTUAL_CAM"
            )

            return True
        except Exception as e:
            self.logger.error(
                "VirtualCamera",
                f"Error resetting virtual camera: {str(e)}",
                log_type="VIRTUAL_CAM",
            )
            return False

    def close(self) -> None:
        """
        Closes virtual camera.
        """
        if self.is_initialized and self.cam is not None:
            try:
                # Before closing, send black frame
                self.send_black_frame()

                # Some backends may need brief pause
                time.sleep(0.1)

                # Close camera
                self.cam.close()
                self.cam = None
                self.is_initialized = False

                self.logger.info(
                    "VirtualCamera", "Virtual camera closed", log_type="VIRTUAL_CAM"
                )

            except Exception as e:
                self.logger.error(
                    "VirtualCamera",
                    f"Error closing virtual camera: {str(e)}",
                    log_type="VIRTUAL_CAM",
                    error={"error": str(e)},
                )

    def __del__(self):
        """
        Class destructor ensuring camera is closed.
        """
        self.close()
