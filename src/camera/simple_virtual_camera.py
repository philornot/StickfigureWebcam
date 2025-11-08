#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simplified virtual camera wrapper with cleaner initialization."""

import platform
import time
from typing import Optional, Tuple
import numpy as np

from src.utils.custom_logger import CustomLogger


class SimpleVirtualCamera:
    """Simplified virtual camera wrapper.

    This version removes complex retry logic and multiple fallback attempts.
    Either initialization succeeds or it fails clearly with helpful error message.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        logger: Optional[CustomLogger] = None
    ):
        """Initialize virtual camera.

        Args:
            width: Video width
            height: Video height
            fps: Frames per second
            logger: Optional logger

        Raises:
            RuntimeError: If initialization fails
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.logger = logger or CustomLogger()

        self.camera = None
        self.backend = None
        self.device_name = None
        self.system = platform.system()

        self._initialize()

    def _initialize(self):
        """Initialize the virtual camera.

        Raises:
            RuntimeError: If initialization fails with helpful message
        """
        try:
            import pyvirtualcam
        except ImportError:
            raise RuntimeError(
                "pyvirtualcam is not installed. "
                "Install it with: pip install pyvirtualcam"
            )

        # Get device name based on system
        device_name = self._get_device_name()

        try:
            # Single initialization attempt
            if device_name:
                self.camera = pyvirtualcam.Camera(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    fmt=pyvirtualcam.PixelFormat.BGR,
                    device=device_name
                )
            else:
                self.camera = pyvirtualcam.Camera(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    fmt=pyvirtualcam.PixelFormat.BGR
                )

            self.backend = self.camera.backend
            self.device_name = getattr(self.camera, 'device', 'default')

            self.logger.info(
                "VirtualCamera",
                f"Initialized: {self.backend} ({self.device_name})"
            )

        except Exception as e:
            error_msg = self._create_helpful_error_message(e)
            raise RuntimeError(error_msg)

    def _get_device_name(self) -> Optional[str]:
        """Get appropriate device name for the system.

        Returns:
            Device name or None for auto-detection
        """
        if self.system == "Windows":
            return "OBS Virtual Camera"
        elif self.system == "Linux":
            return "/dev/video20"  # Common v4l2loopback device
        else:  # macOS
            return None  # Let pyvirtualcam auto-detect

    def _create_helpful_error_message(self, error: Exception) -> str:
        """Create a helpful error message based on system and error.

        Args:
            error: The caught exception

        Returns:
            User-friendly error message with instructions
        """
        base_msg = f"Failed to initialize virtual camera: {str(error)}\n\n"

        if self.system == "Windows":
            return (
                base_msg +
                "Make sure OBS Studio is installed and Virtual Camera is started:\n"
                "1. Install OBS Studio from https://obsproject.com\n"
                "2. Open OBS Studio\n"
                "3. Go to Tools → Start Virtual Camera\n"
                "4. Try running this application again"
            )

        elif self.system == "Linux":
            return (
                base_msg +
                "Make sure v4l2loopback kernel module is loaded:\n"
                "1. Install: sudo apt-get install v4l2loopback-dkms\n"
                "2. Load module: sudo modprobe v4l2loopback\n"
                "3. Verify: ls /dev/video*\n"
                "4. Try running this application again"
            )

        else:  # macOS
            return (
                base_msg +
                "Make sure OBS Studio with virtual camera plugin is installed:\n"
                "1. Install OBS Studio from https://obsproject.com\n"
                "2. Install obs-mac-virtualcam plugin\n"
                "3. Open OBS and start Virtual Camera\n"
                "4. Try running this application again"
            )

    def send(self, frame: np.ndarray):
        """Send frame to virtual camera.

        Args:
            frame: BGR frame to send

        Raises:
            RuntimeError: If camera not initialized or send fails
        """
        if self.camera is None:
            raise RuntimeError("Virtual camera not initialized")

        # Resize if needed
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            import cv2
            frame = cv2.resize(frame, (self.width, self.height))

        try:
            self.camera.send(frame)
        except Exception as e:
            raise RuntimeError(f"Failed to send frame: {str(e)}")

    def close(self):
        """Close the virtual camera."""
        if self.camera:
            try:
                self.camera.close()
                self.logger.info("VirtualCamera", "Closed successfully")
            except Exception as e:
                self.logger.warning(
                    "VirtualCamera",
                    f"Error during close: {str(e)}"
                )
            finally:
                self.camera = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor."""
        if self.camera is not None:
            try:
                self.close()
            except:
                pass


def try_create_virtual_camera(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    logger: Optional[CustomLogger] = None
) -> Optional[SimpleVirtualCamera]:
    """Try to create virtual camera, return None on failure.

    This is a convenience function that catches initialization errors
    and returns None instead of raising an exception.

    Args:
        width: Video width
        height: Video height
        fps: Frames per second
        logger: Optional logger

    Returns:
        SimpleVirtualCamera instance or None if initialization failed
    """
    try:
        return SimpleVirtualCamera(width, height, fps, logger)
    except RuntimeError as e:
        if logger:
            logger.warning("VirtualCamera", str(e))
        return None
