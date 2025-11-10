"""
Camera management module.

This module handles camera initialization, configuration, and provides
a clean interface for camera operations with automatic resource management.
"""

import cv2

import config
from camera_selector import show_camera_selector


class CameraManager:
    """
    Manager class for camera operations.

    Handles camera initialization, configuration, and provides context
    manager support for automatic resource cleanup.
    """

    def __init__(self, camera_id=None, auto_select=True):
        """
        Initialize the camera manager.

        Args:
            camera_id: Optional camera ID. If None and auto_select is True,
                      will show selection dialog.
            auto_select: If True, automatically select camera when camera_id
                        is None.

        Raises:
            RuntimeError: If camera cannot be opened or no camera selected.
        """
        self.camera_id = camera_id
        self.cap = None
        self.width = None
        self.height = None

        # Determine camera ID
        if self.camera_id is None and auto_select:
            self.camera_id = show_camera_selector(use_gui=True)
            if self.camera_id is None:
                raise RuntimeError("No camera selected")

        # Use default camera if still None
        if self.camera_id is None:
            self.camera_id = config.CAMERA_ID

        # Initialize camera
        self._initialize_camera()

        print(f"[CameraManager] Initialized with camera ID: {self.camera_id}")

    def _initialize_camera(self):
        """
        Initialize and configure the camera.

        Raises:
            RuntimeError: If camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        # Configure camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        # Get actual resolution (may differ from requested)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[CameraManager] Camera configured: {self.width}x{self.height}")

    def get_capture(self):
        """
        Get the OpenCV VideoCapture object.

        Returns:
            cv2.VideoCapture: Camera capture object.
        """
        return self.cap

    def get_resolution(self):
        """
        Get the camera resolution.

        Returns:
            tuple: (width, height) in pixels.
        """
        return (self.width, self.height)

    def is_opened(self):
        """
        Check if camera is opened and ready.

        Returns:
            bool: True if camera is opened, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            print("[CameraManager] Camera released")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.release()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.release()