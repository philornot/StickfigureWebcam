"""
Camera management for main window.

This module handles camera thread management and camera-related operations.
"""

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from gui.camera_thread import CameraThread


class CameraManager(QObject):
    """
    Camera manager for main window.

    Handles camera thread lifecycle and camera operations.
    """

    # Signals
    frame_ready = pyqtSignal(np.ndarray, object, object, bool, bool)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    camera_state_changed = pyqtSignal(bool)

    def __init__(self, main_window, live_config, ui_manager):
        """
        Initialize camera manager.

        Args:
            main_window: Main window instance
            live_config: LiveConfig instance
            ui_manager: MainWindowUI instance
        """
        super().__init__(main_window)

        self.main_window = main_window
        self.live_config = live_config
        self.ui_manager = ui_manager

        self.camera_thread = None

        print("[CameraManager] Initialized")

    def start_camera(self):
        """Start the camera thread."""
        if self.camera_thread is not None:
            print("[CameraManager] Camera already running")
            return

        print("[CameraManager] Starting camera...")

        self.camera_thread = CameraThread(self.live_config)

        # Connect signals
        self.camera_thread.frame_ready.connect(self._on_frame_ready)
        self.camera_thread.error_occurred.connect(self._on_error)
        self.camera_thread.fps_updated.connect(self._on_fps_updated)

        # Start thread
        self.camera_thread.start()

        # Update UI
        self.ui_manager.update_status("Status: Running", "#00ff00")
        self.ui_manager.update_camera_button("Stop Camera")

        # Emit state change
        self.camera_state_changed.emit(True)

        print("[CameraManager] Camera started")

    def stop_camera(self):
        """Stop the camera thread."""
        if self.camera_thread is None:
            print("[CameraManager] Camera not running")
            return

        print("[CameraManager] Stopping camera...")

        self.camera_thread.stop()
        self.camera_thread = None

        # Update UI
        self.ui_manager.update_status("Status: Camera Stopped", "#ff9900")
        self.ui_manager.update_camera_button("Start Camera")

        # Emit state change
        self.camera_state_changed.emit(False)

        print("[CameraManager] Camera stopped")

    def toggle_camera(self):
        """Toggle camera on/off."""
        if self.camera_thread and self.camera_thread.running:
            self.stop_camera()
        else:
            self.start_camera()

    def is_running(self):
        """Check if camera is running."""
        return self.camera_thread is not None and self.camera_thread.running

    @pyqtSlot(np.ndarray, object, object, bool, bool)
    def _on_frame_ready(self, frame, pose_results, face_results, mouth_open, eyes_closed):
        """Handle new frame from camera thread."""
        # Forward signal
        self.frame_ready.emit(frame, pose_results, face_results, mouth_open, eyes_closed)

    @pyqtSlot(str)
    def _on_error(self, error_msg):
        """Handle error from camera thread."""
        self.ui_manager.update_status(f"Status: Error - {error_msg}", "#ff0000")
        print(f"[CameraManager] Error: {error_msg}")

        # Forward signal
        self.error_occurred.emit(error_msg)

    @pyqtSlot(float)
    def _on_fps_updated(self, fps):
        """Handle FPS update from camera thread."""
        # Forward signal
        self.fps_updated.emit(fps)
