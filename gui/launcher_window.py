"""
Launcher window - main application startup window.

This window is displayed at application startup and contains all
settings and buttons to start the camera.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QMainWindow, QMessageBox

import config
from gui.camera_thread import CameraThread
from gui.config_manager import LiveConfig
from gui.launcher_window_ui import LauncherWindowUI
from system_tray import SystemTray
from virtual_camera import VirtualCameraOutput


class LauncherWindow(QMainWindow):
    """
    Main application launcher window with settings and controls.

    When the window is closed, the application minimizes to tray and runs
    in the background, ready to be used as a virtual camera.
    """

    # Signal for safe closing from another thread
    safe_close_signal = pyqtSignal()

    def __init__(self):
        """Initialize the launcher window."""
        super().__init__()

        # Application state
        self.live_config = LiveConfig()
        self.is_quitting = False
        self.minimize_to_tray = True

        # Components
        self.ui_manager = None
        self.camera_thread = None
        self.vcam = None
        self.system_tray = None

        # Camera state
        self.camera_running = False
        self.camera_preview_enabled = False

        # Initialization
        self._init_components()

        # Connect safe close signal
        self.safe_close_signal.connect(self.close)

    def _init_components(self):
        """Initialize all components."""
        # UI Manager
        self.ui_manager = LauncherWindowUI(self, self.live_config)
        self.ui_manager.setup_ui()

        # Connect UI signals
        self._connect_ui_signals()

        # System Tray
        self._init_system_tray()

        # Virtual camera (not started yet)
        self._init_virtual_camera()

        print("[LauncherWindow] Initialized")

    def _connect_ui_signals(self):
        """Connect signals from UI manager."""
        signals = self.ui_manager.signals

        signals.start_camera_clicked.connect(self._on_start_camera)
        signals.stop_camera_clicked.connect(self._on_stop_camera)
        signals.minimize_to_tray_changed.connect(self._on_minimize_to_tray_changed)
        signals.config_changed.connect(self._on_config_changed)
        signals.preview_toggled.connect(self._on_preview_toggled)

    def _init_system_tray(self):
        """Initialize system tray icon."""
        self.system_tray = SystemTray("Stickfigure Webcam")

        # Set callbacks
        self.system_tray.set_on_toggle_camera(self._on_tray_toggle_camera)
        self.system_tray.set_on_show_settings(self._on_tray_show_settings)
        self.system_tray.set_on_quit(self._on_tray_quit)

        # Start tray
        self.system_tray.start()
        self.system_tray.set_camera_state(False)

        print("[LauncherWindow] System tray initialized")

    def _init_virtual_camera(self):
        """Initialize virtual camera (without starting)."""
        try:
            self.vcam = VirtualCameraOutput(
                config.CAMERA_WIDTH,
                config.CAMERA_HEIGHT,
                fps=config.CAMERA_FPS,
                mirror=self.live_config.vcam_mirror_output
            )
            print("[LauncherWindow] Virtual camera initialized (not started)")
        except Exception as e:
            print(f"[LauncherWindow] Virtual camera init error: {e}")
            self.vcam = None

    def _on_start_camera(self):
        """Start camera and virtual camera."""
        if self.camera_running:
            print("[LauncherWindow] Camera already running")
            return

        print("[LauncherWindow] Starting camera...")

        # Start virtual camera
        if self.vcam and not self.vcam.is_active:
            if self.vcam.start():
                self.ui_manager.update_vcam_status(
                    f"✓ Virtual Camera Active ({self.vcam.camera.device})",
                    "#00ff00"
                )
            else:
                self.ui_manager.update_vcam_status(
                    "✗ Virtual Camera Failed",
                    "#ff0000"
                )
                QMessageBox.warning(
                    self,
                    "Virtual Camera Error",
                    "Failed to start virtual camera. Make sure OBS Virtual Camera "
                    "(Windows/Mac) or v4l2loopback (Linux) is installed."
                )
                return

        # Start camera thread
        self.camera_thread = CameraThread(self.live_config)
        self.camera_thread.frame_ready.connect(self._on_frame_ready)
        self.camera_thread.error_occurred.connect(self._on_camera_error)
        self.camera_thread.fps_updated.connect(self._on_fps_updated)
        self.camera_thread.start()

        self.camera_running = True

        # Update UI
        self.ui_manager.set_camera_running(True)
        self.ui_manager.update_status("Camera Active", "#00ff00")

        # Update tray
        if self.system_tray:
            self.system_tray.set_camera_state(True)

        print("[LauncherWindow] Camera started successfully")

    def _on_stop_camera(self):
        """Stop camera and virtual camera."""
        if not self.camera_running:
            print("[LauncherWindow] Camera not running")
            return

        print("[LauncherWindow] Stopping camera...")

        # Stop camera thread
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        # Stop virtual camera
        if self.vcam and self.vcam.is_active:
            self.vcam.stop()
            self.ui_manager.update_vcam_status(
                "Virtual Camera Stopped",
                "#ffaa00"
            )

        self.camera_running = False

        # Update UI
        self.ui_manager.set_camera_running(False)
        self.ui_manager.update_status("Camera Stopped", "#ffaa00")

        # Update tray
        if self.system_tray:
            self.system_tray.set_camera_state(False)

        print("[LauncherWindow] Camera stopped")

    def _on_frame_ready(self, frame, pose_results, face_results, mouth_open, eyes_closed):
        """
        Process new camera frame.

        Args:
            frame: Camera frame
            pose_results: Pose detection results
            face_results: Face detection results
            mouth_open: Whether mouth is open
            eyes_closed: Whether eyes are closed
        """
        if not self.camera_running:
            return

        # Render stickfigure
        canvas = self._render_stickfigure(
            frame, pose_results, face_results, mouth_open, eyes_closed
        )

        # Send to virtual camera
        if self.vcam and self.vcam.is_active:
            self.vcam.send_frame(canvas)

        # Update preview if enabled
        if self.camera_preview_enabled:
            self.ui_manager.update_preview(canvas)

    def _render_stickfigure(self, frame, pose_results, face_results, mouth_open, eyes_closed):
        """
        Render stickfigure based on detection.

        Args:
            frame: Camera frame
            pose_results: Pose detection results
            face_results: Face detection results
            mouth_open: Whether mouth is open
            eyes_closed: Whether eyes are closed

        Returns:
            np.ndarray: Rendered canvas
        """
        import numpy as np
        from stickfigure import draw_stickfigure
        from ui import draw_no_person_message

        height, width = frame.shape[:2]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Apply current configuration
        cfg = self.live_config.get_snapshot()
        self._apply_config_to_globals(cfg)

        # Draw stickfigure or message
        if pose_results.pose_landmarks:
            draw_stickfigure(
                canvas,
                pose_results.pose_landmarks.landmark,
                width,
                height,
                mouth_open,
                eyes_closed,
                draw_debug_markers=False
            )
        else:
            draw_no_person_message(canvas, width, height)

        return canvas

    def _apply_config_to_globals(self, cfg):
        """
        Apply configuration to global settings.

        Args:
            cfg: Configuration dictionary
        """
        config.STICKFIGURE_THICKNESS = cfg['stickfigure_thickness']
        config.JOINT_RADIUS = cfg['joint_radius']
        config.EYE_RADIUS_RATIO = cfg['eye_radius_ratio']
        config.MOUTH_WIDTH_RATIO = cfg['mouth_width_ratio']
        config.MOUTH_HEIGHT_RATIO = cfg['mouth_height_ratio']
        config.HEAD_RADIUS_RATIO = cfg['head_radius_ratio']
        config.NECK_LENGTH_RATIO = cfg['neck_length_ratio']
        config.EYE_Y_OFFSET_RATIO = cfg['eye_y_offset_ratio']
        config.EYE_SPACING_RATIO = cfg['eye_spacing_ratio']
        config.MOUTH_Y_OFFSET_RATIO = cfg['mouth_y_offset_ratio']
        config.SHOULDER_CURVE_DEPTH_RATIO = cfg['shoulder_curve_depth_ratio']

    def _on_camera_error(self, error_msg):
        """
        Handle camera error.

        Args:
            error_msg: Error message
        """
        print(f"[LauncherWindow] Camera error: {error_msg}")
        self.ui_manager.update_status(f"Error: {error_msg}", "#ff0000")

        QMessageBox.critical(
            self,
            "Camera Error",
            f"Camera error occurred:\n{error_msg}"
        )

        self._on_stop_camera()

    def _on_fps_updated(self, fps):
        """
        Update FPS display.

        Args:
            fps: Current FPS value
        """
        self.ui_manager.update_fps(fps)

    def _on_config_changed(self, changes):
        """
        Handle configuration change.

        Args:
            changes: Dictionary of changed configuration values
        """
        # If mirror setting changed
        if 'vcam_mirror_output' in changes and self.vcam:
            self.vcam.set_mirror(changes['vcam_mirror_output'])

    def _on_preview_toggled(self, enabled):
        """
        Enable/disable preview.

        Args:
            enabled: Whether preview is enabled
        """
        self.camera_preview_enabled = enabled
        if not enabled:
            self.ui_manager.clear_preview()

    def _on_minimize_to_tray_changed(self, enabled):
        """
        Change minimize to tray setting.

        Args:
            enabled: Whether minimize to tray is enabled
        """
        self.minimize_to_tray = enabled
        print(f"[LauncherWindow] Minimize to tray: {'enabled' if enabled else 'disabled'}")

    # System tray callbacks

    def _on_tray_toggle_camera(self):
        """Toggle camera from tray."""
        if self.camera_running:
            self._on_stop_camera()
            if self.system_tray:
                self.system_tray.show_notification(
                    "Camera Stopped",
                    "Virtual camera has been stopped"
                )
        else:
            self._on_start_camera()
            if self.system_tray:
                self.system_tray.show_notification(
                    "Camera Started",
                    "Virtual camera is now active"
                )

    def _on_tray_show_settings(self):
        """Show settings window from tray."""
        self.show()
        self.activateWindow()
        self.raise_()

    def _on_tray_quit(self):
        """Close application from tray."""
        self.is_quitting = True
        self.safe_close_signal.emit()

    def closeEvent(self, event: QCloseEvent):
        """
        Handle window close event.

        Args:
            event: Close event
        """
        # If explicitly quitting (from tray), always close
        if self.is_quitting:
            print("[LauncherWindow] Closing application...")
            self._cleanup()
            event.accept()
            return

        # If minimize to tray is enabled, minimize instead of closing
        if self.minimize_to_tray:
            print("[LauncherWindow] Minimizing to system tray...")
            self.hide()

            if self.system_tray:
                self.system_tray.show_notification(
                    "Stickfigure Webcam",
                    "Application minimized to system tray. "
                    "Right-click the icon to show settings or exit."
                )

            event.ignore()
            return

        # Otherwise, actually close
        print("[LauncherWindow] Closing application...")
        self._cleanup()
        event.accept()

    def _cleanup(self):
        """Clean up all resources."""
        # Save configuration
        self.live_config.flush_to_disk()

        # Stop camera
        if self.camera_running:
            self._on_stop_camera()

        # Stop system tray BEFORE calling QApplication.quit()
        if self.system_tray:
            self.system_tray.stop()
            self.system_tray = None

        print("[LauncherWindow] Cleanup complete")

        # Force quit the application
        from PyQt6.QtWidgets import QApplication
        QApplication.quit()
