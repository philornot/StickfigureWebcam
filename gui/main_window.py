"""
Main window for GUI application.

This module provides the main application window with video preview,
status indicators, control panel integration, system tray support,
and debug window.
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont, QCloseEvent
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QScrollArea, QSizePolicy
)

import config
from gui.camera_thread import CameraThread
from gui.config_manager import LiveConfig
from gui.control_panel import ControlPanel
from gui.debug_window import DebugWindow
from stickfigure import draw_stickfigure
from system_tray import SystemTray
from ui import draw_no_person_message
from virtual_camera import VirtualCameraOutput


class StickfigureWidget(QLabel):
    """Widget for displaying the stickfigure output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black; border: 2px solid #555;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class MainWindow(QMainWindow):
    """
    Main application window.

    Provides the primary GUI with video preview, control panel,
    status indicators, system tray integration, and debug window
    for the stickfigure webcam application.
    """

    def __init__(self):
        super().__init__()
        self.live_config = LiveConfig()
        self.camera_thread = None
        self.vcam = None
        self.current_frame_data = None
        self.debug_window = None
        self.system_tray = None
        self.is_quitting = False

        self._init_ui()
        self._init_virtual_camera()
        self._init_system_tray()
        self._start_camera()

    def _init_ui(self):
        """Initialize the main UI."""
        self.setWindowTitle("Stickfigure Webcam - Control Panel")
        self.setMinimumSize(1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left side: Video preview
        left_layout = QVBoxLayout()

        # Preview label
        preview_label = QLabel("Stickfigure Output")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        preview_label.setFont(font)
        left_layout.addWidget(preview_label)

        # Video widget
        self.video_widget = StickfigureWidget()
        left_layout.addWidget(self.video_widget)

        # Status bar
        status_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setStyleSheet("color: #ffff00;")
        status_layout.addWidget(self.fps_label)
        status_layout.addStretch()
        status_layout.addWidget(self.status_label)
        left_layout.addLayout(status_layout)

        # Virtual camera controls
        vcam_layout = QHBoxLayout()
        self.vcam_status_label = QLabel("Virtual Camera: Not Started")
        self.vcam_mirror_checkbox = QCheckBox("Mirror Output")
        self.vcam_mirror_checkbox.setChecked(self.live_config.vcam_mirror_output)
        self.vcam_mirror_checkbox.stateChanged.connect(self._on_mirror_changed)
        vcam_layout.addWidget(self.vcam_status_label)
        vcam_layout.addStretch()
        vcam_layout.addWidget(self.vcam_mirror_checkbox)
        left_layout.addLayout(vcam_layout)

        # Debug window controls
        debug_layout = QHBoxLayout()
        self.debug_window_checkbox = QCheckBox("Show Debug Window")
        self.debug_window_checkbox.stateChanged.connect(self._on_debug_window_toggled)
        debug_layout.addWidget(self.debug_window_checkbox)
        debug_layout.addStretch()
        left_layout.addLayout(debug_layout)

        main_layout.addLayout(left_layout, 2)

        # Right side: Control panel in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(500)

        self.control_panel = ControlPanel(self.live_config)
        self.control_panel.config_changed.connect(self._on_config_changed)
        scroll.setWidget(self.control_panel)

        main_layout.addWidget(scroll, 1)

        print("[MainWindow] UI initialized")

    def _init_virtual_camera(self):
        """Initialize virtual camera output."""
        try:
            self.vcam = VirtualCameraOutput(
                config.CAMERA_WIDTH,
                config.CAMERA_HEIGHT,
                fps=config.CAMERA_FPS,
                mirror=self.live_config.vcam_mirror_output
            )
            if self.vcam.start():
                self.vcam_status_label.setText(
                    f"Virtual Camera: Active ({self.vcam.camera.device})"
                )
                self.vcam_status_label.setStyleSheet("color: #00ff00; font-weight: bold;")
                print("[MainWindow] Virtual camera started")
            else:
                self.vcam_status_label.setText("Virtual Camera: Failed to Start")
                self.vcam_status_label.setStyleSheet("color: #ff0000;")
        except Exception as e:
            print(f"[MainWindow] Virtual camera error: {e}")
            self.vcam_status_label.setText("Virtual Camera: Error")
            self.vcam_status_label.setStyleSheet("color: #ff0000;")

    def _init_system_tray(self):
        """Initialize system tray icon."""
        self.system_tray = SystemTray("Stickfigure Webcam")

        # Set callbacks
        self.system_tray.set_on_show(self._on_tray_show_hide)
        self.system_tray.set_on_toggle_camera(self._on_tray_toggle_camera)
        self.system_tray.set_on_toggle_debug(self._on_tray_toggle_debug)
        self.system_tray.set_on_settings(self._on_tray_settings)
        self.system_tray.set_on_quit(self._on_tray_quit)

        # Start tray
        self.system_tray.start()
        self.system_tray.set_camera_state(True)  # Camera starts automatically

        print("[MainWindow] System tray initialized")

    def _start_camera(self):
        """Start the camera thread."""
        self.camera_thread = CameraThread(self.live_config)
        self.camera_thread.frame_ready.connect(self._on_frame_ready)
        self.camera_thread.error_occurred.connect(self._on_error)
        self.camera_thread.fps_updated.connect(self._on_fps_updated)
        self.camera_thread.start()

        self.status_label.setText("Status: Running")
        self.status_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        print("[MainWindow] Camera thread started")

    @pyqtSlot(np.ndarray, object, object, bool, bool)
    def _on_frame_ready(self, frame, pose_results, face_results, mouth_open, eyes_closed):
        """Handle new frame from camera thread."""
        # Store for rendering
        self.current_frame_data = {
            'frame': frame,
            'pose_results': pose_results,
            'face_results': face_results,
            'mouth_open': mouth_open,
            'eyes_closed': eyes_closed
        }

        # Render stickfigure
        self._render_stickfigure()

        # Update debug window if open
        if self.debug_window and self.debug_window.isVisible():
            self.debug_window.update_frame_data(self.current_frame_data)

    def _render_stickfigure(self):
        """Render stickfigure with current configuration."""
        if not self.current_frame_data:
            return

        data = self.current_frame_data
        frame = data['frame']
        height, width = frame.shape[:2]

        # Create canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Get current config snapshot
        cfg = self.live_config.get_snapshot()

        # Temporarily update config module for drawing functions
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

        # Draw stickfigure or message
        if data['pose_results'].pose_landmarks:
            draw_stickfigure(
                canvas,
                data['pose_results'].pose_landmarks.landmark,
                width,
                height,
                data['mouth_open'],
                data['eyes_closed'],
                draw_debug_markers=False
            )
        else:
            draw_no_person_message(canvas, width, height)

        # Convert to QImage and display
        rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_widget.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_widget.setPixmap(scaled_pixmap)

        # Send to virtual camera if active
        if self.vcam and self.vcam.is_active:
            try:
                self.vcam.send_frame(canvas)
            except Exception as e:
                print(f"[MainWindow] Virtual camera send error: {e}")

    @pyqtSlot(str)
    def _on_error(self, error_msg):
        """Handle error from camera thread."""
        self.status_label.setText(f"Status: Error - {error_msg}")
        self.status_label.setStyleSheet("color: #ff0000; font-weight: bold;")
        print(f"[MainWindow] Error: {error_msg}")

    @pyqtSlot(float)
    def _on_fps_updated(self, fps):
        """Handle FPS update."""
        self.fps_label.setText(f"FPS: {fps:.1f}")

        # Update debug window if open
        if self.debug_window and self.debug_window.isVisible():
            self.debug_window.update_fps(fps)

    @pyqtSlot(dict)
    def _on_config_changed(self, changes):
        """Handle configuration change."""
        # Re-render current frame with new settings
        if self.current_frame_data:
            self._render_stickfigure()

    def _on_mirror_changed(self, state):
        """Handle mirror checkbox change."""
        mirror = (state == Qt.CheckState.Checked.value)
        self.live_config.update(vcam_mirror_output=mirror)
        if self.vcam:
            self.vcam.set_mirror(mirror)

    def _on_debug_window_toggled(self, state):
        """Handle debug window checkbox toggle."""
        if state == Qt.CheckState.Checked.value:
            self._show_debug_window()
        else:
            self._hide_debug_window()

    def _show_debug_window(self):
        """Show the debug window."""
        if not self.debug_window:
            self.debug_window = DebugWindow(self)
            self.debug_window.set_live_config(self.live_config)

        self.debug_window.show()
        self.system_tray.set_debug_visible(True)
        print("[MainWindow] Debug window shown")

    def _hide_debug_window(self):
        """Hide the debug window."""
        if self.debug_window:
            self.debug_window.hide()

        self.system_tray.set_debug_visible(False)
        self.debug_window_checkbox.setChecked(False)
        print("[MainWindow] Debug window hidden")

    # System tray callbacks
    def _on_tray_show_hide(self):
        """Handle show/hide from tray."""
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.activateWindow()

    def _on_tray_toggle_camera(self):
        """Handle toggle camera from tray."""
        if self.camera_thread and self.camera_thread.running:
            # Stop camera
            self.camera_thread.stop()
            self.system_tray.set_camera_state(False)
            self.status_label.setText("Status: Camera Stopped")
            self.status_label.setStyleSheet("color: #ff9900;")
            self.system_tray.show_notification(
                "Camera Stopped",
                "Camera has been stopped"
            )
        else:
            # Restart camera
            self._start_camera()
            self.system_tray.set_camera_state(True)
            self.system_tray.show_notification(
                "Camera Started",
                "Camera has been started"
            )

    def _on_tray_toggle_debug(self):
        """Handle toggle debug window from tray."""
        if self.debug_window and self.debug_window.isVisible():
            self._hide_debug_window()
        else:
            self._show_debug_window()
            self.debug_window_checkbox.setChecked(True)

    def _on_tray_settings(self):
        """Handle settings from tray."""
        self.show()
        self.activateWindow()

    def _on_tray_quit(self):
        """Handle quit from tray."""
        self.is_quitting = True
        self.close()

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event."""
        # Check if minimize to tray is enabled
        if self.system_tray and self.system_tray.minimize_to_tray_enabled and not self.is_quitting:
            print("[MainWindow] Minimizing to tray...")
            self.hide()
            self.system_tray.show_notification(
                "Minimized to Tray",
                "Application is still running in the system tray"
            )
            event.ignore()
            return

        # Actually quitting
        print("[MainWindow] Closing application...")

        # Flush configuration to disk immediately
        self.live_config.flush_to_disk()

        # Close debug window
        if self.debug_window:
            self.debug_window.close()

        # Stop camera thread
        if self.camera_thread:
            self.camera_thread.stop()

        # Stop virtual camera
        if self.vcam and self.vcam.is_active:
            self.vcam.stop()

        # Stop system tray
        if self.system_tray:
            self.system_tray.stop()

        print("[MainWindow] Cleanup complete")
        event.accept()
