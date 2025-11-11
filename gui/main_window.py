"""
Main window for GUI application.

This module provides the main application window with video preview,
status indicators, and control panel integration.
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QScrollArea, QSizePolicy
)

import config
from stickfigure import draw_stickfigure
from ui import draw_no_person_message
from virtual_camera import VirtualCameraOutput

from gui.config_manager import LiveConfig
from gui.camera_thread import CameraThread
from gui.control_panel import ControlPanel


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
    and status indicators for the stickfigure webcam application.
    """

    def __init__(self):
        super().__init__()
        self.live_config = LiveConfig()
        self.camera_thread = None
        self.vcam = None
        self.current_frame_data = None

        self._init_ui()
        self._init_virtual_camera()
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

    def closeEvent(self, event):
        """Handle window close event."""
        print("[MainWindow] Closing application...")

        # Flush configuration to disk immediately
        self.live_config.flush_to_disk()

        # Stop camera thread
        if self.camera_thread:
            self.camera_thread.stop()

        # Stop virtual camera
        if self.vcam and self.vcam.is_active:
            self.vcam.stop()

        print("[MainWindow] Cleanup complete")
        event.accept()