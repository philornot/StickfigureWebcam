"""
Debug window for GUI application.

This module provides a separate debug window showing the camera feed
with detection overlays and debug information.
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel,
    QSizePolicy
)

import config
from stickfigure import draw_stickfigure
from face_detection import draw_face_landmarks


class DebugWidget(QLabel):
    """Widget for displaying the debug camera feed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black; border: 2px solid #555;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class DebugWindow(QMainWindow):
    """
    Debug window showing camera feed with detection overlays.

    This window displays the raw camera feed with pose detection markers,
    face landmarks, and real-time debug information overlaid.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_frame_data = None
        self.current_fps = 0.0
        self.live_config = None

        self._init_ui()

        # Update timer for rendering (30 FPS)
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._render_debug)
        self.render_timer.start(33)  # ~30 FPS

        print("[DebugWindow] Initialized")

    def _init_ui(self):
        """Initialize the debug window UI."""
        self.setWindowTitle("Stickfigure Webcam - Debug View")
        self.setMinimumSize(800, 600)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Title label
        title_label = QLabel("Debug Camera View")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)

        # Debug video widget
        self.debug_widget = DebugWidget()
        layout.addWidget(self.debug_widget)

        # Info label
        self.info_label = QLabel("Waiting for camera data...")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(self.info_label)

    def set_live_config(self, live_config):
        """
        Set the live configuration instance.

        Args:
            live_config: LiveConfig instance to use for rendering.
        """
        self.live_config = live_config

    @pyqtSlot(object)
    def update_frame_data(self, frame_data):
        """
        Update the current frame data for rendering.

        Args:
            frame_data: Dictionary containing frame and detection results.
        """
        self.current_frame_data = frame_data

    @pyqtSlot(float)
    def update_fps(self, fps):
        """
        Update the FPS counter.

        Args:
            fps: Current frames per second.
        """
        self.current_fps = fps

    def _render_debug(self):
        """Render the debug view with current frame data."""
        if not self.current_frame_data:
            return

        data = self.current_frame_data
        frame = data['frame'].copy()
        height, width = frame.shape[:2]

        # Draw stick figure overlay with debug markers
        if data['pose_results'].pose_landmarks:
            landmarks = data['pose_results'].pose_landmarks.landmark

            # Temporarily update config for drawing
            if self.live_config:
                cfg = self.live_config.get_snapshot()
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

            # Draw stickfigure with debug markers
            draw_stickfigure(
                frame,
                landmarks,
                width,
                height,
                data['mouth_open'],
                data['eyes_closed'],
                draw_debug_markers=True
            )

        # Draw face landmarks
        if data['face_results'].multi_face_landmarks:
            draw_face_landmarks(
                frame,
                data['face_results'].multi_face_landmarks[0].landmark,
                width,
                height
            )

        # Add debug information overlay
        self._draw_debug_info(frame, data, width, height)

        # Convert to QImage and display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.debug_widget.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.debug_widget.setPixmap(scaled_pixmap)

        # Update info label
        status = "Person detected" if data['pose_results'].pose_landmarks else "No person"
        self.info_label.setText(
            f"FPS: {self.current_fps:.1f} | {status} | "
            f"Mouth: {'OPEN' if data['mouth_open'] else 'CLOSED'} | "
            f"Eyes: {'CLOSED' if data['eyes_closed'] else 'OPEN'}"
        )

    def _draw_debug_info(self, canvas, data, width, height):
        """
        Draw debug information overlay on the frame.

        Args:
            canvas: Frame to draw on.
            data: Detection data dictionary.
            width: Frame width.
            height: Frame height.
        """
        text_scale = config.DEBUG_TEXT_SCALE
        text_thickness = config.DEBUG_TEXT_THICKNESS

        # FPS display
        cv2.putText(
            canvas,
            f'FPS: {self.current_fps:.1f}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 255, 0),
            text_thickness
        )

        # Detection status
        status = "Person Detected" if data['pose_results'].pose_landmarks else "No Person"
        status_color = (0, 255, 0) if data['pose_results'].pose_landmarks else (0, 0, 255)
        cv2.putText(
            canvas,
            status,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            status_color,
            text_thickness
        )

        # Mouth status
        mouth_status = "OPEN" if data['mouth_open'] else "CLOSED"
        cv2.putText(
            canvas,
            f'Mouth: {mouth_status}',
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 255, 255),
            text_thickness
        )

        # Eyes status
        eyes_status = "CLOSED" if data['eyes_closed'] else "OPEN"
        cv2.putText(
            canvas,
            f'Eyes: {eyes_status}',
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 255, 255),
            text_thickness
        )

        # Resolution info
        cv2.putText(
            canvas,
            f'Resolution: {width}x{height}',
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (128, 128, 128),
            1
        )

    def closeEvent(self, event):
        """Handle window close event."""
        print("[DebugWindow] Window closed")
        # Stop render timer
        self.render_timer.stop()
        event.accept()