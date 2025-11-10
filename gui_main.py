"""
GUI Main Window for Stickfigure Webcam Application.

This module provides a user-friendly GUI interface for the stickfigure webcam
application with live configuration changes and threaded camera processing.
"""

import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtCore import (
    Qt, pyqtSignal, QThread, pyqtSlot
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QCheckBox,
    QComboBox, QTabWidget,
    QScrollArea, QSizePolicy, QMessageBox
)

import config
from stickfigure import draw_stickfigure
from ui import draw_no_person_message
from virtual_camera import VirtualCameraOutput


@dataclass
class LiveConfig:
    """Thread-safe configuration that can be updated in real-time."""

    # Appearance settings
    stickfigure_thickness: int = 4
    joint_radius: int = 6
    eye_radius_ratio: float = 0.12
    mouth_width_ratio: float = 0.5
    mouth_height_ratio: float = 0.25

    # Head proportions
    head_radius_ratio: float = 0.4
    neck_length_ratio: float = 0.6
    eye_y_offset_ratio: float = 0.25
    eye_spacing_ratio: float = 0.35
    mouth_y_offset_ratio: float = 0.4
    shoulder_curve_depth_ratio: float = 0.15

    # Detection sensitivity
    mouth_open_threshold_ratio: float = 0.025
    eyes_closed_ratio_threshold: float = 0.055
    eyes_closed_consecutive_frames: int = 3

    # MediaPipe settings
    pose_min_detection_confidence: float = 0.5
    pose_min_tracking_confidence: float = 0.5
    pose_model_complexity: int = 0

    # Virtual camera
    vcam_mirror_output: bool = True

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs):
        """Thread-safe update of configuration values."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def get_snapshot(self):
        """Get thread-safe snapshot of all values."""
        with self._lock:
            return {
                'stickfigure_thickness': self.stickfigure_thickness,
                'joint_radius': self.joint_radius,
                'eye_radius_ratio': self.eye_radius_ratio,
                'mouth_width_ratio': self.mouth_width_ratio,
                'mouth_height_ratio': self.mouth_height_ratio,
                'head_radius_ratio': self.head_radius_ratio,
                'neck_length_ratio': self.neck_length_ratio,
                'eye_y_offset_ratio': self.eye_y_offset_ratio,
                'eye_spacing_ratio': self.eye_spacing_ratio,
                'mouth_y_offset_ratio': self.mouth_y_offset_ratio,
                'shoulder_curve_depth_ratio': self.shoulder_curve_depth_ratio,
                'mouth_open_threshold_ratio': self.mouth_open_threshold_ratio,
                'eyes_closed_ratio_threshold': self.eyes_closed_ratio_threshold,
                'eyes_closed_consecutive_frames': self.eyes_closed_consecutive_frames,
                'pose_min_detection_confidence': self.pose_min_detection_confidence,
                'pose_min_tracking_confidence': self.pose_min_tracking_confidence,
                'pose_model_complexity': self.pose_model_complexity,
                'vcam_mirror_output': self.vcam_mirror_output,
            }


class CameraThread(QThread):
    """Thread for camera capture and processing."""

    frame_ready = pyqtSignal(np.ndarray, object, object, bool, bool)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float)

    def __init__(self, live_config: LiveConfig):
        super().__init__()
        self.live_config = live_config
        self.running = False
        self.cap = None
        self.pose = None
        self.face_mesh = None
        self.eyes_closed_frame_counter = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

    def run(self):
        """Main camera processing loop."""
        try:
            self._initialize_camera()
            self._initialize_mediapipe()

            self.running = True
            print("[CameraThread] Started successfully")

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("[CameraThread] Failed to read frame")
                    time.sleep(0.01)
                    continue

                # Mirror frame
                frame = cv2.flip(frame, 1)

                # Process frame
                self._process_frame(frame)

                # Update FPS
                self._update_fps()

        except Exception as e:
            print(f"[CameraThread] Error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self._cleanup()

    def _initialize_camera(self):
        """Initialize camera with configured settings."""
        self.cap = cv2.VideoCapture(config.CAMERA_ID)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        print(f"[CameraThread] Camera initialized: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")

    def _initialize_mediapipe(self):
        """Initialize MediaPipe models with current config."""
        mp_pose = mp.solutions.pose
        mp_face_mesh = mp.solutions.face_mesh

        cfg = self.live_config.get_snapshot()

        self.pose = mp_pose.Pose(
            min_detection_confidence=cfg['pose_min_detection_confidence'],
            min_tracking_confidence=cfg['pose_min_tracking_confidence'],
            model_complexity=cfg['pose_model_complexity']
        )

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=config.FACE_MESH_MAX_FACES,
            min_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONFIDENCE
        )
        print("[CameraThread] MediaPipe initialized")

    def _process_frame(self, frame):
        """Process single frame through detection pipeline."""
        # Resize for faster processing
        processing_frame = cv2.resize(
            frame,
            (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
        )

        frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

        # Run detections
        pose_results = self.pose.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)

        mouth_open = False
        eyes_closed = False

        # Get current thresholds
        cfg = self.live_config.get_snapshot()

        # Process face landmarks
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark

            # Detect mouth opening with current threshold
            mouth_open = self._detect_mouth_opening(
                landmarks,
                cfg['mouth_open_threshold_ratio']
            )

            # Detect closed eyes with current threshold
            eyes_closed = self._detect_eyes_closed(
                landmarks,
                cfg['eyes_closed_ratio_threshold'],
                cfg['eyes_closed_consecutive_frames']
            )
        else:
            self.eyes_closed_frame_counter = 0

        # Emit frame with detection results
        self.frame_ready.emit(
            frame,
            pose_results,
            face_results,
            mouth_open,
            eyes_closed
        )

    def _detect_mouth_opening(self, landmarks, threshold_ratio):
        """Detect if mouth is open using current threshold."""
        try:
            upper_y = landmarks[config.MOUTH_LANDMARKS['upper_lip_top']].y * config.PROCESSING_HEIGHT
            lower_y = landmarks[config.MOUTH_LANDMARKS['lower_lip_bottom']].y * config.PROCESSING_HEIGHT
            forehead_y = landmarks[config.MOUTH_LANDMARKS['forehead']].y * config.PROCESSING_HEIGHT
            chin_y = landmarks[config.MOUTH_LANDMARKS['chin']].y * config.PROCESSING_HEIGHT

            mouth_distance = abs(lower_y - upper_y)
            face_height = abs(chin_y - forehead_y)

            if face_height == 0:
                return False

            threshold = face_height * threshold_ratio
            return mouth_distance > threshold
        except:
            return False

    def _detect_eyes_closed(self, landmarks, ratio_threshold, consecutive_frames):
        """Detect if eyes are closed using current threshold."""
        try:
            def get_coords(idx):
                lm = landmarks[idx]
                return int(lm.x * config.PROCESSING_WIDTH), int(lm.y * config.PROCESSING_HEIGHT)

            left_eye_top = get_coords(config.LEFT_EYE_TOP)
            left_eye_bottom = get_coords(config.LEFT_EYE_BOTTOM)
            right_eye_top = get_coords(config.RIGHT_EYE_TOP)
            right_eye_bottom = get_coords(config.RIGHT_EYE_BOTTOM)
            forehead = get_coords(config.MOUTH_LANDMARKS['forehead'])
            chin = get_coords(config.MOUTH_LANDMARKS['chin'])

            left_eye_dist = np.sqrt((left_eye_top[0] - left_eye_bottom[0]) ** 2 +
                                    (left_eye_top[1] - left_eye_bottom[1]) ** 2)
            right_eye_dist = np.sqrt((right_eye_top[0] - right_eye_bottom[0]) ** 2 +
                                     (right_eye_top[1] - right_eye_bottom[1]) ** 2)

            avg_eye_dist = (left_eye_dist + right_eye_dist) * 0.5
            face_height = np.sqrt((forehead[0] - chin[0]) ** 2 + (forehead[1] - chin[1]) ** 2)

            if face_height == 0:
                return False

            ear_ratio = avg_eye_dist / face_height

            if ear_ratio < ratio_threshold:
                self.eyes_closed_frame_counter += 1
            else:
                self.eyes_closed_frame_counter = 0

            return self.eyes_closed_frame_counter >= consecutive_frames

        except:
            return False

    def _update_fps(self):
        """Calculate and emit FPS."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            elapsed = time.time() - self.fps_start_time
            if elapsed > 0:
                self.current_fps = self.fps_counter / elapsed
                self.fps_updated.emit(self.current_fps)
            self.fps_counter = 0
            self.fps_start_time = time.time()

    def stop(self):
        """Stop the camera thread."""
        print("[CameraThread] Stopping...")
        self.running = False
        self.wait(2000)  # Wait up to 2 seconds

    def _cleanup(self):
        """Clean up resources."""
        if self.pose:
            self.pose.close()
        if self.face_mesh:
            self.face_mesh.close()
        if self.cap:
            self.cap.release()
        print("[CameraThread] Cleaned up")


class StickfigureWidget(QLabel):
    """Widget for displaying the stickfigure output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black; border: 2px solid #555;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class ControlPanel(QWidget):
    """Control panel with all configuration options."""

    config_changed = pyqtSignal(dict)

    def __init__(self, live_config: LiveConfig, parent=None):
        super().__init__(parent)
        self.live_config = live_config
        self._init_ui()

    def _init_ui(self):
        """Initialize the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Create tabs for different settings
        tabs = QTabWidget()
        tabs.addTab(self._create_appearance_tab(), "Appearance")
        tabs.addTab(self._create_detection_tab(), "Detection")
        tabs.addTab(self._create_advanced_tab(), "Advanced")

        layout.addWidget(tabs)
        layout.addStretch()

        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        layout.addWidget(reset_btn)

    def _create_appearance_tab(self):
        """Create appearance settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Line thickness
        layout.addWidget(self._create_slider_group(
            "Line Thickness",
            "stickfigure_thickness",
            1, 10, 1,
            self.live_config.stickfigure_thickness
        ))

        # Joint size
        layout.addWidget(self._create_slider_group(
            "Joint Size",
            "joint_radius",
            2, 15, 1,
            self.live_config.joint_radius
        ))

        # Head size
        layout.addWidget(self._create_double_slider_group(
            "Head Size",
            "head_radius_ratio",
            0.2, 0.8, 0.05,
            self.live_config.head_radius_ratio
        ))

        # Eye size
        layout.addWidget(self._create_double_slider_group(
            "Eye Size",
            "eye_radius_ratio",
            0.05, 0.25, 0.01,
            self.live_config.eye_radius_ratio
        ))

        # Mouth width
        layout.addWidget(self._create_double_slider_group(
            "Mouth Width",
            "mouth_width_ratio",
            0.2, 0.8, 0.05,
            self.live_config.mouth_width_ratio
        ))

        # Mouth height
        layout.addWidget(self._create_double_slider_group(
            "Mouth Height (Open)",
            "mouth_height_ratio",
            0.1, 0.5, 0.05,
            self.live_config.mouth_height_ratio
        ))

        layout.addStretch()
        return widget

    def _create_detection_tab(self):
        """Create detection sensitivity tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Mouth sensitivity
        layout.addWidget(self._create_double_slider_group(
            "Mouth Open Sensitivity",
            "mouth_open_threshold_ratio",
            0.01, 0.05, 0.005,
            self.live_config.mouth_open_threshold_ratio,
            tooltip="Lower = more sensitive"
        ))

        # Eye sensitivity
        layout.addWidget(self._create_double_slider_group(
            "Eye Closed Sensitivity",
            "eyes_closed_ratio_threshold",
            0.03, 0.08, 0.005,
            self.live_config.eyes_closed_ratio_threshold,
            tooltip="Lower = more sensitive"
        ))

        # Eye frames
        layout.addWidget(self._create_slider_group(
            "Eye Closed Frame Delay",
            "eyes_closed_consecutive_frames",
            1, 10, 1,
            self.live_config.eyes_closed_consecutive_frames,
            tooltip="Frames before eyes register as closed"
        ))

        layout.addStretch()
        return widget

    def _create_advanced_tab(self):
        """Create advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Pose detection confidence
        layout.addWidget(self._create_double_slider_group(
            "Pose Detection Confidence",
            "pose_min_detection_confidence",
            0.1, 1.0, 0.1,
            self.live_config.pose_min_detection_confidence
        ))

        # Pose tracking confidence
        layout.addWidget(self._create_double_slider_group(
            "Pose Tracking Confidence",
            "pose_min_tracking_confidence",
            0.1, 1.0, 0.1,
            self.live_config.pose_min_tracking_confidence
        ))

        # Model complexity
        complexity_group = QGroupBox("Model Complexity")
        complexity_layout = QHBoxLayout(complexity_group)
        complexity_combo = QComboBox()
        complexity_combo.addItems(["Lite (Fast)", "Full (Balanced)", "Heavy (Accurate)"])
        complexity_combo.setCurrentIndex(self.live_config.pose_model_complexity)
        complexity_combo.currentIndexChanged.connect(
            lambda idx: self._on_config_changed("pose_model_complexity", idx)
        )
        complexity_layout.addWidget(QLabel("Mode:"))
        complexity_layout.addWidget(complexity_combo)
        complexity_layout.addStretch()
        layout.addWidget(complexity_group)

        # Neck length
        layout.addWidget(self._create_double_slider_group(
            "Neck Length",
            "neck_length_ratio",
            0.2, 1.0, 0.1,
            self.live_config.neck_length_ratio
        ))

        # Shoulder curve
        layout.addWidget(self._create_double_slider_group(
            "Shoulder Curve Depth",
            "shoulder_curve_depth_ratio",
            0.0, 0.3, 0.05,
            self.live_config.shoulder_curve_depth_ratio
        ))

        layout.addStretch()
        return widget

    def _create_slider_group(self, label, config_key, min_val, max_val,
                             step, initial_value, tooltip=None):
        """Create a labeled slider group for integer values."""
        group = QGroupBox(label)
        if tooltip:
            group.setToolTip(tooltip)

        layout = QHBoxLayout(group)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setSingleStep(step)
        slider.setValue(initial_value)

        value_label = QLabel(str(initial_value))
        value_label.setMinimumWidth(40)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        slider.valueChanged.connect(
            lambda v: (value_label.setText(str(v)),
                       self._on_config_changed(config_key, v))
        )

        layout.addWidget(slider)
        layout.addWidget(value_label)

        return group

    def _create_double_slider_group(self, label, config_key, min_val, max_val,
                                    step, initial_value, tooltip=None):
        """Create a labeled slider group for float values."""
        group = QGroupBox(label)
        if tooltip:
            group.setToolTip(tooltip)

        layout = QHBoxLayout(group)

        # Convert to integer range for slider
        multiplier = int(1 / step)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(min_val * multiplier))
        slider.setMaximum(int(max_val * multiplier))
        slider.setSingleStep(1)
        slider.setValue(int(initial_value * multiplier))

        value_label = QLabel(f"{initial_value:.3f}")
        value_label.setMinimumWidth(50)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        def on_change(v):
            real_value = v / multiplier
            value_label.setText(f"{real_value:.3f}")
            self._on_config_changed(config_key, real_value)

        slider.valueChanged.connect(on_change)

        layout.addWidget(slider)
        layout.addWidget(value_label)

        return group

    def _on_config_changed(self, key, value):
        """Handle configuration change."""
        self.live_config.update(**{key: value})
        self.config_changed.emit({key: value})

    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Create new default config
            default = LiveConfig()

            # Update live config
            self.live_config.update(**default.get_snapshot())

            # Rebuild UI to reflect changes
            print("[ControlPanel] Settings reset to defaults")
            QMessageBox.information(self, "Reset Complete",
                                    "Please restart the application for all changes to take effect.")


class MainWindow(QMainWindow):
    """Main application window."""

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

        # Stop camera thread
        if self.camera_thread:
            self.camera_thread.stop()

        # Stop virtual camera
        if self.vcam and self.vcam.is_active:
            self.vcam.stop()

        print("[MainWindow] Cleanup complete")
        event.accept()


def main():
    """Application entry point."""
    print("=" * 60)
    print("STICKFIGURE WEBCAM - GUI MODE")
    print("=" * 60)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look across platforms

    # Set dark theme
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #0d47a1;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1565c0;
        }
        QPushButton:pressed {
            background-color: #0a3d91;
        }
        QSlider::groove:horizontal {
            border: 1px solid #555;
            height: 8px;
            background: #3a3a3a;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #0d47a1;
            border: 1px solid #0a3d91;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #1565c0;
        }
        QCheckBox {
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #555;
            border-radius: 3px;
            background: #3a3a3a;
        }
        QCheckBox::indicator:checked {
            background: #0d47a1;
            border-color: #0d47a1;
        }
        QComboBox {
            background-color: #3a3a3a;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px;
        }
        QComboBox:hover {
            border-color: #0d47a1;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: #3a3a3a;
            selection-background-color: #0d47a1;
        }
        QScrollArea {
            border: none;
        }
        QScrollBar:vertical {
            background: #2b2b2b;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #555;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover {
            background: #777;
        }
        QTabWidget::pane {
            border: 1px solid #555;
            border-radius: 4px;
        }
        QTabBar::tab {
            background: #3a3a3a;
            color: #ffffff;
            padding: 8px 16px;
            border: 1px solid #555;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #0d47a1;
        }
        QTabBar::tab:hover {
            background: #1565c0;
        }
    """)

    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import sys

    # Check for --no-gui flag
    if '--no-gui' in sys.argv or '--cli' in sys.argv:
        # Import CLI version
        from main import main as cli_main

        cli_main()
    else:
        # Try GUI, fall back to CLI
        try:
            main()  # GUI main
        except ImportError:
            print("GUI not available, falling back to CLI...")
            from main import main as cli_main

            cli_main()
