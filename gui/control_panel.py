"""
Control panel widget for GUI application.

This module provides the settings panel with tabs for appearance,
detection, and advanced configuration options.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QGroupBox, QComboBox,
    QTabWidget, QMessageBox
)


class ControlPanel(QWidget):
    """
    Control panel with all configuration options.

    Provides tabbed interface for configuring appearance, detection
    sensitivity, and advanced MediaPipe settings with real-time updates.
    """

    config_changed = pyqtSignal(dict)

    def __init__(self, live_config, parent=None):
        """
        Initialize the control panel.

        Args:
            live_config: LiveConfig instance to update.
            parent: Parent widget.
        """
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
            self.live_config.reset_to_defaults()
            print("[ControlPanel] Settings reset to defaults")
            QMessageBox.information(
                self,
                "Reset Complete",
                "Please restart the application for all changes to take effect."
            )