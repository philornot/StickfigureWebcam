"""
Settings tabs for the main window.

Contains all configuration controls for the application,
organized into logical tabs.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QGroupBox, QComboBox,
    QTabWidget, QMessageBox, QCheckBox
)


class SettingsTabs(QWidget):
    """
    Widget with settings tabs.

    Organizes all settings in a clear interface
    with tabs: Appearance, Detection, Advanced.
    """

    config_changed = pyqtSignal(dict)

    def __init__(self, live_config, parent=None):
        """
        Initialize settings tabs.

        Args:
            live_config: LiveConfig instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.live_config = live_config
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Title
        title = QLabel("⚙️ Settings")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #ffffff;
            padding: 10px;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #404040;
                border-radius: 5px;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background: #1a1a1a;
                color: #ffffff;
                padding: 10px 20px;
                border: 1px solid #404040;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #0d47a1;
            }
            QTabBar::tab:hover {
                background: #1565c0;
            }
        """)

        tabs.addTab(self._create_appearance_tab(), "Appearance")
        tabs.addTab(self._create_detection_tab(), "Detection")
        tabs.addTab(self._create_advanced_tab(), "Advanced")

        layout.addWidget(tabs)

        # Reset button
        reset_layout = QHBoxLayout()
        reset_layout.addStretch()

        reset_btn = QPushButton("↺ Reset to Defaults")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:pressed {
                background-color: #444444;
            }
        """)
        reset_btn.clicked.connect(self._reset_to_defaults)
        reset_layout.addWidget(reset_btn)

        layout.addLayout(reset_layout)

    def _create_appearance_tab(self):
        """Create the appearance tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        # Line thickness
        layout.addWidget(self._create_slider_group(
            "Line Thickness",
            "stickfigure_thickness",
            1, 10, 1,
            self.live_config.stickfigure_thickness,
            tooltip="Thickness of stickfigure lines"
        ))

        # Joint size
        layout.addWidget(self._create_slider_group(
            "Joint Size",
            "joint_radius",
            2, 15, 1,
            self.live_config.joint_radius,
            tooltip="Size of joint circles"
        ))

        # Head size
        layout.addWidget(self._create_double_slider_group(
            "Head Size",
            "head_radius_ratio",
            0.2, 0.8, 0.05,
            self.live_config.head_radius_ratio,
            tooltip="Relative size of the head"
        ))

        # Eye size
        layout.addWidget(self._create_double_slider_group(
            "Eye Size",
            "eye_radius_ratio",
            0.05, 0.25, 0.01,
            self.live_config.eye_radius_ratio,
            tooltip="Relative size of the eyes"
        ))

        # Mouth width
        layout.addWidget(self._create_double_slider_group(
            "Mouth Width",
            "mouth_width_ratio",
            0.2, 0.8, 0.05,
            self.live_config.mouth_width_ratio,
            tooltip="Width of the mouth"
        ))

        # Mouth height (open)
        layout.addWidget(self._create_double_slider_group(
            "Mouth Height (Open)",
            "mouth_height_ratio",
            0.1, 0.5, 0.05,
            self.live_config.mouth_height_ratio,
            tooltip="Height when mouth is open"
        ))

        layout.addStretch()
        return widget

    def _create_detection_tab(self):
        """Create the detection tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        # Mouth sensitivity
        layout.addWidget(self._create_double_slider_group(
            "Mouth Open Sensitivity",
            "mouth_open_threshold_ratio",
            0.01, 0.05, 0.005,
            self.live_config.mouth_open_threshold_ratio,
            tooltip="Lower = more sensitive (mouth opens easier)"
        ))

        # Eye sensitivity
        layout.addWidget(self._create_double_slider_group(
            "Eye Closed Sensitivity",
            "eyes_closed_ratio_threshold",
            0.03, 0.08, 0.005,
            self.live_config.eyes_closed_ratio_threshold,
            tooltip="Lower = more sensitive (eyes close easier)"
        ))

        # Eye frames
        layout.addWidget(self._create_slider_group(
            "Eye Closed Frame Delay",
            "eyes_closed_consecutive_frames",
            1, 10, 1,
            self.live_config.eyes_closed_consecutive_frames,
            tooltip="Number of frames before eyes register as closed"
        ))

        layout.addStretch()

        # Info box
        info = QLabel(
            "💡 Tip: If mouth/eyes are too sensitive or not sensitive enough, "
            "adjust these settings until comfortable."
        )
        info.setWordWrap(True)
        info.setStyleSheet("""
            color: #aaaaaa;
            font-size: 11px;
            padding: 10px;
            background-color: #1a1a1a;
            border-radius: 5px;
        """)
        layout.addWidget(info)

        return widget

    def _create_advanced_tab(self):
        """Create the advanced settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        # Pose detection confidence
        layout.addWidget(self._create_double_slider_group(
            "Pose Detection Confidence",
            "pose_min_detection_confidence",
            0.1, 1.0, 0.1,
            self.live_config.pose_min_detection_confidence,
            tooltip="Minimum confidence for pose detection"
        ))

        # Pose tracking confidence
        layout.addWidget(self._create_double_slider_group(
            "Pose Tracking Confidence",
            "pose_min_tracking_confidence",
            0.1, 1.0, 0.1,
            self.live_config.pose_min_tracking_confidence,
            tooltip="Minimum confidence for pose tracking"
        ))

        # Model complexity
        complexity_group = QGroupBox("Model Complexity")
        complexity_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                border: 1px solid #404040;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        complexity_layout = QHBoxLayout(complexity_group)

        complexity_label = QLabel("Mode:")
        complexity_label.setStyleSheet("color: #cccccc;")

        complexity_combo = QComboBox()
        complexity_combo.addItems(["Lite (Fast)", "Full (Balanced)", "Heavy (Accurate)"])
        complexity_combo.setCurrentIndex(self.live_config.pose_model_complexity)
        complexity_combo.currentIndexChanged.connect(
            lambda idx: self._on_config_changed("pose_model_complexity", idx)
        )
        complexity_combo.setStyleSheet("""
            QComboBox {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #555555;
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
                color: #ffffff;
                selection-background-color: #0d47a1;
            }
        """)

        complexity_layout.addWidget(complexity_label)
        complexity_layout.addWidget(complexity_combo)
        complexity_layout.addStretch()

        layout.addWidget(complexity_group)

        # Neck length
        layout.addWidget(self._create_double_slider_group(
            "Neck Length",
            "neck_length_ratio",
            0.2, 1.0, 0.1,
            self.live_config.neck_length_ratio,
            tooltip="Length of the neck"
        ))

        # Shoulder curve
        layout.addWidget(self._create_double_slider_group(
            "Shoulder Curve Depth",
            "shoulder_curve_depth_ratio",
            0.0, 0.3, 0.05,
            self.live_config.shoulder_curve_depth_ratio,
            tooltip="How curved the shoulders are"
        ))

        # Virtual camera mirror
        mirror_group = QGroupBox("Virtual Camera Options")
        mirror_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                border: 1px solid #404040;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        mirror_layout = QVBoxLayout(mirror_group)

        mirror_checkbox = QCheckBox("Mirror output horizontally")
        mirror_checkbox.setChecked(self.live_config.vcam_mirror_output)
        mirror_checkbox.setStyleSheet("""
            QCheckBox {
                color: #cccccc;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        mirror_checkbox.setToolTip(
            "When enabled, the output is mirrored horizontally.\n"
            "Enable this if text/logos appear backwards in video calls."
        )
        mirror_checkbox.stateChanged.connect(
            lambda state: self._on_config_changed(
                "vcam_mirror_output",
                state == Qt.CheckState.Checked.value
            )
        )

        mirror_layout.addWidget(mirror_checkbox)

        # Info
        info = QLabel(
            "Note: Different video apps handle mirroring differently. "
            "Toggle this if your output appears backwards to other people."
        )
        info.setWordWrap(True)
        info.setStyleSheet("""
            color: #aaaaaa;
            font-size: 10px;
            padding: 5px;
        """)
        mirror_layout.addWidget(info)

        layout.addWidget(mirror_group)

        layout.addStretch()
        return widget

    def _create_slider_group(self, label, config_key, min_val, max_val,
                             step, initial_value, tooltip=None):
        """
        Create a group with slider for integer values.

        Args:
            label: Group label
            config_key: Configuration key
            min_val: Minimum value
            max_val: Maximum value
            step: Step value
            initial_value: Initial value
            tooltip: Tooltip text

        Returns:
            QGroupBox: Configured group widget
        """
        group = QGroupBox(label)
        if tooltip:
            group.setToolTip(tooltip)

        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                border: 1px solid #404040;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        layout = QHBoxLayout(group)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setSingleStep(step)
        slider.setValue(initial_value)
        slider.setStyleSheet("""
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
        """)

        value_label = QLabel(str(initial_value))
        value_label.setMinimumWidth(40)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_label.setStyleSheet("color: #ffffff; font-weight: bold;")

        slider.valueChanged.connect(
            lambda v: (
                value_label.setText(str(v)),
                self._on_config_changed(config_key, v)
            )
        )

        layout.addWidget(slider)
        layout.addWidget(value_label)

        return group

    def _create_double_slider_group(self, label, config_key, min_val, max_val,
                                    step, initial_value, tooltip=None):
        """
        Create a group with slider for floating-point values.

        Args:
            label: Group label
            config_key: Configuration key
            min_val: Minimum value
            max_val: Maximum value
            step: Step value
            initial_value: Initial value
            tooltip: Tooltip text

        Returns:
            QGroupBox: Configured group widget
        """
        group = QGroupBox(label)
        if tooltip:
            group.setToolTip(tooltip)

        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                border: 1px solid #404040;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        layout = QHBoxLayout(group)

        # Convert to integer range
        multiplier = int(1 / step)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(min_val * multiplier))
        slider.setMaximum(int(max_val * multiplier))
        slider.setSingleStep(1)
        slider.setValue(int(initial_value * multiplier))
        slider.setStyleSheet("""
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
        """)

        value_label = QLabel(f"{initial_value:.3f}")
        value_label.setMinimumWidth(50)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_label.setStyleSheet("color: #ffffff; font-weight: bold;")

        def on_change(v):
            real_value = v / multiplier
            value_label.setText(f"{real_value:.3f}")
            self._on_config_changed(config_key, real_value)

        slider.valueChanged.connect(on_change)

        layout.addWidget(slider)
        layout.addWidget(value_label)

        return group

    def _on_config_changed(self, key, value):
        """
        Handle configuration change.

        Args:
            key: Configuration key
            value: New value
        """
        self.live_config.update(**{key: value})
        self.config_changed.emit({key: value})

    def _reset_to_defaults(self):
        """Reset all settings to default values."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to default values?\n\n"
            "You may need to restart the camera for all changes to take effect.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.live_config.reset_to_defaults()
            QMessageBox.information(
                self,
                "Reset Complete",
                "Settings have been reset to defaults.\n"
                "Restart the camera if it's currently running."
            )
