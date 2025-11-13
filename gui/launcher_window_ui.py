"""
UI manager for the main launcher window.

Manages layout, widgets and aesthetic appearance of the main window.
"""

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QGroupBox, QScrollArea,
    QSizePolicy, QFrame
)

from gui.settings_tabs import SettingsTabs


class LauncherSignals(QObject):
    """Container for LauncherWindowUI signals."""
    start_camera_clicked = pyqtSignal()
    stop_camera_clicked = pyqtSignal()
    minimize_to_tray_changed = pyqtSignal(bool)
    config_changed = pyqtSignal(dict)
    preview_toggled = pyqtSignal(bool)


class PreviewWidget(QLabel):
    """Widget for displaying camera preview."""

    def __init__(self, parent=None):
        """
        Initialize preview widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setMaximumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            background-color: #1a1a1a;
            border: 2px solid #404040;
            border-radius: 8px;
        """)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # Default text
        self.setText("Preview disabled\n(Enable below to see output)")
        self.setStyleSheet(self.styleSheet() + "color: #808080; font-size: 14px;")


class LauncherWindowUI:
    """
    UI Manager for the main launcher window.

    Creates an aesthetic interface with buttons, preview and settings.
    """

    def __init__(self, main_window, live_config):
        """
        Initialize UI manager.

        Args:
            main_window: LauncherWindow instance
            live_config: LiveConfig instance
        """
        self.main_window = main_window
        self.live_config = live_config

        # Signals
        self.signals = LauncherSignals()

        # Widgets
        self.preview_widget = None
        self.start_button = None
        self.stop_button = None
        self.status_label = None
        self.fps_label = None
        self.vcam_status_label = None
        self.preview_checkbox = None
        self.minimize_checkbox = None
        self.settings_tabs = None

        print("[LauncherWindowUI] Initialized")

    def setup_ui(self):
        """Set up the entire user interface."""
        self.main_window.setWindowTitle("Stickfigure Webcam - Settings & Control")
        self.main_window.setMinimumSize(1000, 700)

        # Central widget
        central = QWidget()
        self.main_window.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Left side: Preview and controls
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 2)

        # Right side: Settings
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)

        # Set initial button state
        self.set_camera_running(False)

        print("[LauncherWindowUI] UI setup complete")

    def _create_left_panel(self):
        """Create left panel with preview and controls."""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 10px;
                padding: 15px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # Header
        header = self._create_header()
        layout.addWidget(header)

        # Preview
        preview_group = self._create_preview_group()
        layout.addWidget(preview_group, 1)

        # Status
        status_group = self._create_status_group()
        layout.addWidget(status_group)

        # Control buttons
        controls = self._create_control_buttons()
        layout.addWidget(controls)

        # Options
        options = self._create_options()
        layout.addWidget(options)

        return panel

    def _create_header(self):
        """Create header with title."""
        header = QLabel("Stickfigure Webcam")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        header.setFont(font)

        header.setStyleSheet("color: #ffffff; padding: 10px;")

        return header

    def _create_preview_group(self):
        """Create preview group."""
        group = QGroupBox("Output Preview")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)

        layout = QVBoxLayout(group)

        # Preview widget
        self.preview_widget = PreviewWidget()
        layout.addWidget(self.preview_widget)

        # Preview checkbox
        self.preview_checkbox = QCheckBox("Enable live preview (uses more CPU)")
        self.preview_checkbox.setStyleSheet("color: #cccccc; padding: 5px;")
        self.preview_checkbox.stateChanged.connect(self._on_preview_toggled)
        layout.addWidget(self.preview_checkbox)

        return group

    def _create_status_group(self):
        """Create status group."""
        group = QGroupBox("Status")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #ffffff;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)

        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Camera status
        self.status_label = QLabel("● Camera: Not Started")
        self.status_label.setStyleSheet("color: #ffaa00; font-size: 13px; padding: 3px;")
        layout.addWidget(self.status_label)

        # Virtual camera status
        self.vcam_status_label = QLabel("○ Virtual Camera: Not Started")
        self.vcam_status_label.setStyleSheet("color: #808080; font-size: 13px; padding: 3px;")
        layout.addWidget(self.vcam_status_label)

        # FPS
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #00ff00; font-size: 13px; padding: 3px;")
        layout.addWidget(self.fps_label)

        return group

    def _create_control_buttons(self):
        """Create control buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(10)

        # Start button
        self.start_button = QPushButton("▶ Start Virtual Camera")
        self.start_button.setMinimumHeight(50)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QPushButton:pressed {
                background-color: #1b5e20;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        self.start_button.clicked.connect(self.signals.start_camera_clicked)
        layout.addWidget(self.start_button)

        # Stop button
        self.stop_button = QPushButton("■ Stop Virtual Camera")
        self.stop_button.setMinimumHeight(50)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #c62828;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        self.stop_button.clicked.connect(self.signals.stop_camera_clicked)
        layout.addWidget(self.stop_button)

        return widget

    def _create_options(self):
        """Create options."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)

        # Minimize to tray
        self.minimize_checkbox = QCheckBox(
            "Minimize to system tray when closing this window"
        )
        self.minimize_checkbox.setChecked(True)
        self.minimize_checkbox.setStyleSheet("""
            QCheckBox {
                color: #cccccc;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.minimize_checkbox.setToolTip(
            "When enabled, closing this window minimizes to tray.\n"
            "The virtual camera keeps running in the background.\n"
            "Use the tray icon to show settings or exit."
        )
        self.minimize_checkbox.stateChanged.connect(self._on_minimize_changed)
        layout.addWidget(self.minimize_checkbox)

        # Info text
        info = QLabel(
            "💡 Tip: After starting, you can minimize this window.\n"
            "The virtual camera will keep running in the background."
        )
        info.setStyleSheet("""
            color: #aaaaaa;
            font-size: 11px;
            padding: 5px;
            background-color: #1a1a1a;
            border-radius: 5px;
        """)
        info.setWordWrap(True)
        layout.addWidget(info)

        return widget

    def _create_right_panel(self):
        """Create right panel with settings."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(350)
        scroll.setMaximumWidth(450)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
                border-radius: 10px;
            }
        """)

        # Settings tabs
        self.settings_tabs = SettingsTabs(self.live_config)
        self.settings_tabs.config_changed.connect(self._on_config_changed)

        scroll.setWidget(self.settings_tabs)

        return scroll

    # UI update methods

    def set_camera_running(self, running: bool):
        """
        Set button states based on camera state.

        Args:
            running: Whether camera is running
        """
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def update_status(self, text: str, color: str = "#ffaa00"):
        """
        Update camera status.

        Args:
            text: Status text
            color: Status color
        """
        if self.status_label:
            icon = "●" if "Active" in text else "○"
            self.status_label.setText(f"{icon} Camera: {text}")
            self.status_label.setStyleSheet(
                f"color: {color}; font-size: 13px; font-weight: bold; padding: 3px;"
            )

    def update_vcam_status(self, text: str, color: str = "#ffaa00"):
        """
        Update virtual camera status.

        Args:
            text: Status text
            color: Status color
        """
        if self.vcam_status_label:
            icon = "●" if "Active" in text else "○"
            self.vcam_status_label.setText(f"{icon} {text}")
            self.vcam_status_label.setStyleSheet(
                f"color: {color}; font-size: 13px; font-weight: bold; padding: 3px;"
            )

    def update_fps(self, fps: float):
        """
        Update FPS display.

        Args:
            fps: Current FPS value
        """
        if self.fps_label:
            self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_preview(self, frame: np.ndarray):
        """
        Update camera preview.

        Args:
            frame: Camera frame to display
        """
        if not self.preview_widget:
            return

        # Convert to QPixmap
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_image.data,
            w, h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to widget size
        scaled_pixmap = pixmap.scaled(
            self.preview_widget.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.preview_widget.setPixmap(scaled_pixmap)

    def clear_preview(self):
        """Clear preview."""
        if self.preview_widget:
            self.preview_widget.clear()
            self.preview_widget.setText("Preview disabled")

    # Slot handlers

    def _on_config_changed(self, changes: dict):
        """
        Pass configuration change.

        Args:
            changes: Dictionary of changed configuration values
        """
        self.signals.config_changed.emit(changes)

    def _on_preview_toggled(self, state):
        """
        Handle preview checkbox change.

        Args:
            state: Checkbox state
        """
        enabled = (state == Qt.CheckState.Checked.value)
        self.signals.preview_toggled.emit(enabled)

        if not enabled:
            self.clear_preview()

    def _on_minimize_changed(self, state):
        """
        Handle minimize checkbox change.

        Args:
            state: Checkbox state
        """
        enabled = (state == Qt.CheckState.Checked.value)
        self.signals.minimize_to_tray_changed.emit(enabled)
