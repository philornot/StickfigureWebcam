"""
UI management for main window.

This module handles all UI setup and widget management for the main window.
"""

from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QScrollArea, QSizePolicy, QPushButton
)

from gui.control_panel import ControlPanel
from gui.debug_window import DebugWindow


class StickfigureWidget(QLabel):
    """Widget for displaying the stickfigure output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black; border: 2px solid #555;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class MainWindowSignals(QObject):
    """QObject container for MainWindowUI signals."""

    config_changed = pyqtSignal(dict)
    mirror_changed = pyqtSignal(bool)
    minimize_to_tray_changed = pyqtSignal(bool)
    camera_button_clicked = pyqtSignal()
    debug_window_toggled = pyqtSignal(bool)


class MainWindowUI:
    """
    UI manager for main window.

    Handles all UI setup, widget creation, and UI updates.
    """

    def __init__(self, main_window, live_config):
        """
        Initialize UI manager.

        Args:
            main_window: QMainWindow instance
            live_config: LiveConfig instance
        """
        self.main_window = main_window
        self.live_config = live_config

        # Signals container
        self.signals = MainWindowSignals()

        # Widgets
        self.video_widget = None
        self.fps_label = None
        self.status_label = None
        self.vcam_status_label = None
        self.vcam_mirror_checkbox = None
        self.camera_button = None
        self.debug_window_checkbox = None
        self.minimize_to_tray_checkbox = None
        self.control_panel = None

        # Debug window
        self.debug_window = None

        print("[MainWindowUI] Initialized")

    def setup_ui(self):
        """Set up all UI components."""
        self.main_window.setWindowTitle("Stickfigure Webcam - Control Panel")
        self.main_window.setMinimumSize(1200, 800)

        # Central widget
        central = QWidget()
        self.main_window.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left side: Video preview and controls
        left_layout = self._create_left_panel()
        main_layout.addLayout(left_layout, 2)

        # Right side: Control panel
        scroll = self._create_control_panel()
        main_layout.addWidget(scroll, 1)

        print("[MainWindowUI] UI setup complete")

    def _create_left_panel(self):
        """Create the left panel with video preview and controls."""
        layout = QVBoxLayout()

        # Preview label
        preview_label = QLabel("Stickfigure Output")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        preview_label.setFont(font)
        layout.addWidget(preview_label)

        # Video widget
        self.video_widget = StickfigureWidget()
        layout.addWidget(self.video_widget)

        # Status bar
        layout.addLayout(self._create_status_bar())

        # Virtual camera controls
        layout.addLayout(self._create_vcam_controls())

        # Camera and debug controls
        layout.addLayout(self._create_camera_controls())

        # System tray controls
        layout.addLayout(self._create_tray_controls())

        return layout

    def _create_status_bar(self):
        """Create status bar layout."""
        layout = QHBoxLayout()

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #00ff00; font-weight: bold;")

        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setStyleSheet("color: #ffff00;")

        layout.addWidget(self.fps_label)
        layout.addStretch()
        layout.addWidget(self.status_label)

        return layout

    def _create_vcam_controls(self):
        """Create virtual camera controls layout."""
        layout = QHBoxLayout()

        self.vcam_status_label = QLabel("Virtual Camera: Not Started")

        self.vcam_mirror_checkbox = QCheckBox("Mirror Output")
        self.vcam_mirror_checkbox.setChecked(self.live_config.vcam_mirror_output)
        self.vcam_mirror_checkbox.stateChanged.connect(self._on_mirror_changed)

        layout.addWidget(self.vcam_status_label)
        layout.addStretch()
        layout.addWidget(self.vcam_mirror_checkbox)

        return layout

    def _create_camera_controls(self):
        """Create camera and debug controls layout."""
        layout = QHBoxLayout()

        self.camera_button = QPushButton("Stop Camera")
        self.camera_button.clicked.connect(self._on_camera_button_clicked)
        layout.addWidget(self.camera_button)

        self.debug_window_checkbox = QCheckBox("Show Debug Window")
        self.debug_window_checkbox.stateChanged.connect(self._on_debug_window_toggled)
        layout.addWidget(self.debug_window_checkbox)

        layout.addStretch()
        return layout

    def _create_tray_controls(self):
        """Create system tray controls layout."""
        layout = QHBoxLayout()

        self.minimize_to_tray_checkbox = QCheckBox("Minimize to system tray on close")
        self.minimize_to_tray_checkbox.setChecked(True)
        self.minimize_to_tray_checkbox.setToolTip(
            "When enabled, closing the window minimizes the app to system tray.\n"
            "When disabled, closing the window exits the application."
        )
        self.minimize_to_tray_checkbox.stateChanged.connect(
            self._on_minimize_to_tray_changed
        )

        layout.addWidget(self.minimize_to_tray_checkbox)
        layout.addStretch()

        return layout

    def _create_control_panel(self):
        """Create control panel in scroll area."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(500)

        self.control_panel = ControlPanel(self.live_config)
        self.control_panel.config_changed.connect(self._on_config_changed)
        scroll.setWidget(self.control_panel)

        return scroll

    # Signal handlers
    def _on_mirror_changed(self, state):
        """Handle mirror checkbox change."""
        mirror = (state == Qt.CheckState.Checked.value)
        self.live_config.update(vcam_mirror_output=mirror)
        self.signals.mirror_changed.emit(mirror)

    def _on_minimize_to_tray_changed(self, state):
        """Handle minimize to tray checkbox change."""
        enabled = (state == Qt.CheckState.Checked.value)
        self.signals.minimize_to_tray_changed.emit(enabled)

    def _on_camera_button_clicked(self):
        """Handle camera button click."""
        self.signals.camera_button_clicked.emit()

    def _on_debug_window_toggled(self, state):
        """Handle debug window checkbox toggle."""
        show = (state == Qt.CheckState.Checked.value)
        self.signals.debug_window_toggled.emit(show)

    def _on_config_changed(self, changes):
        """Handle configuration change."""
        self.signals.config_changed.emit(changes)

    # Public methods for updating UI
    def update_status(self, text, color="#ffff00"):
        """Update status label."""
        if self.status_label:
            self.status_label.setText(text)
            self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def update_fps(self, fps):
        """Update FPS label."""
        if self.fps_label:
            self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_vcam_status(self, text, color="#ffff00"):
        """Update virtual camera status."""
        if self.vcam_status_label:
            self.vcam_status_label.setText(text)
            self.vcam_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def update_camera_button(self, text):
        """Update camera button text."""
        if self.camera_button:
            self.camera_button.setText(text)

    # Debug window management
    def show_debug_window(self, live_config, camera_manager):
        """Show the debug window."""
        if not self.debug_window:
            self.debug_window = DebugWindow(self.main_window)
            self.debug_window.set_live_config(live_config)
            self.debug_window.window_closed.connect(self._on_debug_window_closed)

        self.debug_window.show()
        self.debug_window.activateWindow()
        print("[MainWindowUI] Debug window shown")

    def hide_debug_window(self):
        """Hide the debug window."""
        if self.debug_window:
            self.debug_window.hide()
        print("[MainWindowUI] Debug window hidden")

    def close_debug_window(self):
        """Close and destroy debug window."""
        if self.debug_window:
            self.debug_window.close()
            self.debug_window = None

    def _on_debug_window_closed(self):
        """Handle debug window being closed by user."""
        print("[MainWindowUI] Debug window closed by user")
        if self.debug_window_checkbox:
            self.debug_window_checkbox.setChecked(False)

    def get_debug_window(self):
        """Get debug window instance."""
        return self.debug_window
