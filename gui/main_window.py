"""
Main window for GUI application.

This module provides the main application window with video preview,
status indicators, control panel integration, system tray support,
and debug window.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QMainWindow

from gui.config_manager import LiveConfig
from gui.main_window_camera import CameraManager
from gui.main_window_rendering import RenderingManager
from gui.main_window_system_tray import SystemTrayManager
from gui.main_window_ui import MainWindowUI


class MainWindow(QMainWindow):
    """
    Main application window.

    Provides the primary GUI with video preview, control panel,
    status indicators, system tray integration, and debug window
    for the stickfigure webcam application.
    """

    # Signal to safely close the application from another thread
    safe_close_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Core state
        self.live_config = LiveConfig()
        self.is_quitting = False
        self.minimize_to_tray = True

        # Component managers
        self.ui_manager = None
        self.camera_manager = None
        self.rendering_manager = None
        self.tray_manager = None

        # Initialize components
        self._init_components()

        # Connect the safe close signal
        self.safe_close_signal.connect(self.close)

    def _init_components(self):
        """Initialize all component managers."""
        # UI Manager
        self.ui_manager = MainWindowUI(self, self.live_config)
        self.ui_manager.setup_ui()

        # Camera Manager
        self.camera_manager = CameraManager(
            self,
            self.live_config,
            self.ui_manager
        )

        # Rendering Manager
        self.rendering_manager = RenderingManager(
            self.live_config,
            self.ui_manager
        )

        # System Tray Manager
        self.tray_manager = SystemTrayManager(self, self.camera_manager)
        self.tray_manager.initialize()

        # Connect signals
        self._connect_signals()

        # Start camera
        self.camera_manager.start_camera()

    def _connect_signals(self):
        """Connect all inter-component signals."""
        # Camera signals
        self.camera_manager.frame_ready.connect(
            self.rendering_manager.on_frame_ready
        )
        self.camera_manager.fps_updated.connect(
            self.ui_manager.update_fps
        )

        # UI signals
        ui_signals = getattr(self.ui_manager, "signals", None)
        if ui_signals is None:
            print("[MainWindow] Error: 'MainWindowUI' does not expose QObject 'signals' — skipping UI connections.\n"
                  "TODO: implement this sometime.")
        else:
            ui_signals.config_changed.connect(self.rendering_manager.on_config_changed)
            ui_signals.mirror_changed.connect(self.rendering_manager.on_mirror_changed)
            ui_signals.minimize_to_tray_changed.connect(self._on_minimize_to_tray_changed)
            ui_signals.camera_button_clicked.connect(self.camera_manager.toggle_camera)
            ui_signals.debug_window_toggled.connect(self._on_debug_window_toggled)

    def _on_minimize_to_tray_changed(self, enabled: bool):
        """Handle minimize to tray setting change."""
        self.minimize_to_tray = enabled
        print(f"[MainWindow] Minimize to tray: {'enabled' if enabled else 'disabled'}")

    def _on_debug_window_toggled(self, show: bool):
        """Handle debug window toggle."""
        if show:
            self.ui_manager.show_debug_window(
                self.live_config,
                self.camera_manager
            )
        else:
            self.ui_manager.hide_debug_window()

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event."""
        # If minimize to tray is enabled and not explicitly quitting
        if self.minimize_to_tray and not self.is_quitting:
            print("[MainWindow] Minimizing to system tray...")
            self.hide()
            if self.tray_manager:
                self.tray_manager.show_notification(
                    "Stickfigure Webcam",
                    "Application minimized to system tray"
                )
            event.ignore()
            return

        # Actually quitting
        print("[MainWindow] Closing application...")
        self._cleanup()
        event.accept()

        # Force application to quit
        QApplication.instance().quit()

    def _cleanup(self):
        """Clean up all resources."""
        # Flush configuration
        self.live_config.flush_to_disk()

        # Close debug window
        if self.ui_manager:
            self.ui_manager.close_debug_window()

        # Stop camera
        if self.camera_manager:
            self.camera_manager.stop_camera()

        # Stop virtual camera
        if self.rendering_manager:
            self.rendering_manager.cleanup()

        # Stop system tray
        if self.tray_manager:
            self.tray_manager.stop()

        print("[MainWindow] Cleanup complete")
