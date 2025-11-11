"""
System tray management for main window.

This module handles system tray integration and tray-related callbacks.
"""

from PyQt6.QtCore import QObject

from system_tray import SystemTray


class SystemTrayManager(QObject):
    """
    System tray manager for main window.

    Handles system tray icon and menu interactions.
    """

    def __init__(self, main_window, camera_manager):
        """
        Initialize system tray manager.

        Args:
            main_window: Main window instance
            camera_manager: CameraManager instance
        """
        super().__init__(main_window)

        self.main_window = main_window
        self.camera_manager = camera_manager
        self.system_tray = None

        print("[SystemTrayManager] Initialized")

    def initialize(self):
        """Initialize and start system tray."""
        self.system_tray = SystemTray("Stickfigure Webcam")

        # Set callbacks
        self.system_tray.set_on_toggle_camera(self._on_toggle_camera)
        self.system_tray.set_on_show_settings(self._on_show_settings)
        self.system_tray.set_on_quit(self._on_quit)

        # Start tray
        self.system_tray.start()

        # Connect camera state changes
        self.camera_manager.camera_state_changed.connect(
            self._on_camera_state_changed
        )

        # Set initial camera state
        self.system_tray.set_camera_state(self.camera_manager.is_running())

        print("[SystemTrayManager] System tray started")

    def _on_toggle_camera(self):
        """Handle toggle camera from tray."""
        if self.camera_manager.is_running():
            self.camera_manager.stop_camera()
            self.show_notification(
                "Camera Stopped",
                "Camera has been stopped"
            )
        else:
            self.camera_manager.start_camera()
            self.show_notification(
                "Camera Started",
                "Camera has been started"
            )

    def _on_show_settings(self):
        """Handle show settings from tray."""
        self.main_window.show()
        self.main_window.activateWindow()

    def _on_quit(self):
        """Handle quit from tray."""
        self.main_window.is_quitting = True
        self.main_window.safe_close_signal.emit()

    def _on_camera_state_changed(self, running):
        """Handle camera state change."""
        if self.system_tray:
            self.system_tray.set_camera_state(running)

    def show_notification(self, title, message):
        """Show system notification."""
        if self.system_tray:
            self.system_tray.show_notification(title, message)

    def stop(self):
        """Stop system tray."""
        if self.system_tray:
            self.system_tray.stop()
        print("[SystemTrayManager] Stopped")
