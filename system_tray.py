"""
System tray integration module.

This module provides system tray functionality for the Stickfigure Webcam
application, allowing it to run in the background with a tray icon.
"""

import threading
from typing import Callable, Optional

import pystray
from PIL import Image, ImageDraw
from pystray import MenuItem as Item


class SystemTray:
    """
    System tray manager for the application.

    Provides tray icon with context menu for controlling the application
    and showing/hiding windows.
    """

    def __init__(self, app_name: str = "Stickfigure Webcam"):
        """
        Initialize the system tray.

        Args:
            app_name: Application name to display in tray.
        """
        self.app_name = app_name
        self.icon: Optional[pystray.Icon] = None
        self.running = False
        self.tray_thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_show_callback: Optional[Callable] = None
        self.on_hide_callback: Optional[Callable] = None
        self.on_settings_callback: Optional[Callable] = None
        self.on_toggle_camera_callback: Optional[Callable] = None
        self.on_toggle_debug_callback: Optional[Callable] = None
        self.on_quit_callback: Optional[Callable] = None

        # State tracking
        self.camera_running = False
        self.debug_visible = False
        self.minimize_to_tray_enabled = False

        print("[SystemTray] Initialized")

    def create_icon_image(self) -> Image.Image:
        """
        Create the tray icon image.

        Returns:
            Image.Image: PIL Image for the tray icon.
        """
        # Create a simple stick figure icon
        width = 64
        height = 64
        image = Image.new('RGB', (width, height), 'black')
        draw = ImageDraw.Draw(image)

        # Draw stick figure
        color = 'white'

        # Head
        head_center = (width // 2, height // 4)
        head_radius = 8
        draw.ellipse(
            [head_center[0] - head_radius, head_center[1] - head_radius,
             head_center[0] + head_radius, head_center[1] + head_radius],
            outline=color, width=2
        )

        # Body
        body_top = head_center[1] + head_radius
        body_bottom = height * 3 // 4
        draw.line([width // 2, body_top, width // 2, body_bottom], fill=color, width=2)

        # Arms
        arm_y = height // 2
        draw.line([width // 4, arm_y, width * 3 // 4, arm_y], fill=color, width=2)

        # Legs
        draw.line([width // 2, body_bottom, width // 3, height - 5], fill=color, width=2)
        draw.line([width // 2, body_bottom, width * 2 // 3, height - 5], fill=color, width=2)

        return image

    def create_menu(self) -> pystray.Menu:
        """
        Create the context menu for the tray icon.

        Returns:
            pystray.Menu: Context menu with all options.
        """
        return pystray.Menu(
            Item(
                "Show/Hide",
                self._on_show_hide,
                default=True
            ),
            Item(
                "Start Camera" if not self.camera_running else "Stop Camera",
                self._on_toggle_camera
            ),
            Item(
                "Show Debug Window",
                self._on_toggle_debug,
                checked=lambda item: self.debug_visible
            ),
            pystray.Menu.SEPARATOR,
            Item(
                "Settings",
                self._on_settings
            ),
            Item(
                "Minimize to Tray on Close",
                self._on_toggle_minimize_to_tray,
                checked=lambda item: self.minimize_to_tray_enabled
            ),
            pystray.Menu.SEPARATOR,
            Item(
                "Exit",
                self._on_quit
            )
        )

    def start(self):
        """Start the system tray icon in a separate thread."""
        if self.running:
            print("[SystemTray] Already running")
            return

        self.running = True

        # Create icon
        self.icon = pystray.Icon(
            self.app_name,
            self.create_icon_image(),
            self.app_name,
            self.create_menu()
        )

        # Start in separate thread
        self.tray_thread = threading.Thread(
            target=self._run_tray,
            daemon=True,
            name="SystemTrayThread"
        )
        self.tray_thread.start()

        print("[SystemTray] Started")

    def _run_tray(self):
        """Run the tray icon (internal thread method)."""
        if self.icon:
            self.icon.run()

    def stop(self):
        """Stop the system tray icon."""
        if not self.running:
            return

        print("[SystemTray] Stopping...")
        self.running = False

        if self.icon:
            self.icon.stop()
            self.icon = None

        print("[SystemTray] Stopped")

    def update_menu(self):
        """Update the context menu (e.g., after state changes)."""
        if self.icon:
            self.icon.menu = self.create_menu()
            self.icon.update_menu()

    def show_notification(self, title: str, message: str):
        """
        Show a system notification.

        Args:
            title: Notification title.
            message: Notification message.
        """
        if self.icon and self.icon.HAS_NOTIFICATION:
            self.icon.notify(message, title)

    def set_camera_state(self, running: bool):
        """
        Update camera running state.

        Args:
            running: Whether camera is running.
        """
        self.camera_running = running
        self.update_menu()

    def set_debug_visible(self, visible: bool):
        """
        Update debug window visibility state.

        Args:
            visible: Whether debug window is visible.
        """
        self.debug_visible = visible
        self.update_menu()

    def set_minimize_to_tray(self, enabled: bool):
        """
        Update minimize to tray setting.

        Args:
            enabled: Whether minimize to tray is enabled.
        """
        self.minimize_to_tray_enabled = enabled
        self.update_menu()

    # Callback setters
    def set_on_show(self, callback: Callable):
        """Set callback for showing main window."""
        self.on_show_callback = callback

    def set_on_hide(self, callback: Callable):
        """Set callback for hiding main window."""
        self.on_hide_callback = callback

    def set_on_settings(self, callback: Callable):
        """Set callback for opening settings."""
        self.on_settings_callback = callback

    def set_on_toggle_camera(self, callback: Callable):
        """Set callback for toggling camera."""
        self.on_toggle_camera_callback = callback

    def set_on_toggle_debug(self, callback: Callable):
        """Set callback for toggling debug window."""
        self.on_toggle_debug_callback = callback

    def set_on_quit(self, callback: Callable):
        """Set callback for quitting application."""
        self.on_quit_callback = callback

    # Internal menu action handlers
    def _on_show_hide(self, icon, item):
        """Handle show/hide menu item."""
        if self.on_show_callback:
            self.on_show_callback()

    def _on_toggle_camera(self, icon, item):
        """Handle toggle camera menu item."""
        if self.on_toggle_camera_callback:
            self.on_toggle_camera_callback()

    def _on_toggle_debug(self, icon, item):
        """Handle toggle debug window menu item."""
        if self.on_toggle_debug_callback:
            self.on_toggle_debug_callback()

    def _on_settings(self, icon, item):
        """Handle settings menu item."""
        if self.on_settings_callback:
            self.on_settings_callback()

    def _on_toggle_minimize_to_tray(self, icon, item):
        """Handle toggle minimize to tray menu item."""
        self.minimize_to_tray_enabled = not self.minimize_to_tray_enabled
        self.update_menu()

        message = "enabled" if self.minimize_to_tray_enabled else "disabled"
        print(f"[SystemTray] Minimize to tray {message}")
        self.show_notification(
            "Settings Changed",
            f"Minimize to tray {message}"
        )

    def _on_quit(self, icon, item):
        """Handle quit menu item."""
        if self.on_quit_callback:
            self.on_quit_callback()


def test_system_tray():
    """Test function for system tray."""
    import time

    print("Testing SystemTray...")

    tray = SystemTray("Test App")

    # Set up callbacks
    tray.set_on_show(lambda: print("Show callback"))
    tray.set_on_quit(lambda: tray.stop())

    # Start tray
    tray.start()

    print("Tray running... Right-click the icon to test menu.")
    print("Press Ctrl+C to exit.")

    try:
        # Keep running
        while tray.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tray.stop()


if __name__ == "__main__":
    test_system_tray()
