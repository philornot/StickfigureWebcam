"""
GUI package for Stickfigure Webcam.

This package contains all GUI-related components including the launcher window,
control panel, camera thread, debug window, and configuration management.
"""

from gui.config_manager import LiveConfig
from gui.camera_thread import CameraThread
from gui.control_panel import ControlPanel
from gui.debug_window import DebugWindow
from gui.main_window import MainWindow
from gui.main_window_ui import MainWindowUI
from gui.main_window_camera import CameraManager
from gui.main_window_rendering import RenderingManager
from gui.main_window_system_tray import SystemTrayManager
from gui.launcher_window import LauncherWindow  # NOWY
from gui.launcher_window_ui import LauncherWindowUI  # NOWY
from gui.settings_tabs import SettingsTabs  # NOWY
from gui.app import run_gui, create_app

__all__ = [
    'LiveConfig',
    'CameraThread',
    'ControlPanel',
    'DebugWindow',
    'MainWindow',
    'MainWindowUI',
    'CameraManager',
    'RenderingManager',
    'SystemTrayManager',
    'LauncherWindow',
    'LauncherWindowUI',
    'SettingsTabs',
    'run_gui',
    'create_app',
]