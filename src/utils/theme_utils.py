# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/utils/theme_utils.py

import ctypes
import os
import platform
import subprocess
from typing import Optional, Tuple


def detect_system_theme() -> str:
    """
    Detects system theme (light/dark).

    Returns:
        str: "dark" or "light"
    """
    system = platform.system()

    try:
        # Windows
        if system == "Windows":
            # Use Windows API to check theme
            try:
                # Using ctypes on Windows 10+
                HKEY_CURRENT_USER = -2147483647
                SUBKEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"

                try:
                    import winreg

                    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, SUBKEY)
                    value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                    winreg.CloseKey(key)
                    return "light" if value == 1 else "dark"
                except:
                    # If registry cannot be used, use default window color
                    return "dark" if ctypes.windll.user32.GetSysColor(13) < 127 else "light"
            except:
                return "light"  # Default light theme on older Windows versions

        # macOS
        elif system == "Darwin":
            try:
                # Use Apple system command
                result = subprocess.run(
                    ["defaults", "read", "-g", "AppleInterfaceStyle"],
                    capture_output=True,
                    text=True,
                )
                # If result is "Dark", user has dark theme enabled
                # If command returns error, user has light theme
                return "dark" if "Dark" in result.stdout else "light"
            except:
                return "light"  # Default light theme

        # Linux
        elif system == "Linux":
            try:
                # Check GNOME
                result = subprocess.run(
                    ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                    capture_output=True,
                    text=True,
                )
                if "dark" in result.stdout.lower():
                    return "dark"
                return "light"
            except:
                # Check GTK environment variable
                gtk_theme = os.environ.get("GTK_THEME", "").lower()
                if "dark" in gtk_theme:
                    return "dark"
                return "light"  # Default light theme

        else:
            return "light"  # Default light theme

    except Exception:
        return "light"  # In case of error, assume light theme


def get_theme_colors(theme: str = None) -> Tuple[str, str, str, str, str]:
    """
    Returns colors appropriate for theme.

    Args:
        theme (str, optional): "dark", "light" or None to auto-detect

    Returns:
        Tuple[str, str, str, str, str]:
            - Primary text color
            - Highlight color
            - Background color
            - Success color (green)
            - Error color (red)
    """
    if theme is None:
        theme = detect_system_theme()

    if theme == "dark":
        # Dark theme - with lighter colors for better contrast
        return (
            "#ffffff",  # Text: white
            "#1890ff",  # Highlight: light blue
            "#1e1e1e",  # Background: dark gray
            "#80ff80",  # Success: bright green
            "#ff8080",  # Error: bright red
        )
    else:
        # Light theme
        return (
            "#000000",  # Text: black
            "#0366d6",  # Highlight: dark blue
            "#f0f0f0",  # Background: light gray
            "#2b8a3e",  # Success: dark green
            "#e03131",  # Error: dark red
        )


def apply_theme_to_tkinter(root, theme: Optional[str] = None) -> bool:
    """
    Applies theme to Tkinter window.

    Args:
        root: Main Tkinter window
        theme (str, optional): "dark", "light" or None to auto-detect

    Returns:
        bool: True if theme applied, False otherwise
    """
    if theme is None:
        theme = detect_system_theme()

    try:
        # Try to use Sun Valley Tkinter Theme (sv_ttk)
        import sv_ttk

        sv_ttk.set_theme(theme)
        return True
    except ImportError:
        # If sv_ttk is not available, apply basic styles
        try:
            from tkinter import ttk

            style = ttk.Style()

            text_color, accent_color, bg_color, success_color, error_color = get_theme_colors(theme)

            if theme == "dark":
                # Dark theme - attempt to adjust standard styles
                style.configure("TFrame", background=bg_color)
                style.configure("TLabel", foreground=text_color, background=bg_color)
                style.configure("TButton", foreground=text_color, background=accent_color)

                # Adjust special colors
                style.configure("Success.TLabel", foreground=success_color)
                style.configure("Error.TLabel", foreground=error_color)

                # Set main window colors
                root.configure(background=bg_color)
            else:
                # Light theme - default styles
                pass

            return True
        except Exception:
            return False  # Failed to apply theme
