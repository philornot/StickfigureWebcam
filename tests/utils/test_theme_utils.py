#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for theme utilities."""

import unittest
from unittest.mock import MagicMock, patch

from src.utils.theme_utils import apply_theme_to_tkinter, detect_system_theme, get_theme_colors


class TestThemeUtils(unittest.TestCase):
    """Tests for theme utility functions."""

    @patch("platform.system")
    def test_detect_system_theme_windows_dark(self, mock_system):
        """Test detecting dark theme on Windows."""
        mock_system.return_value = "Windows"

        with patch("winreg.OpenKey"), patch("winreg.QueryValueEx", return_value=(0, None)):
            theme = detect_system_theme()

            self.assertEqual(theme, "dark")

    @patch("platform.system")
    def test_detect_system_theme_windows_light(self, mock_system):
        """Test detecting light theme on Windows."""
        mock_system.return_value = "Windows"

        with patch("winreg.OpenKey"), patch("winreg.QueryValueEx", return_value=(1, None)):
            theme = detect_system_theme()

            self.assertEqual(theme, "light")

    @patch("platform.system")
    @patch("subprocess.run")
    def test_detect_system_theme_macos_dark(self, mock_run, mock_system):
        """Test detecting dark theme on macOS."""
        mock_system.return_value = "Darwin"
        mock_run.return_value.stdout = "Dark"

        theme = detect_system_theme()

        self.assertEqual(theme, "dark")

    @patch("platform.system")
    @patch("subprocess.run")
    def test_detect_system_theme_macos_light(self, mock_run, mock_system):
        """Test detecting light theme on macOS."""
        mock_system.return_value = "Darwin"
        mock_run.return_value.stdout = ""

        theme = detect_system_theme()

        self.assertEqual(theme, "light")

    @patch("platform.system")
    @patch("subprocess.run")
    def test_detect_system_theme_linux_dark(self, mock_run, mock_system):
        """Test detecting dark theme on Linux."""
        mock_system.return_value = "Linux"
        mock_run.return_value.stdout = "'prefer-dark'"

        theme = detect_system_theme()

        self.assertEqual(theme, "dark")

    def test_get_theme_colors_dark(self):
        """Test getting colors for dark theme."""
        colors = get_theme_colors("dark")

        text, highlight, bg, success, error = colors

        # Dark theme should have light text
        self.assertEqual(text, "#ffffff")
        # Dark background
        self.assertEqual(bg, "#1e1e1e")

    def test_get_theme_colors_light(self):
        """Test getting colors for light theme."""
        colors = get_theme_colors("light")

        text, highlight, bg, success, error = colors

        # Light theme should have dark text
        self.assertEqual(text, "#000000")
        # Light background
        self.assertEqual(bg, "#f0f0f0")

    def test_get_theme_colors_auto(self):
        """Test getting colors with auto-detect."""
        colors = get_theme_colors(None)

        # Should return tuple of 5 colors
        self.assertEqual(len(colors), 5)

    @patch("src.utils.theme_utils.detect_system_theme")
    def test_apply_theme_to_tkinter_success(self, mock_detect):
        """Test applying theme to Tkinter with sv_ttk."""
        mock_detect.return_value = "dark"
        mock_root = MagicMock()

        with patch("sv_ttk.set_theme") as mock_set_theme:
            result = apply_theme_to_tkinter(mock_root, "dark")

            self.assertTrue(result)
            mock_set_theme.assert_called_once_with("dark")

    @patch("src.utils.theme_utils.detect_system_theme")
    def test_apply_theme_to_tkinter_fallback(self, mock_detect):
        """Test applying theme falls back when sv_ttk unavailable."""
        mock_detect.return_value = "dark"
        mock_root = MagicMock()

        with patch("sv_ttk.set_theme", side_effect=ImportError):
            result = apply_theme_to_tkinter(mock_root, "dark")

            # Should still succeed with fallback
            self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
