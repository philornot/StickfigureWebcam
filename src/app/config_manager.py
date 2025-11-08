#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration manager for application settings."""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.custom_logger import CustomLogger


class ConfigurationManager:
    """Manages application configuration with validation."""

    DEFAULT_CONFIG = {
        "camera": {
            "id": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "flip_horizontal": True
        },
        "processing": {
            "line_thickness": 3,
            "head_radius_factor": 0.075,
            "bg_color": [255, 255, 255],
            "figure_color": [0, 0, 0],
            "smooth_factor": 0.3
        },
        "app": {
            "debug": False,
            "show_preview": True,
            "show_landmarks": False
        }
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        logger: Optional[CustomLogger] = None
    ):
        """Initialize configuration manager.

        Args:
            config_path: Path to config file (default: settings.json)
            logger: Optional custom logger
        """
        self.logger = logger or CustomLogger()

        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "settings.json"

        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.logger.info("Config", f"Configuration loaded from {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults.

        Returns:
            Configuration dictionary
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)

                # Merge with defaults
                config = self._deep_merge(
                    self.DEFAULT_CONFIG.copy(),
                    loaded_config
                )

                self.logger.debug("Config", "Configuration loaded from file")
                return config

            except Exception as e:
                self.logger.warning(
                    "Config",
                    f"Failed to load config: {e}, using defaults"
                )

        return self.DEFAULT_CONFIG.copy()

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Dictionary to merge into base

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "camera.width")
            default: Default value if key not found

        Returns:
            Configuration value

        Examples:
            >>> config.get("camera.width")
            640
            >>> config.get("app.debug", False)
            False
        """
        parts = key.split('.')
        value = self.config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "camera.width")
            value: Value to set

        Examples:
            >>> config.set("camera.width", 1280)
            >>> config.set("app.debug", True)
        """
        parts = key.split('.')
        target = self.config

        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value
        self.logger.debug("Config", f"Set {key} = {value}")

    def save(self) -> bool:
        """Save configuration to file.

        Returns:
            True if save successful
        """
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)

            self.logger.info("Config", f"Configuration saved to {self.config_path}")
            return True

        except Exception as e:
            self.logger.error("Config", f"Failed to save config: {e}")
            return False

    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration.

        Returns:
            Camera configuration dict
        """
        return self.config.get("camera", {})

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration.

        Returns:
            Processing configuration dict
        """
        return self.config.get("processing", {})

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration.

        Returns:
            UI configuration dict
        """
        return self.config.get("app", {})

    def validate(self) -> bool:
        """Validate configuration values.

        Returns:
            True if configuration is valid
        """
        is_valid = True

        # Validate camera settings
        width = self.get("camera.width", 640)
        height = self.get("camera.height", 480)
        fps = self.get("camera.fps", 30)

        if not (320 <= width <= 3840):
            self.logger.warning("Config", f"Invalid width: {width}")
            is_valid = False

        if not (240 <= height <= 2160):
            self.logger.warning("Config", f"Invalid height: {height}")
            is_valid = False

        if not (1 <= fps <= 120):
            self.logger.warning("Config", f"Invalid FPS: {fps}")
            is_valid = False

        # Validate colors
        bg_color = self.get("processing.bg_color", [255, 255, 255])
        if not self._is_valid_color(bg_color):
            self.logger.warning("Config", f"Invalid bg_color: {bg_color}")
            is_valid = False

        return is_valid

    def _is_valid_color(self, color) -> bool:
        """Check if color value is valid RGB.

        Args:
            color: Color value to check

        Returns:
            True if valid
        """
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            return False

        return all(isinstance(c, int) and 0 <= c <= 255 for c in color)
