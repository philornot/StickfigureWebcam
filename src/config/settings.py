#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict

# Default application settings
DEFAULT_SETTINGS = {
    "camera": {"id": 0, "width": 640, "height": 480, "fps": 30, "flip_horizontal": True},
    "pose_detection": {
        "model_complexity": 1,
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.6,
        "smooth_landmarks": True,
    },
    "posture_analyzer": {
        "standing_hip_threshold": 0.7,
        "confidence_threshold": 0.6,
        "smoothing_factor": 0.7,
        "temporal_smoothing": 5,
    },
    "stick_figure": {
        "line_thickness": 3,
        "head_radius_factor": 0.075,
        "smooth_factor": 0.3,
        "smoothing_history": 3,
        "bg_color": [255, 255, 255],
        "figure_color": [0, 0, 0],
        "chair_color": [150, 75, 0],
    },
    "app": {"debug": False, "log_level": "INFO", "show_preview": True, "show_landmarks": True},
}

# Path to the settings file
SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "settings.json"
)


def get_settings() -> Dict[str, Any]:
    """
    Loads application settings. If the settings file doesn't exist,
    returns default settings.

    Returns:
        Dict[str, Any]: Application settings
    """
    # If settings file exists, load it
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)

            # Fill missing settings with defaults
            merged_settings = DEFAULT_SETTINGS.copy()
            _deep_update(merged_settings, settings)

            return merged_settings
        except Exception as e:
            print(f"Error loading settings: {e}")
            print("Using default settings...")
            return DEFAULT_SETTINGS.copy()
    else:
        # If file doesn't exist, return default settings
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> bool:
    """
    Saves settings to file.

    Args:
        settings (Dict[str, Any]): Settings to save

    Returns:
        bool: True if settings were saved successfully, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)

        # Save settings
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)

        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False


def _deep_update(target: Dict, source: Dict) -> Dict:
    """
    Deep updates target dictionary with values from source.

    Args:
        target (Dict): Target dictionary
        source (Dict): Source dictionary

    Returns:
        Dict: Updated target dictionary
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

    return target


def get_setting(section: str, key: str, default: Any = None) -> Any:
    """
    Gets a single setting from specified section.

    Args:
        section (str): Section name
        key (str): Setting key
        default (Any, optional): Default value if setting doesn't exist

    Returns:
        Any: Setting value or default value
    """
    settings = get_settings()

    if section in settings and key in settings[section]:
        return settings[section][key]

    return default


def update_setting(section: str, key: str, value: Any) -> bool:
    """
    Updates a single setting and saves changes.

    Args:
        section (str): Section name
        key (str): Setting key
        value (Any): New value

    Returns:
        bool: True if setting was updated successfully, False otherwise
    """
    settings = get_settings()

    if section not in settings:
        settings[section] = {}

    settings[section][key] = value

    return save_settings(settings)
