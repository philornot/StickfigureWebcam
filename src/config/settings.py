#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, Any

# Domyślne ustawienia aplikacji
DEFAULT_SETTINGS = {
    "camera": {
        "id": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "flip_horizontal": True
    },
    "pose_detection": {
        "model_complexity": 1,
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.6,
        "smooth_landmarks": True
    },
    "posture_analyzer": {
        "standing_hip_threshold": 0.7,
        "confidence_threshold": 0.6,
        "smoothing_factor": 0.7,
        "temporal_smoothing": 5
    },
    "stick_figure": {
        "line_thickness": 3,
        "head_radius_factor": 0.075,
        "smooth_factor": 0.3,
        "smoothing_history": 3,
        "bg_color": [255, 255, 255],
        "figure_color": [0, 0, 0],
        "chair_color": [150, 75, 0]
    },
    "app": {
        "debug": False,
        "log_level": "INFO",
        "show_preview": True,
        "show_landmarks": True
    }
}

# Ścieżka do pliku z zapisanymi ustawieniami
SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "settings.json"
)


def get_settings() -> Dict[str, Any]:
    """
    Wczytuje ustawienia aplikacji. Jeśli plik ustawień nie istnieje,
    zwraca domyślne ustawienia.

    Returns:
        Dict[str, Any]: Ustawienia aplikacji
    """
    # Jeśli plik ustawień istnieje, wczytaj go
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            # Uzupełnij brakujące ustawienia domyślnymi
            merged_settings = DEFAULT_SETTINGS.copy()
            _deep_update(merged_settings, settings)

            return merged_settings
        except Exception as e:
            print(f"Błąd podczas wczytywania ustawień: {e}")
            print("Używanie domyślnych ustawień...")
            return DEFAULT_SETTINGS.copy()
    else:
        # Jeśli plik nie istnieje, zwróć domyślne ustawienia
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> bool:
    """
    Zapisuje ustawienia do pliku.

    Args:
        settings (Dict[str, Any]): Ustawienia do zapisania

    Returns:
        bool: True jeśli udało się zapisać ustawienia, False w przeciwnym przypadku
    """
    try:
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)

        # Zapisz ustawienia
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)

        return True
    except Exception as e:
        print(f"Błąd podczas zapisywania ustawień: {e}")
        return False


def _deep_update(target: Dict, source: Dict) -> Dict:
    """
    Głębokie aktualizowanie słownika target wartościami z source.

    Args:
        target (Dict): Słownik docelowy
        source (Dict): Słownik źródłowy

    Returns:
        Dict: Zaktualizowany słownik target
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

    return target


def get_setting(section: str, key: str, default: Any = None) -> Any:
    """
    Pobiera pojedyncze ustawienie z określonej sekcji.

    Args:
        section (str): Nazwa sekcji
        key (str): Klucz ustawienia
        default (Any, optional): Wartość domyślna jeśli ustawienie nie istnieje

    Returns:
        Any: Wartość ustawienia lub wartość domyślna
    """
    settings = get_settings()

    if section in settings and key in settings[section]:
        return settings[section][key]

    return default


def update_setting(section: str, key: str, value: Any) -> bool:
    """
    Aktualizuje pojedyncze ustawienie i zapisuje zmiany.

    Args:
        section (str): Nazwa sekcji
        key (str): Klucz ustawienia
        value (Any): Nowa wartość

    Returns:
        bool: True jeśli udało się zaktualizować ustawienie, False w przeciwnym przypadku
    """
    settings = get_settings()

    if section not in settings:
        settings[section] = {}

    settings[section][key] = value

    return save_settings(settings)
