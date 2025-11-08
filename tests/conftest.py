#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/conftest.py
"""
Konfiguracja testów dla pytest.
To jest plik konfiguracyjny, który jest automatycznie ładowany przez pytest
i może zawierać wspólne fixtures dla testów.
"""

import os
import sys

import pytest

# Dodajemy główny katalog projektu do ścieżki systemowej,
# aby moduły projektu były importowalne w testach
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_landmarks():
    """
    Fixture dostarczający przykładowe punkty charakterystyczne dla testów.

    Returns:
        list: Lista 33 punktów charakterystycznych w formacie (x, y, z, visibility)
    """
    # Tworzymy 33 punkty (MediaPipe Pose używa 33 punktów)
    landmarks = [(0, 0, 0, 0.5)] * 33

    # Podstawowe punkty dla testów
    # Format: (x, y, z, visibility) gdzie x, y to wartości 0.0-1.0

    # Głowa
    landmarks[0] = (0.5, 0.1, 0, 0.9)  # NOSE
    landmarks[2] = (0.45, 0.08, 0, 0.9)  # LEFT_EYE
    landmarks[5] = (0.55, 0.08, 0, 0.9)  # RIGHT_EYE
    landmarks[7] = (0.4, 0.1, 0, 0.9)  # LEFT_EAR
    landmarks[8] = (0.6, 0.1, 0, 0.9)  # RIGHT_EAR

    # Ramiona
    landmarks[11] = (0.4, 0.2, 0, 0.9)  # LEFT_SHOULDER
    landmarks[12] = (0.6, 0.2, 0, 0.9)  # RIGHT_SHOULDER

    # Ręce
    landmarks[13] = (0.3, 0.3, 0, 0.9)  # LEFT_ELBOW
    landmarks[14] = (0.7, 0.3, 0, 0.9)  # RIGHT_ELBOW
    landmarks[15] = (0.25, 0.4, 0, 0.9)  # LEFT_WRIST
    landmarks[16] = (0.75, 0.4, 0, 0.9)  # RIGHT_WRIST

    # Biodra
    landmarks[23] = (0.45, 0.5, 0, 0.9)  # LEFT_HIP
    landmarks[24] = (0.55, 0.5, 0, 0.9)  # RIGHT_HIP

    # Nogi
    landmarks[25] = (0.43, 0.7, 0, 0.9)  # LEFT_KNEE
    landmarks[26] = (0.57, 0.7, 0, 0.9)  # RIGHT_KNEE
    landmarks[27] = (0.42, 0.9, 0, 0.9)  # LEFT_ANKLE
    landmarks[28] = (0.58, 0.9, 0, 0.9)  # RIGHT_ANKLE

    return landmarks


@pytest.fixture
def sample_image():
    """
    Fixture dostarczający przykładowy obraz dla testów.

    Returns:
        numpy.ndarray: Przykładowy obraz 640x480 RGB
    """
    import numpy as np

    # Tworzymy pusty obraz 640x480 z 3 kanałami (RGB)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Możemy dodać trochę kolorów dla lepszej wizualizacji
    # Środek obrazu - czerwony prostokąt
    image[200:300, 270:370, 0] = 200  # kanał R

    return image


@pytest.fixture
def mock_logger():
    """
    Fixture dostarczający mock loggera dla testów.

    Returns:
        MagicMock: Mock obiektu loggera
    """
    from unittest.mock import MagicMock

    logger = MagicMock()

    # Dodajemy metody logowania
    logger.trace = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()

    return logger
