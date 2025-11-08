#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/conftest.py
"""
Test configuration for pytest.
This is a configuration file that is automatically loaded by pytest
and can contain common fixtures for tests.
"""

import os
import sys

import pytest

# Add main project directory to system path,
# so project modules are importable in tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_landmarks():
    """
    Fixture providing sample landmarks for tests.

    Returns:
        list: List of 33 landmarks in format (x, y, z, visibility)
    """
    # Create 33 points (MediaPipe Pose uses 33 points)
    landmarks = [(0, 0, 0, 0.5)] * 33

    # Basic points for tests
    # Format: (x, y, z, visibility) where x, y are values 0.0-1.0

    # Head
    landmarks[0] = (0.5, 0.1, 0, 0.9)  # NOSE
    landmarks[2] = (0.45, 0.08, 0, 0.9)  # LEFT_EYE
    landmarks[5] = (0.55, 0.08, 0, 0.9)  # RIGHT_EYE
    landmarks[7] = (0.4, 0.1, 0, 0.9)  # LEFT_EAR
    landmarks[8] = (0.6, 0.1, 0, 0.9)  # RIGHT_EAR

    # Shoulders
    landmarks[11] = (0.4, 0.2, 0, 0.9)  # LEFT_SHOULDER
    landmarks[12] = (0.6, 0.2, 0, 0.9)  # RIGHT_SHOULDER

    # Arms
    landmarks[13] = (0.3, 0.3, 0, 0.9)  # LEFT_ELBOW
    landmarks[14] = (0.7, 0.3, 0, 0.9)  # RIGHT_ELBOW
    landmarks[15] = (0.25, 0.4, 0, 0.9)  # LEFT_WRIST
    landmarks[16] = (0.75, 0.4, 0, 0.9)  # RIGHT_WRIST

    # Hips
    landmarks[23] = (0.45, 0.5, 0, 0.9)  # LEFT_HIP
    landmarks[24] = (0.55, 0.5, 0, 0.9)  # RIGHT_HIP

    # Legs
    landmarks[25] = (0.43, 0.7, 0, 0.9)  # LEFT_KNEE
    landmarks[26] = (0.57, 0.7, 0, 0.9)  # RIGHT_KNEE
    landmarks[27] = (0.42, 0.9, 0, 0.9)  # LEFT_ANKLE
    landmarks[28] = (0.58, 0.9, 0, 0.9)  # RIGHT_ANKLE

    return landmarks


@pytest.fixture
def sample_image():
    """
    Fixture providing sample image for tests.

    Returns:
        numpy.ndarray: Sample 640x480 RGB image
    """
    import numpy as np

    # Create empty 640x480 image with 3 channels (RGB)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # We can add some colors for better visualization
    # Image center - red rectangle
    image[200:300, 270:370, 0] = 200  # R channel

    return image


@pytest.fixture
def mock_logger():
    """
    Fixture providing mock logger for tests.

    Returns:
        MagicMock: Mock logger object
    """
    from unittest.mock import MagicMock

    logger = MagicMock()

    # Add logging methods
    logger.trace = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()

    return logger
