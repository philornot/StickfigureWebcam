"""
Face detection and mouth/eye state detection module.

This module handles face-related detection using MediaPipe Face Mesh,
including mouth state and eye blink detection based on landmark positions.
"""

import math

import cv2

import config


def _euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two 2D points.

    Args:
        point1 (tuple): Coordinates of the first point (x, y).
        point2 (tuple): Coordinates of the second point (x, y).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_mouth_openness(face_landmarks, width, height):
    """Calculate if mouth is open based on Face Mesh landmarks.

    Args:
        face_landmarks: MediaPipe face mesh landmarks.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        bool: True if mouth is open, False otherwise.
    """
    if not face_landmarks:
        return False

    landmarks = config.MOUTH_LANDMARKS

    try:
        # Cache pixel conversions
        upper_y = face_landmarks[landmarks['upper_lip_top']].y * height
        lower_y = face_landmarks[landmarks['lower_lip_bottom']].y * height
        forehead_y = face_landmarks[landmarks['forehead']].y * height
        chin_y = face_landmarks[landmarks['chin']].y * height

        # Calculate mouth opening distance
        mouth_distance = abs(lower_y - upper_y)

        # Calculate face height for relative threshold
        face_height = abs(chin_y - forehead_y)
        if face_height == 0:
            return False

        # Adaptive threshold based on face size
        threshold = face_height * config.MOUTH_OPEN_THRESHOLD_RATIO

        return mouth_distance > threshold

    except (IndexError, AttributeError):
        return False


def calculate_eye_aspect_ratio(face_landmarks, width, height):
    """Calculate a simplified eye aspect ratio (EAR) based on vertical distances.

    This ratio is normalized by face height to be robust to scale changes.
    A smaller ratio indicates a closed eye.

    Args:
        face_landmarks: MediaPipe face mesh landmarks.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        float: A normalized ratio of eye openness. Returns a high value (open)
               on error or if no landmarks are found.
    """
    if not face_landmarks:
        return 1.0  # Default to open if no landmarks

    try:
        # Coordinate getter with caching
        def get_coords(idx):
            lm = face_landmarks[idx]
            return int(lm.x * width), int(lm.y * height)

        # Get all required points at once
        left_eye_top = get_coords(config.LEFT_EYE_TOP)
        left_eye_bottom = get_coords(config.LEFT_EYE_BOTTOM)
        right_eye_top = get_coords(config.RIGHT_EYE_TOP)
        right_eye_bottom = get_coords(config.RIGHT_EYE_BOTTOM)
        forehead = get_coords(config.MOUTH_LANDMARKS['forehead'])
        chin = get_coords(config.MOUTH_LANDMARKS['chin'])

        # Calculate vertical distances
        left_eye_dist = _euclidean_distance(left_eye_top, left_eye_bottom)
        right_eye_dist = _euclidean_distance(right_eye_top, right_eye_bottom)

        # Average vertical eye distance
        avg_eye_dist = (left_eye_dist + right_eye_dist) * 0.5

        # Calculate face height
        face_height = _euclidean_distance(forehead, chin)
        if face_height == 0:
            return 1.0  # Default to open

        # Normalize eye distance by face height
        return avg_eye_dist / face_height

    except (IndexError, AttributeError, ZeroDivisionError):
        return 1.0  # Default to open on error


def draw_face_landmarks(canvas, face_landmarks, width, height):
    """Draw face mesh landmarks for debugging.

    Highlights key facial landmarks used for mouth and eye detection.

    Args:
        canvas: Numpy array to draw on.
        face_landmarks: MediaPipe face mesh landmarks.
        width: Canvas width.
        height: Canvas height.
    """
    if not face_landmarks:
        return

    # Key points for mouth detection
    mouth_points = [
        config.MOUTH_LANDMARKS['upper_lip_top'],
        config.MOUTH_LANDMARKS['lower_lip_bottom'],
        config.MOUTH_LANDMARKS['forehead'],
        config.MOUTH_LANDMARKS['chin']
    ]

    # Key points for eye detection
    eye_points = [
        config.LEFT_EYE_TOP,
        config.LEFT_EYE_BOTTOM,
        config.RIGHT_EYE_TOP,
        config.RIGHT_EYE_BOTTOM
    ]

    try:
        # Batch convert mouth points
        for idx in mouth_points:
            landmark = face_landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)

        # Batch convert eye points
        for idx in eye_points:
            landmark = face_landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(canvas, (x, y), 2, (255, 0, 0), -1)

    except (IndexError, AttributeError):
        pass  # Silently skip on error
