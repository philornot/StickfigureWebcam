"""
Face detection and mouth openness detection module.

This module handles all face-related detection using MediaPipe Face Mesh,
including mouth state detection based on landmark positions.
"""

import cv2
import config


def calculate_mouth_openness(face_landmarks, width, height):
    """
    Calculate if mouth is open based on Face Mesh landmarks.

    Uses multiple landmark points for more accurate detection:
    - Upper lip: points 13, 14
    - Lower lip: points 78, 308, 87, 317
    - Calculates vertical distance and compares to face height ratio

    Args:
        face_landmarks: MediaPipe face mesh landmarks
        width: Frame width in pixels
        height: Frame height in pixels

    Returns:
        bool: True if mouth is open, False otherwise
    """
    if not face_landmarks:
        return False

    landmarks = config.MOUTH_LANDMARKS

    try:
        # Get multiple points for more accurate measurement
        upper_lip_top = face_landmarks[landmarks['upper_lip_top']]
        lower_lip_bottom = face_landmarks[landmarks['lower_lip_bottom']]
        upper_outer_1 = face_landmarks[landmarks['upper_outer_1']]
        upper_outer_2 = face_landmarks[landmarks['upper_outer_2']]
        lower_outer_1 = face_landmarks[landmarks['lower_outer_1']]
        lower_outer_2 = face_landmarks[landmarks['lower_outer_2']]

        # Convert to pixel coordinates
        upper_y = upper_lip_top.y * height
        lower_y = lower_lip_bottom.y * height

        # Calculate mouth opening distance
        mouth_distance = abs(lower_y - upper_y)

        # Calculate face height for relative threshold
        forehead = face_landmarks[landmarks['forehead']]
        chin = face_landmarks[landmarks['chin']]
        face_height = abs((chin.y - forehead.y) * height)

        # Adaptive threshold based on face size
        threshold = face_height * config.MOUTH_OPEN_THRESHOLD_RATIO

        return mouth_distance > threshold

    except Exception as e:
        print(f"Error calculating mouth openness: {e}")
        return False


def draw_face_landmarks(canvas, face_landmarks, width, height):
    """
    Draw face mesh landmarks for debugging.

    Highlights key facial landmarks used for mouth detection in different colors:
    - Red: Mouth landmarks
    - Green: Face reference points (forehead, chin)

    Args:
        canvas: Numpy array to draw on
        face_landmarks: MediaPipe face mesh landmarks
        width: Canvas width
        height: Canvas height
    """
    if not face_landmarks:
        return

    landmarks = config.MOUTH_LANDMARKS

    # All key points to visualize
    key_points = [
        landmarks['upper_lip_top'],
        landmarks['lower_lip_bottom'],
        landmarks['upper_outer_1'],
        landmarks['upper_outer_2'],
        landmarks['lower_outer_1'],
        landmarks['lower_outer_2'],
        landmarks['forehead'],
        landmarks['chin']
    ]

    mouth_points = [
        landmarks['upper_lip_top'],
        landmarks['lower_lip_bottom'],
        landmarks['upper_outer_1'],
        landmarks['upper_outer_2'],
        landmarks['lower_outer_1'],
        landmarks['lower_outer_2']
    ]

    try:
        for idx in key_points:
            landmark = face_landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # Color code: red for mouth points, green for face reference points
            if idx in mouth_points:
                color = (0, 0, 255)  # Red for mouth
            else:
                color = (0, 255, 0)  # Green for reference points

            cv2.circle(canvas, (x, y), 2, color, -1)
            cv2.putText(canvas, str(idx), (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

    except Exception as e:
        print(f"Error drawing face landmarks: {e}")