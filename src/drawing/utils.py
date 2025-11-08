#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# drawing/utils.py

from typing import List, Optional, Tuple


def get_midpoint(
    p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    """
    Safely calculates the midpoint between two points.

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        Midpoint (x, y) or None if points unavailable
    """
    if p1 is None or p2 is None:
        return None

    try:
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        return (int((x1 + x2) // 2), int((y1 + y2) // 2))
    except (TypeError, IndexError, ValueError):
        return None


def smooth_points(
    current_points: List[Tuple[float, float, float, float]],
    history: List[List[Tuple[float, float, float, float]]],
    history_length: int,
    smooth_factor: float,
) -> List[Tuple[float, float, float, float]]:
    """
    Smooths landmark point movement between frames.

    Args:
        current_points: Current points
        history: History of previous points
        history_length: Maximum history length
        smooth_factor: Smoothing factor (0.0-1.0)

    Returns:
        Smoothed points
    """
    # If no history, return current points
    if not history or len(history) == 0:
        return current_points

    # Smooth points
    smoothed_points = []

    # For each point
    for i in range(len(current_points)):
        # Collect the same point from previous frames
        point_history = []
        for frame in history:
            if i < len(frame):
                point_history.append(frame[i])

        # If no history for this point, use current value
        if not point_history:
            smoothed_points.append(current_points[i])
            continue

        # Calculate smoothed value with higher weight for newer points
        x_sum, y_sum, z_sum, vis_sum = 0.0, 0.0, 0.0, 0.0
        weight_sum = 0.0

        for j, (x, y, z, vis) in enumerate(point_history):
            # Higher weight for newer points
            weight = j + 1
            weight_sum += weight

            x_sum += x * weight
            y_sum += y * weight
            z_sum += z * weight
            vis_sum += vis * weight

        # Normalize by weight sum
        if weight_sum > 0:
            x_smooth = x_sum / weight_sum
            y_smooth = y_sum / weight_sum
            z_smooth = z_sum / weight_sum
            vis_smooth = vis_sum / weight_sum

            # Apply smoothing factor (higher = more smoothed movement)
            x_final = current_points[i][0] * (1 - smooth_factor) + x_smooth * smooth_factor
            y_final = current_points[i][1] * (1 - smooth_factor) + y_smooth * smooth_factor
            z_final = current_points[i][2] * (1 - smooth_factor) + z_smooth * smooth_factor
            vis_final = current_points[i][3] * (1 - smooth_factor) + vis_smooth * smooth_factor

            smoothed_points.append((x_final, y_final, z_final, vis_final))
        else:
            smoothed_points.append(current_points[i])

    return smoothed_points


def calculate_visibility(landmarks: List[Tuple[float, float, float, float]], index: int) -> float:
    """
    Returns the visibility of a given landmark point.

    Args:
        landmarks: List of landmark points
        index: Point index

    Returns:
        Visibility value (0.0-1.0) or 0.0 if point unavailable
    """
    if landmarks is None or index >= len(landmarks):
        return 0.0

    try:
        return landmarks[index][3]  # Fourth element is visibility
    except (IndexError, TypeError):
        return 0.0


def normalize_coordinates(
    x: float, y: float, canvas_width: int, canvas_height: int
) -> Tuple[int, int]:
    """
    Converts normalized coordinates (0.0-1.0) to pixel coordinates.

    Args:
        x: Normalized X coordinate
        y: Normalized Y coordinate
        canvas_width: Target image width
        canvas_height: Target image height

    Returns:
        Pixel coordinates (x, y)
    """
    return (int(x * canvas_width), int(y * canvas_height))


def is_point_visible(
    landmarks: List[Tuple[float, float, float, float]], index: int, threshold: float = 0.5
) -> bool:
    """
    Checks if a given point is sufficiently visible.

    Args:
        landmarks: List of landmark points
        index: Point index
        threshold: Visibility threshold (0.0-1.0)

    Returns:
        True if point visibility is above threshold, False otherwise
    """
    visibility = calculate_visibility(landmarks, index)
    return visibility >= threshold
