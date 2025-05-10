#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# drawing/utils.py

from typing import List, Tuple, Optional


def get_midpoint(p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """
    Bezpiecznie oblicza punkt środkowy między dwoma punktami.

    Args:
        p1: Pierwszy punkt (x, y)
        p2: Drugi punkt (x, y)

    Returns:
        Optional[Tuple[int, int]]: Punkt środkowy (x, y) lub None jeśli punkty niedostępne
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
    smooth_factor: float
) -> List[Tuple[float, float, float, float]]:
    """
    Wygładza ruch punktów charakterystycznych między klatkami.

    Args:
        current_points (List[Tuple[float, float, float, float]]): Aktualne punkty
        history (List[List[Tuple[float, float, float, float]]]): Historia poprzednich punktów
        history_length (int): Maksymalna długość historii
        smooth_factor (float): Współczynnik wygładzania (0.0-1.0)

    Returns:
        List[Tuple[float, float, float, float]]: Wygładzone punkty
    """
    # Jeśli nie ma historii, zwracamy bieżące punkty
    if not history or len(history) == 0:
        return current_points

    # Wygładzanie punktów
    smoothed_points = []

    # Dla każdego punktu
    for i in range(len(current_points)):
        # Zbierz ten sam punkt z poprzednich klatek
        point_history = []
        for frame in history:
            if i < len(frame):
                point_history.append(frame[i])

        # Jeśli nie ma historii dla tego punktu, użyj bieżącej wartości
        if not point_history:
            smoothed_points.append(current_points[i])
            continue

        # Oblicz wygładzoną wartość z większą wagą dla nowszych punktów
        x_sum, y_sum, z_sum, vis_sum = 0.0, 0.0, 0.0, 0.0
        weight_sum = 0.0

        for j, (x, y, z, vis) in enumerate(point_history):
            # Większa waga dla nowszych punktów
            weight = j + 1
            weight_sum += weight

            x_sum += x * weight
            y_sum += y * weight
            z_sum += z * weight
            vis_sum += vis * weight

        # Normalizuj przez sumę wag
        if weight_sum > 0:
            x_smooth = x_sum / weight_sum
            y_smooth = y_sum / weight_sum
            z_smooth = z_sum / weight_sum
            vis_smooth = vis_sum / weight_sum

            # Zastosuj współczynnik wygładzania (im większy, tym bardziej wygładzony ruch)
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
    Zwraca widoczność danego punktu charakterystycznego.

    Args:
        landmarks (List[Tuple[float, float, float, float]]): Lista punktów charakterystycznych
        index (int): Indeks punktu

    Returns:
        float: Wartość widoczności (0.0-1.0) lub 0.0 jeśli punkt niedostępny
    """
    if landmarks is None or index >= len(landmarks):
        return 0.0

    try:
        return landmarks[index][3]  # Czwarty element to widoczność
    except (IndexError, TypeError):
        return 0.0


def normalize_coordinates(
    x: float,
    y: float,
    canvas_width: int,
    canvas_height: int
) -> Tuple[int, int]:
    """
    Konwertuje znormalizowane współrzędne (0.0-1.0) na współrzędne pikselowe.

    Args:
        x (float): Znormalizowana współrzędna X
        y (float): Znormalizowana współrzędna Y
        canvas_width (int): Szerokość obrazu docelowego
        canvas_height (int): Wysokość obrazu docelowego

    Returns:
        Tuple[int, int]: Współrzędne pikselowe (x, y)
    """
    return (int(x * canvas_width), int(y * canvas_height))


def is_point_visible(landmarks: List[Tuple[float, float, float, float]], index: int, threshold: float = 0.5) -> bool:
    """
    Sprawdza, czy dany punkt jest wystarczająco widoczny.

    Args:
        landmarks (List[Tuple[float, float, float, float]]): Lista punktów charakterystycznych
        index (int): Indeks punktu
        threshold (float): Próg widoczności (0.0-1.0)

    Returns:
        bool: True jeśli punkt jest widoczny powyżej progu, False w przeciwnym razie
    """
    visibility = calculate_visibility(landmarks, index)
    return visibility >= threshold
