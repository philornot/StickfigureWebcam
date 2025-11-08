#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/drawing/pose_analyzer.py

from typing import Any, Dict, List, Optional, Tuple

from src.utils.custom_logger import CustomLogger


class PoseAnalyzer:
    """
    Uproszczona klasa do analizy pozy człowieka na podstawie punktów charakterystycznych.
    Koncentruje się na górnej części ciała (popiersiu).
    """

    # Indeksy punktów z MediaPipe Pose/FaceMesh
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    def __init__(self, sitting_threshold: float = 0.3, logger: Optional[CustomLogger] = None):
        """
        Inicjalizacja analizatora pozy.

        Args:
            sitting_threshold (float): Próg określający stosunek pozycji ramion.
                                      (pozostawiony dla kompatybilności)
            logger (Optional[CustomLogger]): Logger do zapisywania komunikatów
        """
        self.sitting_threshold = sitting_threshold
        self.logger = logger or CustomLogger()
        self.logger.debug("PoseAnalyzer", "Analizator pozy zainicjalizowany", log_type="POSE")

    def analyze_upper_body(
        self,
        landmarks: List[Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, Any]:
        """
        Analizuje górną część ciała (popiersie) na podstawie punktów charakterystycznych.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): Lista punktów (x, y, z, visibility)
            frame_width (int): Szerokość klatki
            frame_height (int): Wysokość klatki

        Returns:
            Dict[str, Any]: Słownik z informacjami o pozycji górnej części ciała
        """
        result = {
            "has_shoulders": False,
            "has_arms": False,
            "shoulder_positions": None,
            "elbow_positions": None,
            "wrist_positions": None,
            "shoulder_width_ratio": 0.0,
            "arms_extended": False,
            "left_arm_angle": 0.0,
            "right_arm_angle": 0.0,
        }

        try:
            if (
                landmarks is None or len(landmarks) < 17
            ):  # Potrzebujemy co najmniej punktów do nadgarstków
                return result

            # Sprawdzanie widoczności ramion
            left_shoulder_visible = self._is_point_visible(landmarks[self.LEFT_SHOULDER])
            right_shoulder_visible = self._is_point_visible(landmarks[self.RIGHT_SHOULDER])
            result["has_shoulders"] = left_shoulder_visible and right_shoulder_visible

            # Pozycje ramion (pikselowe)
            if result["has_shoulders"]:
                left_shoulder_pos = self._to_pixel_coords(
                    landmarks[self.LEFT_SHOULDER], frame_width, frame_height
                )
                right_shoulder_pos = self._to_pixel_coords(
                    landmarks[self.RIGHT_SHOULDER], frame_width, frame_height
                )
                result["shoulder_positions"] = (left_shoulder_pos, right_shoulder_pos)

                # Obliczenie szerokości ramion jako proporcji szerokości klatki
                shoulder_width = abs(right_shoulder_pos[0] - left_shoulder_pos[0])
                result["shoulder_width_ratio"] = shoulder_width / frame_width

            # Sprawdzanie widoczności łokci
            left_elbow_visible = self._is_point_visible(landmarks[self.LEFT_ELBOW])
            right_elbow_visible = self._is_point_visible(landmarks[self.RIGHT_ELBOW])
            left_wrist_visible = self._is_point_visible(landmarks[self.LEFT_WRIST])
            right_wrist_visible = self._is_point_visible(landmarks[self.RIGHT_WRIST])

            # Flaga określająca czy wykryto ramiona
            result["has_arms"] = left_elbow_visible or right_elbow_visible

            # Pozycje łokci (pikselowe)
            if left_elbow_visible or right_elbow_visible:
                elbow_positions = []
                if left_elbow_visible:
                    elbow_positions.append(
                        self._to_pixel_coords(landmarks[self.LEFT_ELBOW], frame_width, frame_height)
                    )
                else:
                    elbow_positions.append(None)

                if right_elbow_visible:
                    elbow_positions.append(
                        self._to_pixel_coords(
                            landmarks[self.RIGHT_ELBOW], frame_width, frame_height
                        )
                    )
                else:
                    elbow_positions.append(None)

                result["elbow_positions"] = tuple(elbow_positions)

            # Pozycje nadgarstków (pikselowe)
            if left_wrist_visible or right_wrist_visible:
                wrist_positions = []
                if left_wrist_visible:
                    wrist_positions.append(
                        self._to_pixel_coords(landmarks[self.LEFT_WRIST], frame_width, frame_height)
                    )
                else:
                    wrist_positions.append(None)

                if right_wrist_visible:
                    wrist_positions.append(
                        self._to_pixel_coords(
                            landmarks[self.RIGHT_WRIST], frame_width, frame_height
                        )
                    )
                else:
                    wrist_positions.append(None)

                result["wrist_positions"] = tuple(wrist_positions)

            # Sprawdzanie czy ręce są wyciągnięte (np. ręce uniesione)
            # Tylko jeśli mamy zarówno ramiona jak i łokcie
            if (
                result["has_shoulders"]
                and result["has_arms"]
                and result["shoulder_positions"] is not None
                and result["elbow_positions"] is not None
            ):

                left_extended = False
                right_extended = False

                # Lewe ramię
                if left_elbow_visible and left_shoulder_visible:
                    left_shoulder = landmarks[self.LEFT_SHOULDER]
                    left_elbow = landmarks[self.LEFT_ELBOW]

                    # Kąt między ramieniem a pionem
                    left_angle = self._calculate_vertical_angle(left_shoulder, left_elbow)
                    result["left_arm_angle"] = left_angle

                    # Jeśli kąt jest większy niż 45 stopni, ramię jest wyciągnięte
                    left_extended = left_angle > 45

                # Prawe ramię
                if right_elbow_visible and right_shoulder_visible:
                    right_shoulder = landmarks[self.RIGHT_SHOULDER]
                    right_elbow = landmarks[self.RIGHT_ELBOW]

                    # Kąt między ramieniem a pionem
                    right_angle = self._calculate_vertical_angle(right_shoulder, right_elbow)
                    result["right_arm_angle"] = right_angle

                    # Jeśli kąt jest większy niż 45 stopni, ramię jest wyciągnięte
                    right_extended = right_angle > 45

                result["arms_extended"] = left_extended or right_extended

            if self.logger and result["has_shoulders"]:
                self.logger.debug(
                    "PoseAnalyzer",
                    f"Górna część ciała wykryta: ramiona={result['has_shoulders']}, "
                    f"ręce={result['has_arms']}, uniesione={result['arms_extended']}",
                    log_type="POSE",
                )

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "PoseAnalyzer",
                    f"Błąd podczas analizy górnej części ciała: {str(e)}",
                    log_type="POSE",
                    error={"error": str(e)},
                )

        return result

    def _is_point_visible(
        self, point: Tuple[float, float, float, float], threshold: float = 0.5
    ) -> bool:
        """
        Sprawdza, czy punkt jest wystarczająco widoczny.

        Args:
            point (Tuple[float, float, float, float]): Punkt (x, y, z, visibility)
            threshold (float): Próg widoczności (0.0-1.0)

        Returns:
            bool: True jeśli punkt jest widoczny powyżej progu
        """
        return point[3] >= threshold

    def _to_pixel_coords(
        self, point: Tuple[float, float, float, float], frame_width: int, frame_height: int
    ) -> Tuple[int, int]:
        """
        Konwertuje znormalizowane współrzędne punktu (0.0-1.0) na współrzędne pikselowe.

        Args:
            point (Tuple[float, float, float, float]): Punkt (x, y, z, visibility)
            frame_width (int): Szerokość klatki
            frame_height (int): Wysokość klatki

        Returns:
            Tuple[int, int]: Współrzędne pikselowe (x, y)
        """
        return (int(point[0] * frame_width), int(point[1] * frame_height))

    def _calculate_vertical_angle(
        self, point1: Tuple[float, float, float, float], point2: Tuple[float, float, float, float]
    ) -> float:
        """
        Oblicza kąt między linią łączącą dwa punkty a pionem.

        Args:
            point1 (Tuple[float, float, float, float]): Pierwszy punkt (x, y, z, visibility)
            point2 (Tuple[float, float, float, float]): Drugi punkt (x, y, z, visibility)

        Returns:
            float: Kąt w stopniach (0-90)
        """
        import math

        # Punkty są już znormalizowane (0.0-1.0)
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        # Wektor od punktu 1 do punktu 2
        vector_x = x2 - x1
        vector_y = y2 - y1

        # Jeśli wektor jest zerowy, zwracamy 0
        if vector_x == 0 and vector_y == 0:
            return 0.0

        # Pionem jest wektor (0, 1)
        # Kąt między wektorami: cos(θ) = (u·v) / (|u|·|v|)
        # Gdzie u·v to iloczyn skalarny, a |u| i |v| to długości wektorów

        # Iloczyn skalarny
        dot_product = vector_y  # dot product z wektorem (0, 1) to po prostu vector_y

        # Długości wektorów
        vector_length = math.sqrt(vector_x**2 + vector_y**2)
        vertical_length = 1.0  # długość wektora (0, 1)

        # Kosinus kąta
        cos_angle = dot_product / (vector_length * vertical_length)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Ograniczenie do zakresu [-1, 1]

        # Kąt w radianach, a następnie w stopniach
        angle_rad = math.acos(cos_angle)
        angle_deg = angle_rad * 180 / math.pi

        return angle_deg
